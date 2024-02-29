import os
import time
import copy
import torch
import pprint
import logging
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import multiprocessing as mp
from tqdm import tqdm
from collections import defaultdict, OrderedDict

from runner import build_vocab, build_vocab_from_pkl
from runner import build_dataloader, build_model
from runner import build_criterion
from utils import TestOptions, AverageMeter
from utils import PostProcessorDETR
from utils import count_parameters, merge_state_dict_with_module
from utils import span_cxw_to_xx 
from utils import save_json, save_jsonl
from utils import compute_temporal_iou_batch_cross
from utils import interpolated_precision_recall, compute_temporal_iou_batch_paired
from utils import get_window_len, temporal_nms
from dataset import prepare_batch_input


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def eval_epoch(model, eval_loader, opt, save_submission_filename, epoch_i=None, criterion=None, tb_writer=None):
    logger.info("Generate submissions")
    submission, eval_loss_meters = get_eval_res(model, eval_loader, opt, epoch_i, criterion, tb_writer)
    
    if not opt.sort_results:
        save_submission_filename = save_submission_filename.replace(".jsonl", "_unsorted.jsonl")
    metrics, metrics_nms, latest_file_paths = eval_epoch_post_processing(
        submission, opt, eval_loader.dataset.data, save_submission_filename)
    return metrics, metrics_nms, eval_loss_meters, latest_file_paths


def get_eval_res(model, eval_loader, opt, epoch_i, criterion, tb_writer):
    """compute and save query and video proposal embeddings"""
    eval_res, eval_loss_meters = compute_mr_results(model, eval_loader, opt, epoch_i, criterion, tb_writer)  # list(dict)
    return eval_res, eval_loss_meters


@torch.no_grad()
def compute_mr_results(model, eval_loader, opt, epoch_i=None, criterion=None, tb_writer=None):
    model.eval()
    if criterion is not None:
        criterion.eval()

    loss_meters = defaultdict(AverageMeter)
    write_tb = tb_writer is not None and epoch_i is not None

    mr_res = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        prepare_batch_input(batch, opt.device, non_blocking=opt.pin_memory)
        outputs = model(**batch, dataset_name=opt.dataset_name, is_training=False)
        prob = F.softmax(outputs["pred_logits"], -1)  # (batch_size, #queries, #classes=2)
        if opt.span_loss_type == "l1":
            scores = prob[..., 0]  # * (batch_size, #queries)  foreground label is 0, we directly take it
            pred_spans = outputs["pred_spans"]  # (bsz, #queries, 2)
            _saliency_scores = outputs["saliency_scores"].half()  # (bsz, L)
            saliency_scores = []
            valid_vid_lengths = batch["video_mask"].sum(1).cpu().tolist()
            for j in range(len(valid_vid_lengths)):
                saliency_scores.append(_saliency_scores[j, :int(valid_vid_lengths[j])].tolist())
        else:
            raise NotImplementedError
            bsz, n_queries = outputs["pred_spans"].shape[:2]  # # (bsz, #queries, max_v_l *2)
            pred_spans_logits = outputs["pred_spans"].view(bsz, n_queries, 2, opt.max_video_l)
            # TODO use more advanced decoding method with st_ed product
            pred_span_scores, pred_spans = F.softmax(pred_spans_logits, dim=-1).max(-1)  # 2 * (bsz, #queries, 2)
            scores = torch.prod(pred_span_scores, 2)  # (bsz, #queries)
            pred_spans[:, 1] += 1
            pred_spans *= opt.clip_len

        # compose predictions
        for idx, (spans, score) in enumerate(zip(pred_spans, scores)):
            if opt.span_loss_type == "l1":
                spans = span_cxw_to_xx(spans) * batch["duration"][idx]
            # # (#queries, 3), [st(float), ed(float), score(float)]
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).cpu().tolist()
            if opt.sort_results:
                cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
            cur_query_pred = dict(
                qid=batch["qid"][idx],
                query=batch["sentence"][idx],
                vid=batch["video_id"][idx],
                pred_relevant_windows=cur_ranked_preds,
                pred_saliency_scores=saliency_scores[idx]
            )
            mr_res.append(cur_query_pred)

        if criterion:
            loss_dict, loss = criterion(outputs, batch, is_training=False)
            loss_dict["loss_overall"] = float(loss)  # for logging only
            for k, v in loss_dict.items():
                loss_meters[k].update(float(v) * criterion.weight_dict[k] if k in criterion.weight_dict else float(v))

    if write_tb and criterion:
        for k, v in loss_meters.items():
            tb_writer.add_scalar("Eval/{}".format(k), v.avg, epoch_i + 1)

    post_processor = PostProcessorDETR(
        clip_length=opt.clip_len, min_ts_val=0, max_ts_val=opt.max_ts_val,
        min_w_l=2, max_w_l=150, move_window_method="left",
        process_func_names=("clip_ts", "round_multiple") if opt.clip_len != -1 else ("clip_ts",)
    )
    mr_res = post_processor(mr_res)
    return mr_res, loss_meters


def eval_epoch_post_processing(submission, opt, gt_data, save_submission_filename):
    # IOU_THDS = (0.5, 0.7)
    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(opt.result_dir, save_submission_filename)
    save_jsonl(submission, submission_path)

    # if opt.eval_split_name in ["val"]:  # since test_public has no GT
    # if opt.dataset_name in ["qvhighlights"] and opt.is_inference:
    #     return None, None, None
    
    metrics = eval_submission(
        submission, gt_data,
        # verbose=opt.debug, match_number=not opt.debug,
        verbose=False, match_number=True,
        dataset_name=opt.dataset_name
    )
    save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")
    save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
    latest_file_paths = [submission_path, save_metrics_path]

    if opt.nms_thd != -1:
        logger.info("[MR] Performing nms with nms_thd {}".format(opt.nms_thd))
        submission_after_nms = post_processing_mr_nms(
            submission, nms_thd=opt.nms_thd,
            max_before_nms=opt.max_before_nms, max_after_nms=opt.max_after_nms
        )

        logger.info("Saving/Evaluating nms results")
        submission_nms_path = submission_path.replace(".jsonl", "_nms_thd_{}.jsonl".format(opt.nms_thd))
        save_jsonl(submission_after_nms, submission_nms_path)

        metrics_nms = eval_submission(
            submission_after_nms, gt_data,
            verbose=False, match_number=True
        )
        save_metrics_nms_path = submission_nms_path.replace(".jsonl", "_metrics.json")
        save_json(metrics_nms, save_metrics_nms_path, save_pretty=True, sort_keys=False)
        latest_file_paths += [submission_nms_path, save_metrics_nms_path]

    else:
        metrics_nms = None
    return metrics, metrics_nms, latest_file_paths


def eval_submission(submission, ground_truth, verbose=True, match_number=True, dataset_name="charades"):
    """
    Args:
        submission: list(dict), each dict is {
            qid: str,
            query: str,
            vid: str,
            pred_relevant_windows: list([st, ed]),
            pred_saliency_scores: list(float), len == #clips in video.
                i.e., each clip in the video will have a saliency score.
        }
        ground_truth: list(dict), each dict is     {
          "qid": 7803,
          "query": "Man in gray top walks from outside to inside.",
          "duration": 150,
          "vid": "RoripwjYFp8_360.0_510.0",
          "relevant_clip_ids": [13, 14, 15, 16, 17]
          "saliency_scores": [[4, 4, 2], [3, 4, 2], [2, 2, 3], [2, 2, 2], [0, 1, 3]]
               each sublist corresponds to one clip in relevant_clip_ids.
               The 3 elements in the sublist are scores from 3 different workers. The
               scores are in [0, 1, 2, 3, 4], meaning [Very Bad, ..., Good, Very Good]
        }
        verbose:
        match_number:

    Returns:

    """
    # pred_qids = set([e["qid"] for e in submission])
    # gt_qids = set([e["qid"] for e in ground_truth])
    # if match_number:
    #     assert pred_qids == gt_qids, \
    #         f"qids in ground_truth and submission must match. " \
    #         f"use `match_number=False` if you wish to disable this check"
    # else:  # only leave the items that exists in both submission and ground_truth
    #     shared_qids = pred_qids.intersection(gt_qids)
    #     submission = [e for e in submission if e["qid"] in shared_qids]
    #     ground_truth = [e for e in ground_truth if e["qid"] in shared_qids]

    eval_metrics = {}
    eval_metrics_brief = OrderedDict()
    if "pred_relevant_windows" in submission[0]:
        moment_ret_scores = eval_moment_retrieval(
            submission, ground_truth, verbose=verbose, dataset_name=dataset_name)
        eval_metrics.update(moment_ret_scores)
        moment_ret_scores_brief = {
            "MR-full-R1@0.3": moment_ret_scores["full"]["MR-R1"]["0.3"],
            "MR-full-R1@0.5": moment_ret_scores["full"]["MR-R1"]["0.5"],
            "MR-full-R1@0.7": moment_ret_scores["full"]["MR-R1"]["0.7"],
            "MR-full-miou": moment_ret_scores["full"]["MR-R1"]["miou"],
        }
        eval_metrics_brief.update(
            sorted([(k, v) for k, v in moment_ret_scores_brief.items()], key=lambda x: x[0]))

    # sort by keys
    final_eval_metrics = OrderedDict()
    final_eval_metrics["brief"] = eval_metrics_brief
    final_eval_metrics.update(sorted([(k, v) for k, v in eval_metrics.items()], key=lambda x: x[0]))
    return final_eval_metrics


def eval_moment_retrieval(submission, ground_truth, verbose=True, dataset_name="charades"):
    if dataset_name in ["tacos"]:
        length_ranges = [[0, 10], [10, 30], [30, 150], [150, 600], [0, 600], ]  #
        range_names = ["short", "middle", "long", "superlong", "full"]
        max_length = 600
    else:
        length_ranges = [[0, 10], [10, 30], [30, 150], [0, 150], ]  #
        range_names = ["short", "middle", "long", "full"]
        max_length = 150

    ret_metrics = {}
    for l_range, name in zip(length_ranges, range_names):
        if verbose:
            start_time = time.time()
        _submission, _ground_truth = get_data_by_range(submission, ground_truth, l_range, max_length)
        print(f"{name}: {l_range}, {len(_ground_truth)}/{len(ground_truth)}="
              f"{100*len(_ground_truth)/len(ground_truth):.2f} examples.")
        if len(_ground_truth) == 0:
            continue
        else:
            iou_thd2average_precision = compute_mr_ap(_submission, _ground_truth, num_workers=8, chunksize=50)
        if dataset_name in ["tacos"]:
            iou_thds = np.array([0.1, 0.3, 0.5, 0.7])
        else:
            iou_thds = np.concatenate([np.array([0.3]), np.linspace(0.5, 0.95, 10)])
        iou_thd2recall_at_one = compute_mr_r1(_submission, _ground_truth, iou_thds=iou_thds)
        ret_metrics[name] = {"MR-mAP": iou_thd2average_precision, "MR-R1": iou_thd2recall_at_one}
        if verbose:
            print(f"[eval_moment_retrieval] [{name}] {time.time() - start_time:.2f} seconds")
    return ret_metrics


def compute_mr_ap(submission, ground_truth, iou_thds=np.linspace(0.5, 0.95, 10),
                  max_gt_windows=None, max_pred_windows=10, num_workers=8, chunksize=50):
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2data = defaultdict(list)
    for d in submission:
        pred_windows = d["pred_relevant_windows"][:max_pred_windows] \
            if max_pred_windows is not None else d["pred_relevant_windows"]
        qid = d["qid"]
        for w in pred_windows:
            pred_qid2data[qid].append({
                "video-id": qid,  # in order to use the API
                "t-start": w[0],
                "t-end": w[1],
                "score": w[2]
            })

    gt_qid2data = defaultdict(list)
    for d in ground_truth:
        gt_windows = d["relevant_windows"][:max_gt_windows] \
            if max_gt_windows is not None else d["relevant_windows"]
        qid = d["qid"]
        for w in gt_windows:
            gt_qid2data[qid].append({
                "video-id": d["qid"],
                "t-start": w[0],
                "t-end": w[1]
            })
    qid2ap_list = {}
    # start_time = time.time()
    data_triples = [[qid, gt_qid2data[qid], pred_qid2data[qid]] for qid in pred_qid2data]
    from functools import partial
    compute_ap_from_triple = partial(
        compute_average_precision_detection_wrapper, tiou_thresholds=iou_thds)

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for qid, scores in pool.imap_unordered(compute_ap_from_triple, data_triples, chunksize=chunksize):
                qid2ap_list[qid] = scores
    else:
        for data_triple in data_triples:
            qid, scores = compute_ap_from_triple(data_triple)
            qid2ap_list[qid] = scores

    # print(f"compute_average_precision_detection {time.time() - start_time:.2f} seconds.")
    ap_array = np.array(list(qid2ap_list.values()))  # (#queries, #thd)
    ap_thds = ap_array.mean(0)  # mAP at different IoU thresholds.
    iou_thd2ap = dict(zip([str(e) for e in iou_thds], ap_thds))
    iou_thd2ap["average"] = np.mean(ap_thds)
    # formatting
    iou_thd2ap = {k: float(f"{100 * v:.2f}") for k, v in iou_thd2ap.items()}
    return iou_thd2ap


def compute_average_precision_detection_wrapper(
        input_triple, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    qid, ground_truth, prediction = input_triple
    scores = compute_average_precision_detection(
        ground_truth, prediction, tiou_thresholds=tiou_thresholds)
    return qid, scores


def compute_average_precision_detection(ground_truth,
                                        prediction,
                                        tiou_thresholds=np.linspace(
                                            0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as true
    positive. This code is greatly inspired by Pascal VOC devkit.

    Args:
        ground_truth (list[dict]): List containing the ground truth instances
            (dictionaries). Required keys are 'video-id', 't-start' and
            't-end'.
        prediction (list[dict]): List containing the prediction instances
            (dictionaries). Required keys are: 'video-id', 't-start', 't-end'
            and 'score'.
        tiou_thresholds (np.ndarray): A 1darray indicates the temporal
            intersection over union threshold, which is optional.
            Default: ``np.linspace(0.5, 0.95, 10)``.

    Returns:
        Float: ap, Average precision score.
    """
    num_thresholds = len(tiou_thresholds)
    num_gts = len(ground_truth)
    num_preds = len(prediction)
    ap = np.zeros(num_thresholds)
    if len(prediction) == 0:
        return ap

    num_positive = float(num_gts)
    lock_gt = np.ones((num_thresholds, num_gts)) * -1
    # Sort predictions by decreasing score order.
    prediction.sort(key=lambda x: -x['score'])
    # Initialize true positive and false positive vectors.
    tp = np.zeros((num_thresholds, num_preds))
    fp = np.zeros((num_thresholds, num_preds))

    # Adaptation to query faster
    ground_truth_by_videoid = {}
    for i, item in enumerate(ground_truth):
        item['index'] = i
        ground_truth_by_videoid.setdefault(item['video-id'], []).append(item)

    # Assigning true positive to truly grount truth instances.
    for idx, pred in enumerate(prediction):
        if pred['video-id'] in ground_truth_by_videoid:
            gts = ground_truth_by_videoid[pred['video-id']]
        else:
            fp[:, idx] = 1
            continue

        _pred = np.array([[pred['t-start'], pred['t-end']], ])
        _gt = np.array([[gt['t-start'], gt['t-end']] for gt in gts])
        tiou_arr = compute_temporal_iou_batch_cross(_pred, _gt)[0]

        tiou_arr = tiou_arr.reshape(-1)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for t_idx, tiou_threshold in enumerate(tiou_thresholds):
            for j_idx in tiou_sorted_idx:
                if tiou_arr[j_idx] < tiou_threshold:
                    fp[t_idx, idx] = 1
                    break
                if lock_gt[t_idx, gts[j_idx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[t_idx, idx] = 1
                lock_gt[t_idx, gts[j_idx]['index']] = idx
                break

            if fp[t_idx, idx] == 0 and tp[t_idx, idx] == 0:
                fp[t_idx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum = tp_cumsum / num_positive

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for t_idx in range(len(tiou_thresholds)):
        ap[t_idx] = interpolated_precision_recall(precision_cumsum[t_idx, :],
                                                  recall_cumsum[t_idx, :])
    return ap


def compute_mr_r1(submission, ground_truth, iou_thds=np.linspace(0.5, 0.95, 10)):
    """If a predicted segment has IoU >= iou_thd with one of the 1st GT segment, we define it positive"""
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2window = {d["qid"]: d["pred_relevant_windows"][0][:2] for d in submission}  # :2 rm scores
    # gt_qid2window = {d["qid"]: d["relevant_windows"][0] for d in ground_truth}
    gt_qid2window = {}
    ious = []
    for d in ground_truth:
        cur_gt_windows = d["relevant_windows"]
        cur_qid = d["qid"]
        cur_max_iou_idx = 0
        if len(cur_gt_windows) > 0:  # select the GT window that has the highest IoU
            cur_ious = compute_temporal_iou_batch_cross(
                np.array([pred_qid2window[cur_qid]]), np.array(d["relevant_windows"])
            )[0]
            ious.append(np.max(cur_ious))
            cur_max_iou_idx = np.argmax(cur_ious)
        gt_qid2window[cur_qid] = cur_gt_windows[cur_max_iou_idx]
    
    miou = np.array(ious).mean()
    qids = list(pred_qid2window.keys())
    pred_windows = np.array([pred_qid2window[k] for k in qids]).astype(float)
    gt_windows = np.array([gt_qid2window[k] for k in qids]).astype(float)
    pred_gt_iou = compute_temporal_iou_batch_paired(pred_windows, gt_windows)
    iou_thd2recall_at_one = {}
    for thd in iou_thds:
        iou_thd2recall_at_one[str(thd)] = float(f"{np.mean(pred_gt_iou >= thd) * 100:.2f}")
    iou_thd2recall_at_one["miou"] = float(f"{miou * 100:.2f}")
    return iou_thd2recall_at_one


def get_data_by_range(submission, ground_truth, len_range, global_max_length):
    """ keep queries with ground truth window length in the specified length range.
    Args:
        submission:
        ground_truth:
        len_range: [min_l (int), max_l (int)]. the range is (min_l, max_l], i.e., min_l < l <= max_l
    """
    min_l, max_l = len_range
    if min_l == 0 and max_l == global_max_length:  # min and max l in dataset
        return submission, ground_truth

    # only keep ground truth with windows in the specified length range
    # if multiple GT windows exists, we only keep the ones in the range
    ground_truth_in_range = []
    gt_qids_in_range = set()
    for d in ground_truth:
        rel_windows_in_range = [
            w for w in d["relevant_windows"] if min_l < get_window_len(w) <= max_l]
        if len(rel_windows_in_range) > 0:
            d = copy.deepcopy(d)
            d["relevant_windows"] = rel_windows_in_range
            ground_truth_in_range.append(d)
            gt_qids_in_range.add(d["qid"])

    # keep only submissions for ground_truth_in_range
    submission_in_range = []
    for d in submission:
        if d["qid"] in gt_qids_in_range:
            submission_in_range.append(copy.deepcopy(d))

    return submission_in_range, ground_truth_in_range


def post_processing_mr_nms(mr_res, nms_thd, max_before_nms, max_after_nms):
    mr_res_after_nms = []
    for e in mr_res:
        e["pred_relevant_windows"] = temporal_nms(
            e["pred_relevant_windows"][:max_before_nms],
            nms_thd=nms_thd,
            max_after_nms=max_after_nms
        )
        mr_res_after_nms.append(e)
    return mr_res_after_nms


def inference():
    logger.info(f"Inference Mode")
    opt = TestOptions().parse()
    cudnn.benchmark = True
    cudnn.deterministic = False

    if opt.tokenizer_type == "GloVeSimple":
        vocab = build_vocab(opt)
    elif opt.tokenizer_type == "GloVeNLTK":
        if opt.load_vocab_pkl:
            vocab = build_vocab_from_pkl(opt)
        else:
            vocab = build_vocab(opt)
    else:
        vocab = None

    _, _, test_loaders = build_dataloader(opt, vocab)
    assert len(test_loaders) == 1
    for split in test_loaders.keys():
        test_loader = test_loaders[split]
        break
    model = build_model(opt, vocab)
    # criterion = build_criterion(opt)
    criterion = None

    logger.info(f"Load checkpoint from {opt.resume}")
    checkpoint = torch.load(opt.resume, map_location="cpu")
    if model.text_encoder is not None:
        model_state_dict = merge_state_dict_with_module(
            checkpoint["model"], model.text_encoder.state_dict(), "text_encoder"
        )
    else:
        model_state_dict = checkpoint["model"]
    model.load_state_dict(model_state_dict)
    logger.info(f"Loaded model saved at epoch {checkpoint['epoch']} from checkpoint: {opt.resume}")
    logger.info(f"Model {model}")
    count_parameters(model)

    save_submission_filename = f"{opt.dataset_name}_test_submission.jsonl"
    with torch.inference_mode():
        metrics_no_nms, metrics_nms, _, _ = \
            eval_epoch(model, test_loader, opt, save_submission_filename, criterion)
    if metrics_no_nms is not None:
        logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
    if metrics_nms is not None:
        logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))


if __name__ == "__main__":
    inference()
