import os
import time
import json
import pprint
import random
import torch
import logging
import numpy as np
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from collections import defaultdict

from utils import BaseOptions
from runner import build_vocab, build_vocab_from_pkl
from runner import build_dataloader, build_model
from runner import build_criterion, build_optimizer
from dataset import prepare_batch_input
from utils import AverageMeter
from utils import dict_to_markdown, count_parameters
from utils import state_dict_without_module
from eval import eval_epoch


logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer):
    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()
    criterion.train()

    # init meters
    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()
    for batch_idx, batch in tqdm(enumerate(train_loader),
                                 desc="Training Iteration",
                                 total=num_training_examples):
        time_meters["dataloading_time"].update(time.time() - timer_dataloading)

        timer_start = time.time()

        prepare_batch_input(batch, opt.device, non_blocking=opt.pin_memory)
        time_meters["prepare_inputs_time"].update(time.time() - timer_start)
        timer_start = time.time()

        outputs = model(**batch, dataset_name=opt.dataset_name, is_training=True)
        loss_dict, loss = criterion(outputs, batch, is_training=True)
        time_meters["model_forward_time"].update(time.time() - timer_start)
        timer_start = time.time()
        optimizer.zero_grad()
        loss.backward()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        time_meters["model_backward_time"].update(time.time() - timer_start)

        loss_dict["loss_overall"] = float(loss)  # for logging only
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v) * criterion.weight_dict[k] if k in criterion.weight_dict else float(v))

        timer_dataloading = time.time()
        
    # print/add logs
    tb_writer.add_scalar("Train/lr", float(optimizer.param_groups[0]["lr"]), epoch_i+1)
    for k, v in loss_meters.items():
        tb_writer.add_scalar("Train/{}".format(k), v.avg, epoch_i+1)

    to_write = opt.train_log_txt_formatter.format(
        time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
        epoch=epoch_i+1,
        loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]))
    with open(opt.train_log_filepath, "a") as f:
        f.write(to_write)

    logger.info("Epoch time stats:")
    for name, meter in time_meters.items():
        d = {k: f"{getattr(meter, k):.4f}" for k in ["max", "min", "avg"]}
        logger.info(f"{name} ==> {d}")


def train():
    opt = BaseOptions().parse()
    set_seed(opt.seed)

    if opt.tokenizer_type == "GloVeSimple":
        vocab = build_vocab(opt)
    elif opt.tokenizer_type == "GloVeNLTK":
        if opt.load_vocab_pkl:
            vocab = build_vocab_from_pkl(opt)
        else:
            vocab = build_vocab(opt)
    else:
        vocab = None
    train_loader, val_loaders, _ = build_dataloader(opt, vocab)
    model = build_model(opt, vocab)
    criterion = build_criterion(opt)
    optimizer, lr_scheduler = build_optimizer(opt, model)

    if opt.resume is not None:
        logger.info(f"Load checkpoint from {opt.resume}")
        checkpoint = torch.load(opt.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if opt.resume_all:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            opt.start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Loaded model saved at epoch {checkpoint['epoch']} from checkpoint: {opt.resume}")
    else:
        logger.warning("If you intend to evaluate the model, please specify --resume with ckpt path")
    
    logger.info(f"Model {model}")
    count_parameters(model)
    logger.info("Start training...")

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Split] {split} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"
    
    prev_best_score_dict = {key: 0. for key in val_loaders.keys()}
    es_cnt = 0
    # start_epoch = 0
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = opt.start_epoch
    
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer)
            lr_scheduler.step()
        
        if (epoch_i + 1) % opt.eval_epoch_interval == 0:
            for key, val_loader in val_loaders.items():
                logger.info(f"Evaluating {key} split")
                save_submission_filename = "{}_latest_{}_val_preds.jsonl".format(key, opt.dataset_name)
                with torch.no_grad():
                    metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
                        eval_epoch(model, val_loader, opt, save_submission_filename, epoch_i, criterion, tb_writer)
                
                # log
                to_write = opt.eval_log_txt_formatter.format(
                    time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                    epoch=epoch_i,
                    split=key,
                    loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]),
                    eval_metrics_str=json.dumps(metrics_no_nms))

                with open(opt.eval_log_filepath, "a") as f:
                    f.write(to_write)
                if metrics_no_nms is not None:
                    logger.info("{} metrics_no_nms {}".format(key, pprint.pformat(metrics_no_nms["brief"], indent=4)))
                if metrics_nms is not None:
                    logger.info("{} metrics_nms {}".format(key, pprint.pformat(metrics_nms["brief"], indent=4)))

                metrics = metrics_no_nms
                for k, v in metrics["brief"].items():
                    if v is None:
                        continue
                    tb_writer.add_scalar(f"Eval/{key}-{k}", float(v), epoch_i+1)

                stop_score = metrics["brief"][f"MR-full-{opt.stop_score}"]
                if stop_score > prev_best_score_dict[key]:
                    es_cnt = 0
                    prev_best_score_dict[key] = stop_score

                    checkpoint = {
                        "model": state_dict_without_module(model, "text_encoder"),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch_i,
                        "opt": opt
                    }
                    torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_{key}_best.ckpt"))

                    best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                    for src, tgt in zip(latest_file_paths, best_file_paths):
                        os.renames(src, tgt)
                    logger.info("The checkpoint file has been updated.")
                else:
                    es_cnt += 1
                    if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                        with open(opt.train_log_filepath, "a") as f:
                            f.write(f"Early Stop at epoch {epoch_i}")
                        logger.info(f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score_dict[key]}\n")
                        break

                # save ckpt
                checkpoint = {
                    "model": state_dict_without_module(model, "text_encoder"),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt
                }
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        if (epoch_i + 1) % opt.save_interval == 0 or (epoch_i + 1) % opt.lr_drop == 0:  # additional copies
            checkpoint = {
                "model": state_dict_without_module(model, "text_encoder"),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_e{epoch_i:04d}.ckpt"))

    tb_writer.close()


if __name__ == "__main__":
    train()
