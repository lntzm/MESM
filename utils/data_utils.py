import torch
import numpy as np
from .span_utils import generalized_temporal_iou


class AverageMeter(object):
    """Computes and stores the average and current/max/min value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e10
        self.min = 1e10
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = -1e10
        self.min = 1e10

    def update(self, val, n=1):
        self.max = max(val, self.max)
        self.min = min(val, self.min)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pad_sequences_1d(sequences, dtype=torch.long, device=torch.device("cpu"), fixed_length=None):
    """ Pad a single-nested list or a sequence of n-d array (torch.tensor or np.ndarray)
    into a (n+1)-d array, only allow the first dim has variable lengths.
    Args:
        sequences: list(n-d tensor or list)
        dtype: np.dtype or torch.dtype
        device:
        fixed_length: pad all seq in sequences to fixed length. All seq should have a length <= fixed_length.
            return will be of shape [len(sequences), fixed_length, ...]
    Returns:
        padded_seqs: ((n+1)-d tensor) padded with zeros
        mask: (2d tensor) of the same shape as the first two dims of padded_seqs,
              1 indicate valid, 0 otherwise
    Examples:
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=torch.long)
        >>> test_data_3d = [torch.randn(2,3,4), torch.randn(4,3,4), torch.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=torch.float)
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=np.float32)
        >>> test_data_3d = [np.random.randn(2,3,4), np.random.randn(4,3,4), np.random.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=np.float32)
    """
    if isinstance(sequences[0], list):
        if "torch" in str(dtype):
            sequences = [torch.tensor(s, dtype=dtype, device=device) for s in sequences]
        else:
            sequences = [np.asarray(s, dtype=dtype) for s in sequences]

    extra_dims = sequences[0].shape[1:]  # the extra dims should be the same for all elements
    lengths = [len(seq) for seq in sequences]
    if fixed_length is not None:
        max_length = fixed_length
    else:
        max_length = max(lengths)
    if isinstance(sequences[0], torch.Tensor):
        assert "torch" in str(dtype), "dtype and input type does not match"
        padded_seqs = torch.zeros((len(sequences), max_length) + extra_dims, dtype=dtype, device=device)
        mask = torch.zeros((len(sequences), max_length), dtype=torch.float32, device=device)
    else:  # np
        assert "numpy" in str(dtype), "dtype and input type does not match"
        padded_seqs = np.zeros((len(sequences), max_length) + extra_dims, dtype=dtype)
        mask = np.zeros((len(sequences), max_length), dtype=np.float32)

    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    return padded_seqs, mask.bool()  # , lengths


def split_and_pad(split_num, in_tensor):
    # split = torch.where(index[1:] - index[:-1] != 0)[0] + 1
    # split = torch.cat(
    #     [torch.tensor([0], device=index.device), split, torch.tensor([len(index)], device=index.device)],
    #     dim=0
    # )
    # split_num = split[1:] - split[:-1]
    res = torch.split(in_tensor, split_num.tolist(), dim=0)
    pad_res = pad_sequences_1d(res, dtype=in_tensor.dtype, fixed_length=None, device=in_tensor.device)
    return pad_res[0], pad_res[1].bool()


def split_expand_and_pad(split_num, expand_num, in_tensor):
    # split = torch.where(index[1:] - index[:-1] != 0)[0] + 1
    # split = torch.cat(
    #     [torch.tensor([0], device=index.device), split, torch.tensor([len(index)], device=index.device)],
    #     dim=0
    # )
    # split_num = split[1:] - split[:-1]
    expanded_tensor = []
    for num, each_tensor in zip(expand_num, torch.split(in_tensor, split_num.tolist(), dim=0)):
        for i in range(num):
            expanded_tensor.append(each_tensor.clone())
    # res = torch.split(in_tensor, split_num.tolist(), dim=0)
    pad_res = pad_sequences_1d(expanded_tensor, dtype=in_tensor.dtype, fixed_length=None, device=in_tensor.device)
    return pad_res[0], pad_res[1].bool()


def sample_outclass_neg(num_clips):
    neg_index = []
    num_clips_end = num_clips.cumsum(dim=0)
    num_clips_start = torch.cat([torch.zeros([1], dtype=num_clips.dtype, device=num_clips.device), num_clips_end[:-1]])
    for num, start, end in zip(num_clips, num_clips_start, num_clips_end):
        for i in range(num):
            each_neg = torch.ones([num_clips_end[-1]], dtype=torch.bool, device=num_clips.device)
            each_neg[start: end] = False
            candidate = torch.where(each_neg)[0]
            neg_index.append(candidate[torch.randperm(candidate.shape[0])][0])
    neg_index = torch.stack(neg_index)
    return neg_index


def sample_inclass_neg(num_clips, norm_moments):
    assert num_clips.shape[0] == 1
    neg_index = []
    ious = generalized_temporal_iou(norm_moments, norm_moments)
    candidate_mask = ious < 0.4
    for i in range(candidate_mask.shape[0]):
        candidate = torch.where(candidate_mask[i])[0]
        neg_index.append(candidate[torch.randperm(candidate.shape[0])][0])
    neg_index = torch.stack(neg_index)
    return neg_index


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    output: (#items, #classes)
    target: int,
    """
    maxk = max(topk)
    num_items = output.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / num_items))
    return res


def interpolated_precision_recall(precision, recall):
    """Interpolated AP - VOCdevkit from VOC 2011.

    Args:
        precision (np.ndarray): The precision of different thresholds.
        recall (np.ndarray): The recall of different thresholds.

    Returns：
        float: Average precision score.
    """
    mprecision = np.hstack([[0], precision, [0]])
    mrecall = np.hstack([[0], recall, [1]])
    for i in range(len(mprecision) - 1)[::-1]:
        mprecision[i] = max(mprecision[i], mprecision[i + 1])
    idx = np.where(mrecall[1::] != mrecall[0:-1])[0] + 1
    ap = np.sum((mrecall[idx] - mrecall[idx - 1]) * mprecision[idx])
    return ap


def compute_temporal_iou_batch_paired(pred_windows, gt_windows):
    """ compute intersection-over-union along temporal axis for each pair of windows in pred_windows and gt_windows.
    Args:
        pred_windows: np.ndarray, (N, 2), [st (float), ed (float)] * N
        gt_windows: np.ndarray, (N, 2), [st (float), ed (float)] * N
    Returns:
        iou (float): np.ndarray, (N, )

    References:
        for np.divide with zeros, see https://stackoverflow.com/a/37977222
    """
    intersection = np.maximum(
        0, np.minimum(pred_windows[:, 1], gt_windows[:, 1]) - np.maximum(pred_windows[:, 0], gt_windows[:, 0])
    )
    union = np.maximum(pred_windows[:, 1], gt_windows[:, 1]) \
            - np.minimum(pred_windows[:, 0], gt_windows[:, 0])  # not the correct union though
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0)

