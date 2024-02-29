import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
from collections import defaultdict

from .tokenizer import CLIPTokenizer, GloVeSimpleTokenizer
from .tokenizer import NLTKTokenizer, NLTKTokenizerWithFeature
from utils import pad_sequences_1d, span_xx_to_cxw


class BaseDataset(Dataset):
    def __init__(self, ann_path, feat_files, split,
                 use_tef, clip_len, max_words_l, max_video_l,
                 tokenizer_type, load_vocab_pkl, bpe_path, vocab,
                 normalize_video, contra_samples,
                 recfw, vocab_size, max_gather_size):
        super(Dataset, self).__init__()

        self.ann_path = ann_path
        self.feat_files = feat_files
        self.split = split
        self.use_tef = use_tef
        self.clip_len = clip_len
        self.max_words_l = max_words_l
        self.max_video_l = max_video_l
        self.normalize_video = normalize_video
        self.contra_samples = contra_samples
        self.recfw = recfw
        self.vocab_size = vocab_size
        self.max_gather_size = max_gather_size
        
        if tokenizer_type == "CLIP":
            id2label = self.load_CLIP_keep_vocab() if self.recfw else None
            self.tokenizer = CLIPTokenizer(recfw, id2label, bpe_path)
        elif tokenizer_type == "GloVeSimple":
            id2label = self.load_GloVe_keep_vocab() if self.recfw else None
            self.tokenizer = GloVeSimpleTokenizer(recfw, id2label, vocab)
        elif tokenizer_type == "GloVeNLTK":
            if load_vocab_pkl:
                id2label = self.load_GloVe_pkl_keep_vocab(vocab) if self.recfw else None
                self.tokenizer = NLTKTokenizerWithFeature(recfw, id2label, vocab)
            else:
                id2label = self.load_GloVe_keep_vocab() if self.recfw else None
                self.tokenizer = NLTKTokenizer(recfw, id2label, vocab)
        self.data = self.load_annotations()
        self.merged_data = self._gather_data_by_video_id()

    def __len__(self):
        return len(self.merged_data)
    
    def load_CLIP_keep_vocab(self):
        id2label = {}
        vocab_file = os.path.join(self.ann_path, "CLIP_tokenized_count.txt")
        with open(vocab_file, 'r') as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                words_id = int(line.split(' ')[0])
                id2label[words_id] = count
                count += 1
                if count == self.vocab_size:
                    break
        id2label['<unknown>'] = self.vocab_size
        id2label['<start>'] = self.vocab_size + 1
        id2label['<end>'] = self.vocab_size + 2
        return id2label

    def load_GloVe_keep_vocab(self):
        id2label = {}
        vocab_file = os.path.join(self.ann_path, "GloVe_tokenized_count.txt")
        with open(vocab_file, 'r') as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                words_id = int(line.split(' ')[1])
                id2label[words_id] = count
                count += 1
                if count == self.vocab_size:
                    break
        id2label['<unknown>'] = self.vocab_size
        return id2label
    
    def load_GloVe_pkl_keep_vocab(self, vocab):
        id2label = {}
        count = 0
        for w, _ in vocab['counter'].most_common(self.vocab_size):
            id2label[w] = count
            count += 1
        id2label['<unknown>'] = self.vocab_size
        return id2label
    
    def load_annotations(self):
        raise NotImplementedError

    def get_video_feat(self, video_id):
        raise NotImplementedError

    def sample_video_feat(self, video_feat):
        video_length = video_feat.shape[0]
        if video_length > self.max_video_l:
            idxs = torch.arange(0, self.max_video_l+1, 1.0) / self.max_video_l * video_length
            idxs = idxs.round().long().clamp(max=video_length-1)
            mean_feat = []
            for i in range(self.max_video_l):
                s, e = idxs[i], idxs[i+1]
                if s < e:
                    mean_feat.append(video_feat[s:e].mean(dim=0))
                else:
                    mean_feat.append(video_feat[s])
            return torch.stack(mean_feat)
        else:
            return video_feat
    
    def _gather_data_by_video_id(self):
        gathered_data = defaultdict(list)
        for i, meta in enumerate(self.data):
            video_id = meta['video_id']
            gathered_data[video_id].append(meta)

        merged_data = []
        if 'start_idx' in meta:
            sort_key = 'start_idx'
        elif 'video_start' in meta:
            sort_key = 'video_start'
        else:
            raise ValueError("start_idx and video_start not found")
        
        # for video_id, metas in gathered_data.items():
        #     random.shuffle(metas)
        #     for start_idx in range(0, len(metas), self.max_gather_size):
        #         merged_meta = defaultdict(list)
        #         end_index = min(start_idx + self.max_gather_size, len(metas))
        #         sub_metas = metas[start_idx: end_index]
        #         sub_metas = sorted(sub_metas, key=lambda x: x[sort_key])
        #         for meta in sub_metas:
        #             for key, value in meta.items():
        #                 merged_meta[key].append(value)
        #         merged_data.append(merged_meta)

        for video_id, metas in gathered_data.items():
            if self.max_gather_size > 0:
                random.shuffle(metas)
                for start_idx in range(0, len(metas), self.max_gather_size):
                    merged_meta = defaultdict(list)
                    end_index = min(start_idx + self.max_gather_size, len(metas))
                    sub_metas = metas[start_idx: end_index]
                    sub_metas = sorted(sub_metas, key=lambda x: x[sort_key])
                    for meta in sub_metas:
                        for key, value in meta.items():
                            merged_meta[key].append(value)
                    merged_data.append(merged_meta)
            else:
                merged_meta = defaultdict(list)
                metas = sorted(metas, key=lambda x: x[sort_key])
                for meta in metas:
                    for key, value in meta.items():
                        merged_meta[key].append(value)
                merged_data.append(merged_meta)
        
        return merged_data

    def __getitem__(self, index):
        meta = self.merged_data[index]
        num_clips = len(meta['video_id'])
        video_feat = self.get_video_feat(meta['video_id'][0])
        video_feat = self.sample_video_feat(video_feat)
        video_length = video_feat.shape[0]
        if self.use_tef:
            video_feat = self.add_tef(video_length, video_feat)
        start_idx = meta['start_idx']
        end_idx = meta['end_idx']
        if self.clip_len == -1:
            start_idx = [int(idx * video_length) for idx in start_idx]
            end_idx = [int(idx * video_length) for idx in end_idx]
        clip_mask = []
        pos_idxes = []
        neg_idxes = []
        for i in range(num_clips):
            if end_idx[i] > video_length - 1:
                end_idx[i] = video_length - 1
            if start_idx[i] > end_idx[i]:
                start_idx[i] = end_idx[i]
            start = start_idx[i]
            end = end_idx[i]
            mask = torch.zeros([video_feat.shape[0]], dtype=torch.bool)
            mask[start: end + 1] = True
            clip_mask.append(mask)
            
            # sample positive frames and negative frames
            if self.contra_samples > 0:
                try:
                    pos_idx = np.random.choice(np.arange(start, end+1), self.contra_samples, replace=False)
                except ValueError:
                    pos_idx = np.random.choice(np.arange(start, end+1), self.contra_samples, replace=True)
                pos_idxes.append(torch.from_numpy(pos_idx))

                neg_pool = np.hstack([np.arange(0, start), np.arange(end+1, video_length)])
                if len(neg_pool) >= self.contra_samples:
                    neg_idx = np.random.choice(neg_pool, self.contra_samples, replace=False)
                else:
                    neg_idx = np.random.choice(neg_pool, self.contra_samples, replace=True)
                neg_idxes.append(torch.from_numpy(neg_idx))
        
        return {
            "num_clips": num_clips,
            "video_feat": video_feat,
            "video_id": meta['video_id'][0],
            "duration": meta['duration'][0],
            "moment": meta['moment'],
            "sentence": meta['sentence'],
            "words_id": meta['words_id'],
            "words_weight": meta['words_weight'],
            "unknown_mask": meta['unknown_mask'],
            "words_label": meta['words_label'],
            "start_idx": start_idx,
            "end_idx": end_idx,
            "clip_mask": clip_mask,
            "pos_idx": pos_idxes if self.contra_samples > 0 else [None],
            "neg_idx": neg_idxes if self.contra_samples > 0 else [None],
            "qid": meta['qid'],
        }
    
    def add_tef(self, ctx_l, video_feat):
        tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
        tef_ed = tef_st + 1.0 / ctx_l
        tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
        video_feat = torch.cat([video_feat, tef], dim=1)  # (Lv, Dv+2)
        return video_feat


class SplitGatherBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle):
        # self.dataset = dataset
        self.merged_data = dataset.merged_data
        self.batch_size = batch_size
        self.shuffle = shuffle

        id_groups = defaultdict(list)
        for idx, data in enumerate(self.merged_data):
            video_id = data['video_id'][0]
            id_groups[video_id].append(idx)

        if self.shuffle:
            for key in id_groups.keys():
                random.shuffle(id_groups[key])
        
        self.id_groups = id_groups

    def __iter__(self):
        groups_iterators = [iter(group) for group in self.id_groups.values()]
        num_groups = len(groups_iterators)
        group_idx = list(range(num_groups))
        batch = []
        while True:
            if self.shuffle:
                random.shuffle(group_idx)
            breakFlag = False
            for idx in range(num_groups):
                try:
                    data_idx = next(groups_iterators[group_idx[idx]])
                    # batch.append(self.dataset[data_idx])
                    batch.append(data_idx)
                except StopIteration:
                    continue
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                    breakFlag = True
                    break

            # must make sure the real batch_size > 1
            if not breakFlag:
                if len(batch) <= 1:
                    break
                else:
                    yield batch
                    batch = []

    def __len__(self):
        length = [len(group) for group in self.id_groups.values()]
        sorted_length = sorted(length)
        dropped_length = sorted_length[-1] - sorted_length[-2]
        return (len(self.merged_data) - dropped_length + self.batch_size - 1) // self.batch_size


def collate(batch):
    batched_data = dict()
    num_clips = []
    video_feat = []
    video_id = []
    duration = []
    moment = []
    sentence = []
    words_id = []
    words_weight = []
    unknown_mask = []
    words_label = []
    start_idx = []
    end_idx = []
    clip_mask = []
    pos_idx = []
    neg_idx = []
    qid = []

    for e in batch:
        num_clips.append(e['num_clips'])
        for i in range(e['num_clips']):
            video_feat.append(e['video_feat'])
            video_id.append(e['video_id'])
            duration.append(e['duration'])
        moment += e['moment']
        sentence += e['sentence']
        words_id += e['words_id']
        words_weight += e['words_weight']
        unknown_mask += e['unknown_mask']
        words_label += e['words_label']
        start_idx += e['start_idx']
        end_idx += e['end_idx']
        clip_mask += e['clip_mask']
        pos_idx += e['pos_idx']
        neg_idx += e['neg_idx']
        qid += e['qid']

    batched_data['num_clips'] = torch.LongTensor(num_clips)
    batched_data['video_feat'], batched_data['video_mask'] = \
        pad_sequences_1d(video_feat, dtype=video_feat[0].dtype, fixed_length=None)
    batched_data['duration'] = torch.Tensor(duration)
    batched_data['moment'] = torch.Tensor(moment)
    batched_data['words_id'] = torch.cat(words_id, dim=0)
    if batched_data['words_id'].ndim == 2:
        batched_data['words_mask'] = batched_data['words_id'] != 0
    elif batched_data['words_id'].ndim == 3:
        # batched_data['words_mask'] = batched_data['words_id'].abs().sum(dim=-1) != 0
        batched_data['words_mask'] = None
    else:
        raise ValueError(f"words_id has shape {batched_data['words_id'].shape}")
    batched_data['words_weight'] = torch.cat(words_weight, dim=0)
    if words_label[0] is not None:
        batched_data['unknown_mask'] = torch.cat(unknown_mask, dim=0)
        batched_data['words_label'] = torch.cat(words_label, dim=0)
    batched_data['start_idx'] = torch.LongTensor(start_idx)
    batched_data['end_idx'] = torch.LongTensor(end_idx)
    batched_data['clip_mask'], _ = \
        pad_sequences_1d(clip_mask, dtype=clip_mask[0].dtype, fixed_length=None)
    if pos_idx[0] is not None:
        batched_data['pos_idx'] = torch.stack(pos_idx, dim=0)
        batched_data['neg_idx'] = torch.stack(neg_idx, dim=0)
    
    batched_data['qid'] = qid
    batched_data['video_id'] = video_id
    batched_data['sentence'] = sentence

    return batched_data


def prepare_batch_input(batched_data, device, non_blocking=False):
    for key, value in batched_data.items():
        if key == "words_weight":
            continue
        if isinstance(value, torch.Tensor):
            batched_data[key] = value.to(device, non_blocking=non_blocking)
        # if isinstance(value, List):
        #     for i in range(len(value)):
        #         if isinstance(value[i], torch.Tensor):
        #             batched_data[key][i] = value.to(device, non_blocking=non_blocking)
        if key == "norm_moment":
            batched_data[key] = [
                dict(moments=e["moments"].to(device, non_blocking=non_blocking))
                for e in batched_data[key]
            ]
        if key == "norm_span":
            batched_data[key] = [
                dict(spans=e["spans"].to(device, non_blocking=non_blocking))
                for e in batched_data[key]
            ]
    
    # calculate extra information
    if "moment" in batched_data and "norm_span" not in batched_data:
        moment = batched_data["moment"]
        duration = batched_data["duration"]
        batched_data["norm_moment"] = moment / duration.unsqueeze(1)
        batched_data["norm_span"] = span_xx_to_cxw(batched_data["norm_moment"])

