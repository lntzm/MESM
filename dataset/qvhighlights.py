import os
import h5py
import json
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from utils import pad_sequences_1d, span_xx_to_cxw
from .base import BaseDataset


class QVHighlightsDataset(BaseDataset):
    def __init__(self, ann_path, feat_files, split,
                 use_tef, clip_len, max_words_l, max_video_l,
                 tokenizer_type, load_vocab_pkl, bpe_path, vocab,
                 normalize_video, contra_samples,
                 recfw, vocab_size, max_windows, max_gather_size):
        super().__init__(ann_path, feat_files, split,
                         use_tef, clip_len, max_words_l, max_video_l,
                         tokenizer_type, load_vocab_pkl, bpe_path, vocab,
                         normalize_video, contra_samples,
                         recfw, vocab_size, max_gather_size)
        self.max_windows = max_windows
    
    def load_annotations(self):
        split2filename = {
            "train": "highlight_train_release.jsonl",
            "val": "highlight_val_release.jsonl",
            "test": "highlight_test_release.jsonl",
        }
        ann_file = os.path.join(self.ann_path, split2filename[self.split])
        annotations = []

        with open(ann_file, 'r') as f:
            lines = f.readlines()
            for i in tqdm(range(len(lines)), desc=f"Load QVHighlights {self.split} annotations"):
                meta = json.loads(lines[i].strip("\n"))
                sentence = meta['query']
                [video_id, st, ed] = meta['vid'].rsplit('_', 2)

                words_id, words_weight, unknown_mask, words_label = \
                    self.tokenizer.tokenize(sentence, max_valid_length=self.max_words_l)
                
                data = {
                    "video_id": video_id,
                    "video_start": float(st),
                    "vid": meta["vid"],
                    "duration": meta["duration"],
                    "sentence": sentence,
                    "words_id": words_id,
                    "words_weight": words_weight,
                    "unknown_mask": unknown_mask,
                    "words_label": words_label,
                    "qid": meta["qid"],
                }
                if self.split != "test":
                    data.update({
                        "relevant_clip_ids": meta["relevant_clip_ids"],
                        "saliency_scores": meta["saliency_scores"],
                        "relevant_windows": meta["relevant_windows"],
                    })
                annotations.append(data)
        return annotations

    def __getitem__(self, index):
        meta = self.merged_data[index]
        num_clips = len(meta['video_id'])
        
        # start_idx = meta['start_idx']
        # end_idx = meta['end_idx']
        video_feats = []
        if self.split != "test":
            norm_moments = []
            norm_spans = []
            pos_idxes = []
            neg_idxes = []
            saliency_labels = []
            clip_mask = []
        for i in range(num_clips):
            video_feat = self.get_video_feat(meta['vid'][i])
            video_length = video_feat.shape[0]
            if self.use_tef:
                video_feat = self.add_tef(video_length, video_feat)  
            video_feats.append(video_feat)
            if self.split != "test":
                norm_moment, norm_span = self.get_span_labels(meta["relevant_windows"][i], video_length)
                norm_moments.append(norm_moment)
                norm_spans.append(norm_span)
                saliency_pos_labels, saliency_neg_labels, saliency_all_labels = \
                    self.get_saliency_labels_all(meta["relevant_clip_ids"][i], meta["saliency_scores"][i], video_length)
                saliency_all_labels = torch.from_numpy(saliency_all_labels)
                pos_idxes.append(torch.tensor(saliency_pos_labels))
                neg_idxes.append(torch.tensor(saliency_neg_labels))
                saliency_labels.append(saliency_all_labels)
                clip_mask.append(saliency_all_labels != 0)
          
            # mask = torch.zeros([video_feat.shape[0]], dtype=torch.bool)
            # mask[start: end + 1] = True
            # clip_mask.append(mask)
            
            # sample positive frames and negative frames
            # if self.contra_samples > 0:
            #     try:
            #         pos_idx = np.random.choice(np.arange(start, end+1), self.contra_samples, replace=False)
            #     except ValueError:
            #         pos_idx = np.random.choice(np.arange(start, end+1), self.contra_samples, replace=True)
            #     pos_idxes.append(torch.from_numpy(pos_idx))

            #     neg_pool = np.hstack([np.arange(0, start), np.arange(end+1, video_length)])
            #     if len(neg_pool) >= self.contra_samples:
            #         neg_idx = np.random.choice(neg_pool, self.contra_samples, replace=False)
            #     else:
            #         neg_idx = np.random.choice(neg_pool, self.contra_samples, replace=True)
            #     neg_idxes.append(torch.from_numpy(neg_idx))
        
        data = {
            "num_clips": num_clips,
            "video_feat": video_feats,
            "video_id": meta['vid'],
            "duration": meta['duration'],
            "sentence": meta['sentence'],
            "words_id": meta['words_id'],
            "words_weight": meta['words_weight'],
            "unknown_mask": meta['unknown_mask'],
            "words_label": meta['words_label'],
            "qid": meta['qid'],
        }
        if self.split != "test":
            data.update({
                "norm_moment": norm_moments,
                "norm_span": norm_spans,
                "pos_idx": pos_idxes if self.contra_samples > 0 else [None],
                "neg_idx": neg_idxes if self.contra_samples > 0 else [None],
                "saliency_label": saliency_labels,
                "clip_mask": clip_mask,
            })

        return data
    
    def get_span_labels(self, windows, ctx_l):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) > self.max_windows:
            random.shuffle(windows)
            windows = windows[:self.max_windows]
        windows = torch.Tensor(windows) / (ctx_l * self.clip_len)  # normalized windows in xx
        spans = span_xx_to_cxw(windows)  # normalized windows in cxw
        return windows, spans
    
    def get_saliency_labels_all(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
        """Sum the scores from the three annotations, then take the two clips with the
        maximum scores as positive, and two with the minimum scores as negative.
        Args:
            rel_clip_ids: list(int), list of relevant clip ids
            scores: list([anno1_score, anno2_score, anno3_score]),
            ctx_l: int
            max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
            add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
        """
        # indices inside rel_clip_ids
        scores = np.array(scores)  # (#rel_clips, 3)
        agg_scores = np.sum(scores, 1)  # (#rel_clips, )
        sort_indices = np.argsort(agg_scores)  # increasing

        # score_array = [min(agg_scores[idx], ctx_l-1) for idx in range(ctx_l)]
        score_array = np.zeros(ctx_l)
        for idx in range(len(rel_clip_ids)):
            if rel_clip_ids[idx] >= ctx_l:
                score_array_new = np.zeros(ctx_l + 1)
                score_array_new[:ctx_l] = score_array
                score_array = score_array_new
            # if rel_clip_ids[idx] == ctx_l:
            #     print(rel_clip_ids[idx], ctx_l)
            score_array[rel_clip_ids[idx]] = agg_scores[idx]

        # indices in the whole video
        # the min(_, ctx_l-1) here is incorrect, but should not cause
        # much troubles since this should be rarely used.
        hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
        hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
        easy_pos_clip_indices = []
        easy_neg_clip_indices = []
        if add_easy_negative:
            easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
            if len(easy_neg_pool) >= max_n:
                easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
                easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
            else:  # copy the hard ones
                easy_pos_clip_indices = hard_pos_clip_indices
                easy_neg_clip_indices = hard_neg_clip_indices

        pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
        neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
        return pos_clip_indices, neg_clip_indices, score_array
    
    def get_video_feat(self, video_id):
        feats = []
        for feat_file in self.feat_files:
            with h5py.File(feat_file, 'r') as f:
                feat = f[video_id][:self.max_video_l].astype(np.float32)
                if self.normalize_video:
                    feat = F.normalize(torch.from_numpy(feat), dim=1)
                feats.append(feat)
        min_len = min([len(e) for e in feats])
        feats = [e[:min_len] for e in feats]
        return torch.cat(feats, dim=1)
    

def collate(batch):
    batched_data = dict()
    num_clips = []
    video_feat = []
    video_id = []
    duration = []
    norm_moment = []
    norm_span = []
    sentence = []
    words_id = []
    words_weight = []
    unknown_mask = []
    words_label = []
    saliency_label = []
    clip_mask = []
    pos_idx = []
    neg_idx = []
    qid = []

    for e in batch:
        num_clips.append(e['num_clips'])
        video_feat += e['video_feat']
        video_id += e['video_id']
        duration += e['duration']
        sentence += e['sentence']
        words_id += e['words_id']
        words_weight += e['words_weight']
        unknown_mask += e['unknown_mask']
        words_label += e['words_label']
        qid += e['qid']
        if 'norm_moment' in e:
            norm_moment += e['norm_moment']
            norm_span += e['norm_span']
            saliency_label += e['saliency_label']
            clip_mask += e['clip_mask']
            pos_idx += e['pos_idx']
            neg_idx += e['neg_idx']

    batched_data['num_clips'] = torch.LongTensor(num_clips)
    batched_data['video_feat'], batched_data['video_mask'] = \
        pad_sequences_1d(video_feat, dtype=video_feat[0].dtype, fixed_length=None)
    batched_data['duration'] = torch.Tensor(duration)
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
    
    if len(norm_moment) > 0:
        batched_data['norm_moment'] = [dict(moments=moment) for moment in norm_moment]
        batched_data['norm_span'] = [dict(spans=span) for span in norm_span]
        batched_data['saliency_label'], _ = \
            pad_sequences_1d(saliency_label, dtype=saliency_label[0].dtype, fixed_length=None)
        batched_data['clip_mask'], _ = \
            pad_sequences_1d(clip_mask, dtype=clip_mask[0].dtype, fixed_length=None)
        if pos_idx[0] is not None:
            batched_data['pos_idx'] = torch.stack(pos_idx, dim=0)
            batched_data['neg_idx'] = torch.stack(neg_idx, dim=0)
    
    batched_data['qid'] = qid
    batched_data['video_id'] = video_id
    batched_data['sentence'] = sentence

    return batched_data