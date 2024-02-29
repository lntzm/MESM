import os
import csv
import h5py
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from .base import BaseDataset


"""
Charades-STA:
- CLIP image with clip_len=1: max_video_l = 194
- Slowfast with clip_len=1: max_video_l = 195
- Train dataset
    - CLIP text tokenizer: max_words_l = 16
    - min_clip_len = 1.68
    - max_clip_len = 80.8

- Test dataset
    - CLIP text tokenizer: max_words_l = 16
    - min_clip_len = 1.82
    - max_clip_len = 24.3
"""

class CharadesDataset(BaseDataset):
    def __init__(self, ann_path, feat_files, split,
                 use_tef, clip_len, max_words_l, max_video_l,
                 tokenizer_type, load_vocab_pkl, bpe_path, vocab,
                 normalize_video, contra_samples,
                 recfw, vocab_size, max_gather_size):
        super().__init__(ann_path, feat_files, split,
                         use_tef, clip_len, max_words_l, max_video_l,
                         tokenizer_type, load_vocab_pkl, bpe_path, vocab,
                         normalize_video, contra_samples,
                         recfw, vocab_size, max_gather_size)
    
    def load_annotations(self):
        durations = self._load_durations()
        split2filename = {
            "train": "charades_sta_train.txt",
            "test": "charades_sta_test.txt",
        }
        ann_file = os.path.join(self.ann_path, split2filename[self.split])
        annotations = []
        with open(ann_file, 'r') as f:
            lines = f.readlines()
            for i in tqdm(range(len(lines)), desc=f"Load Charades {self.split} annotations"):
                meta = lines[i].split("##")
                video_id, start, end = meta[0].split()
                start, end = float(start), float(end)
                duration = durations[video_id]
                if start > duration:
                    continue
                if start > end:     # fix wrong annotation
                    start, end = end, start
                if end > duration:
                    end = duration
                moment = [start, end]
                if self.clip_len == -1:
                    start_idx = start / duration
                    end_idx = end / duration
                else:
                    start_idx = int(start / self.clip_len)
                    end_idx = int(end / self.clip_len)
                sentence = meta[1].rstrip()
                words_id, words_weight, unknown_mask, words_label = \
                    self.tokenizer.tokenize(sentence, max_valid_length=self.max_words_l)

                data = {
                    "video_id": video_id,
                    "duration": duration,
                    "moment": moment,
                    "sentence": sentence,
                    "words_id": words_id,
                    "words_weight": words_weight,
                    "unknown_mask": unknown_mask,
                    "words_label": words_label,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "qid": None if self.split=="train" else i,
                    "relevant_windows": None if self.split=="train" else [moment],
                }
                annotations.append(data)

        return annotations

    def _load_durations(self):
        split2filename = {
            "train": "Charades_v1_train.csv",
            "val": "Charades_v1_test.csv",
            "test": "Charades_v1_test.csv",
        }
        ann_file = os.path.join(self.ann_path, split2filename[self.split])

        with open(ann_file, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            first_line_flag = True
            durations = dict()
            for row in csv_reader:
                if not first_line_flag:
                    durations[row[0]] = float(row[-1])
                first_line_flag = False

        return durations

    def get_video_feat(self, video_id):
        feats = []
        for feat_file in self.feat_files:
            with h5py.File(feat_file, 'r') as f:
                # feat = f[video_id][:self.max_video_l].astype(np.float32)
                feat = f[video_id][:].astype(np.float32)
                if self.normalize_video:
                    feat = F.normalize(torch.from_numpy(feat), dim=1)
                feats.append(feat)
        min_len = min([len(e) for e in feats])
        feats = [e[:min_len] for e in feats]
        return torch.cat(feats, dim=1)

# max_video_l = 0
# with h5py.File(feat_file, 'r') as f:
#     for id in tqdm(f.keys()):
#         feat = f[id][:]
#         max_video_l = max(max_video_l, feat.shape[0])
# print(max_video_l)