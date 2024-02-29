import os
import json
from tqdm import tqdm

from .charades import CharadesDataset


"""
Charades-CD:
- CLIP image with clip_len=1: max_video_l = 194
- Slowfast with clip_len=1: max_video_l = 195
- Train dataset
    - CLIP text tokenizer: max_words_l = 16
    - min_clip_len = 1.8625
    - max_clip_len = 28.9

- Val dataset
    - CLIP text tokenizer: max_words_l = 16
    - min_clip_len = 
    - max_clip_len = 

- Test_iid dataset
    - CLIP text tokenizer: max_words_l = 16
    - min_clip_len = 2.0875
    - max_clip_len = 24.4

- Test_ood dataset
    - CLIP text tokenizer: max_words_l = 16
    - min_clip_len = 1.8375
    - max_clip_len = 80.8
"""


class CharadesCDDataset(CharadesDataset):
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
        split2filename = {
            "train": "charades_train.json",
            "val": "charades_val.json",
            "test_iid": "charades_test_iid.json",
            "test_ood": "charades_test_ood.json",
        }
        ann_file = os.path.join(self.ann_path, split2filename[self.split])
        annotations = []

        with open(ann_file, 'r') as f:
            json_obj = json.load(f)
            count = 0
            # max_words_l = 0
            # min_clip_len = 100000
            # max_clip_len = 0
            for video_id in tqdm(json_obj.keys(), desc=f"Load Charades-CD {self.split} annotations"):
                meta = json_obj[video_id]
                duration = meta["video_duration"]
                for i in range(len(meta["timestamps"])):
                    count += 1
                    start = meta["timestamps"][i][0]
                    end = meta["timestamps"][i][1]
                    if start > duration:
                        continue
                    if start > end:     # fix wrong annotation
                        start, end = end, start
                    if end > duration:
                        end = duration
                    moment = [start, end]
                    sentence = meta["sentences"][i]
                    words_id, words_weight, unknown_mask, words_label = \
                        self.tokenizer.tokenize(sentence, max_valid_length=self.max_words_l)
                    # max_words_l = max(max_words_l, words_id.count_nonzero())
                    # min_clip_len = min(min_clip_len, end-start)
                    # max_clip_len = max(max_clip_len, end-start)
                    data = {
                        "video_id": video_id,
                        "duration": duration,
                        "moment": moment,
                        "sentence": sentence,
                        "words_id": words_id,
                        "words_weight": words_weight,
                        "unknown_mask": unknown_mask,
                        "words_label": words_label,
                        "start_idx": int(start / self.clip_len),
                        "end_idx": int(end / self.clip_len),
                        "qid": None if self.split=="train" else count,
                        "relevant_windows": None if self.split=="train" else [moment],
                    }
                    annotations.append(data)

        return annotations
