import os
import csv
import string
import pickle
from tqdm import tqdm
from dataset import CLIPTokenizer, Vocabulary


def word_tokenize(text):
    """ Tokenize text on word level: converting to lower case, eliminating punctuations.
    Args:
        text: str
    Returns:
        [str_word]
    """
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    tokens = str(text).lower().translate(translator).strip().split()
    return tokens


def count_charades_tokenized_id(ann_path, split, max_words_l, tokenizer=None):
    def _load_durations():
        split2filename = {
            "train": "Charades_v1_train.csv",
            "val": "Charades_v1_test.csv",
            "test": "Charades_v1_test.csv",
        }
        ann_file = os.path.join(ann_path, split2filename[split])
        with open(ann_file, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            first_line_flag = True
            durations = dict()
            for row in csv_reader:
                if not first_line_flag:
                    durations[row[0]] = float(row[-1])
                first_line_flag = False
        return durations
    
    durations = _load_durations()
    split2filename = {
        "train": "charades_sta_train.txt",
        "test": "charades_sta_test.txt",
    }
    ann_file = os.path.join(ann_path, split2filename[split])
    vocab_dict = {}
    with open(ann_file, 'r') as f:
        lines = f.readlines()
        for i in tqdm(range(len(lines)), desc=f"Load Charades {split} annotations"):
            meta = lines[i].split("##")
            video_id, start, end = meta[0].split()
            start, end = float(start), float(end)
            duration = durations[video_id]
            if start > duration:
                continue
            sentence = meta[1].rstrip()
                        
            if tokenizer is None:
                words = word_tokenize(sentence)
                for word in words:
                    if word in vocab_dict:
                        vocab_dict[word] += 1
                    else:
                        vocab_dict[word] = 1
            else:
                words_id, words_weight, unknown_mask, words_label = \
                    tokenizer.tokenize(sentence, max_valid_length=max_words_l)
                for j in range(1, words_id.count_nonzero() - 1):
                    tokenize_id = int(words_id[0,j])
                    if tokenize_id in vocab_dict:
                        vocab_dict[tokenize_id] += 1
                    else:
                        vocab_dict[tokenize_id] = 1
    return vocab_dict


def count_CLIP_tokenizer():
    tokenizer = CLIPTokenizer(
        recfw=False,
        id2label=None,
        bpe_path="./pretrained_models/bpe_simple_vocab_16e6.txt.gz"
    )
    vocab_dict = count_charades_tokenized_id(
        ann_path="./data/charades/annotations",
        split="train",
        max_words_l=16,
        tokenizer=tokenizer,
    )
    val_vocab_dict = count_charades_tokenized_id(
        ann_path="./data/charades/annotations",
        split="test",
        max_words_l=16,
        tokenizer=tokenizer,
    )
    for key, value in val_vocab_dict.items():
        if key in vocab_dict:
            vocab_dict[key] += value
        else:
            vocab_dict[key] = value
    vocab_count = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
    with open("./data/charades/annotations/CLIP_tokenized_count.txt", 'w') as f:
        for vocab in vocab_count:
            f.write(f"{vocab[0]} {vocab[1]}\n")


def count_GloVe_tokenizer():
    vocab_dict = count_charades_tokenized_id(
        ann_path="./data/charades/annotations",
        split="train",
        max_words_l=16,
        tokenizer=None,
    )
    val_vocab_dict = count_charades_tokenized_id(
        ann_path="./data/charades/annotations",
        split="test",
        max_words_l=16,
        tokenizer=None,
    )
    for key, value in val_vocab_dict.items():
        if key in vocab_dict:
            vocab_dict[key] += value
        else:
            vocab_dict[key] = value
    vocabs = set(vocab_dict.keys())
    vocab = Vocabulary(vocabs)
    vocab_count = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
    with open("./data/charades/annotations/GloVe_tokenized_count.txt", 'w') as f:
        for v in vocab_count:
            f.write(f"{v[0]} {vocab.wtoi[v[0]]} {v[1]}\n")

if __name__ == "__main__":
    count_CLIP_tokenizer()
    count_GloVe_tokenizer()