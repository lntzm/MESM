import os
import gzip
import html
import ftfy
import nltk
import torch
import string
import regex as re
from functools import lru_cache
from typing import Union, List
from collections import OrderedDict
import numpy as np


@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class CLIPTokenizer(object):
    def __init__(self, recfw, id2label,
                 bpe_path: str = default_bpe()):
        self.recfw = recfw
        self.id2label = id2label
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        encoded_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            for bpe_token in self.bpe(token).split(' '):
                bpe_tokens.append(bpe_token.replace('</w>', ''))
                encoded_tokens.append(self.encoder[bpe_token])

        weights = []
        for _, tag in nltk.pos_tag(bpe_tokens):
            if ('NN' in tag) or ('VB' in tag) or ('JJ' in tag) or ('RB' in tag):
                weights.append(2)
            else:
                weights.append(1)
            # token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return encoded_tokens, weights

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text
    
    def tokenize(self,
                 texts: Union[str, List[str]],
                 context_length: int = 77,
                 max_valid_length: int = 32):
        """
        Returns the tokenized representation of given input string(s)

        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize

        context_length : int
            The context length to use; all CLIP models use 77 as the context length

        max_valid_length:

        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
        """
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.encoder["<|startoftext|>"]
        eot_token = self.encoder["<|endoftext|>"]
        all_tokens = []
        weights = []
        unknowns = []
        labels = []
        for text in texts:
            txt_token, weight = self.encode(text)
            txt_token = txt_token[:max_valid_length-2]
            weight = weight[:max_valid_length-2]
            all_tokens.append([sot_token] + txt_token + [eot_token])
            weights.append([0] + weight + [0])
            if self.recfw:
                unknowns.append(
                    [False] + [False if token in self.id2label else True for token in txt_token] + [False]
                )
                labels.append(
                    [self.id2label['<start>']] + 
                    [self.id2label[token] if token in self.id2label else self.id2label['<unknown>'] for token in txt_token] +
                    [self.id2label['<end>']]
                )

        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        result_weight = torch.zeros(len(all_tokens), max_valid_length, dtype=torch.long)
        unknown_mask = torch.zeros_like(result_weight, dtype=torch.bool) if self.recfw else None
        result_label = result_weight.clone() if self.recfw else None

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)
            result_weight[i, :len(tokens)] = torch.tensor(weights[i])
            if self.recfw:
                unknown_mask[i, :len(tokens)] = torch.tensor(unknowns[i])
                result_label[i, :len(tokens)] = torch.tensor(labels[i])

        return result, result_weight, unknown_mask, result_label


class Vocabulary(object):
    """ Natural language vocabulary.
    """
    def __init__(self, *word_set):
        """
        Args:
            *word_set: any number of {str}
        """
        self.special_words = ["<PAD>", "<UNK>"]
        self.wtoi, self.itow = OrderedDict(), OrderedDict()
        self._build(word_set)

    def _build(self, word_set_tuple):
        # 0: <PAD>, 1: <UNK>
        for i, word in enumerate(self.special_words):
            self.wtoi[word] = i
            self.itow[i] = word

        words = set()
        for x in word_set_tuple:
            words.update(x)

        for i, word in enumerate(sorted(words)):
            j = i + len(self.special_words)
            self.wtoi[word] = j
            self.itow[j] = word

    def __len__(self):
        return len(self.wtoi)


class GloVeSimpleTokenizer():
    def __init__(self, recfw, id2label, vocab):
        self.recfw = recfw
        self.id2label = id2label
        self.vocab = vocab
    
    def encode(self, text):
        words = self.split_words(text)
        txt_tokens = []
        weights = []
        for word, tag in nltk.pos_tag(words):
            txt_tokens.append(self.vocab.wtoi.get(word, 1))
            if ('NN' in tag) or ('VB' in tag) or ('JJ' in tag) or ('RB' in tag):
                weights.append(2)
            else:
                weights.append(1)
        return txt_tokens, weights

    def tokenize(self,
                 texts: Union[str, List[str]],
                 context_length: int = 77,
                 max_valid_length: int = 32):
        if isinstance(texts, str):
            texts = [texts]

        all_tokens = []
        weights = []
        unknowns = []
        labels = []
        for text in texts:
            txt_token, weight = self.encode(text)
            txt_token = txt_token[:max_valid_length]
            weight = weight[:max_valid_length]
            all_tokens.append(txt_token)
            weights.append(weight)
            if self.recfw:
                unknowns.append(
                    [False if token in self.id2label else True for token in txt_token]
                )
                labels.append(
                    [self.id2label[token] if token in self.id2label else self.id2label['<unknown>'] for token in txt_token]
                )

        result = torch.zeros(len(all_tokens), max_valid_length, dtype=torch.long)
        result_weight = torch.zeros(len(all_tokens), max_valid_length, dtype=torch.long)
        unknown_mask = torch.zeros_like(result_weight, dtype=torch.bool) if self.recfw else None
        result_label = result_weight.clone() if self.recfw else None

        for i, tokens in enumerate(all_tokens):
            # if len(tokens) > context_length:
            #     raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)
            result_weight[i, :len(tokens)] = torch.tensor(weights[i])
            if self.recfw:
                unknown_mask[i, :len(tokens)] = torch.tensor(unknowns[i])
                result_label[i, :len(tokens)] = torch.tensor(labels[i])

        return result, result_weight, unknown_mask, result_label

    def split_words(self, text):
        """ Tokenize text on word level: converting to lower case, eliminating punctuations.
        Args:
            text: str
        Returns:
            [str_word]
        """
        translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
        tokens = str(text).lower().translate(translator).strip().split()
        return tokens


class NLTKTokenizer(GloVeSimpleTokenizer):
    def __init__(self, recfw, id2label, vocab):
        super().__init__(recfw, id2label, vocab)

    def encode(self, sentence):
        txt_tokens = []
        weights = []
        for word, tag in nltk.pos_tag(nltk.tokenize.word_tokenize(sentence)):
            word = word.lower()
            txt_tokens.append(self.vocab.wtoi.get(word, 1))
            if ('NN' in tag) or ('VB' in tag) or ('JJ' in tag) or ('RB' in tag):
                weights.append(2)
            else:
                weights.append(1)
        return txt_tokens, weights


class NLTKTokenizerWithFeature():
    def __init__(self, recfw, id2label, vocab):
        self.recfw = recfw
        self.id2label = id2label
        self.vocab = vocab
    
    def encode(self, sentence):
        words = []
        weights = []
        for word, tag in nltk.pos_tag(nltk.tokenize.word_tokenize(sentence)):
            word = word.lower()
            if word in self.vocab['w2id']:
                words.append(word)
                if ('NN' in tag) or ('VB' in tag) or ('JJ' in tag) or ('RB' in tag):
                    weights.append(2)
                else:
                    weights.append(1)
        words_feat = [self.vocab['id2vec'][self.vocab['w2id'][w]].astype(np.float32) for w in words]
        return words, words_feat, weights

    def tokenize(self,
                 texts: Union[str, List[str]],
                 context_length: int = 77,
                 max_valid_length: int = 32):
        if isinstance(texts, str):
            texts = [texts]

        all_tokens = []
        weights = []
        unknowns = []
        labels = []
        for text in texts:
            words, words_feat, weight = self.encode(text)
            words = words[:max_valid_length]
            words_feat = words_feat[:max_valid_length]
            words_feat = torch.from_numpy(np.stack(words_feat))
            weight = weight[:max_valid_length]
            all_tokens.append(words_feat)
            weights.append(weight)
            if self.recfw:
                unknowns.append(
                    [False if word in self.id2label else True for word in words]
                )
                labels.append(
                    [self.id2label[word] if word in self.id2label else self.id2label['<unknown>'] for word in words]
                )

        result = torch.zeros(len(all_tokens), max_valid_length, words_feat.shape[-1])
        result_weight = torch.zeros(len(all_tokens), max_valid_length, dtype=torch.long)
        unknown_mask = torch.zeros_like(result_weight, dtype=torch.bool) if self.recfw else None
        result_label = result_weight.clone() if self.recfw else None

        for i, tokens in enumerate(all_tokens):
            # if len(tokens) > context_length:
            #     raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens), :] = torch.tensor(tokens)
            result_weight[i, :len(tokens)] = torch.tensor(weights[i])
            if self.recfw:
                unknown_mask[i, :len(tokens)] = torch.tensor(unknowns[i])
                result_label[i, :len(tokens)] = torch.tensor(labels[i])

        return result, result_weight, unknown_mask, result_label