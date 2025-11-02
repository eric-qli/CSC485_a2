'''
This code is provided solely for the personal and private use of students
taking the CSC485H/2501H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Jinman Zhao, Jingcheng Niu, Ruiyu Wang, Gerald Penn

All of the files in this directory and all subdirectories are:
Copyright (c) 2024 University of Toronto
'''

import typing as T
from dataclasses import dataclass, field
from pathlib import Path

import gzip
import pickle

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

import numpy as np

from torch import Tensor

from tqdm import tqdm, trange

from transformers import (AutoTokenizer, AutoModel, PreTrainedTokenizerFast,
                          PreTrainedModel)
from transformers import logging as hftf_log

hftf_log.set_verbosity_error()

# set PUB_DIR to None in order to run this outside of teach.cs
# PUB_DIR = Path('/u/csc485h/fall/pub/a2')
PUB_DIR = None


@dataclass
class WSDToken:
    wordform: str
    lemma: str
    synsets: T.Set[str] = field(default_factory=set)


TOKENIZER: PreTrainedTokenizerFast = None
BERT_MODEL: PreTrainedModel = None


def load_bert(model: str = 'bert-base-cased') -> None:
    global TOKENIZER, BERT_MODEL
    tf_dir = PUB_DIR / 'transformers' if PUB_DIR else PUB_DIR
    TOKENIZER = AutoTokenizer.from_pretrained(model, cache_dir=tf_dir,
                                              use_fast=True)
    BERT_MODEL = AutoModel.from_pretrained(model, cache_dir=tf_dir)


def run_bert(batch: T.List[T.List[str]]) -> T.Tuple[Tensor, Tensor]:
    tok = TOKENIZER(batch, is_split_into_words=True, return_tensors='pt',
                    return_offsets_mapping=True, padding=True)
    offset_mapping = tok.pop('offset_mapping')

    return BERT_MODEL(**tok)[0], offset_mapping


def _load_data(subset: str) -> T.List[T.List[WSDToken]]:
    pub_dir = PUB_DIR if PUB_DIR else Path()
    with gzip.open(f'/u/csc485h/fall/pub/corpora/wsd/{subset}.pkl.gz') as data_in:
        return pickle.load(data_in)


def load_eval() -> T.List[T.List[WSDToken]]:
    return _load_data('eval')


def load_train() -> T.List[T.List[WSDToken]]:
    return _load_data('train')


def load_word2vec() -> T.Tuple[T.Mapping[str, int], np.ndarray]:
    w2v_dir = PUB_DIR / 'w2v' if PUB_DIR else Path()
    with gzip.open(w2v_dir / 'w2v.vocab.pkl.gz') as vocab_in:
        vocab = {w: i for i, w in enumerate(pickle.load(vocab_in))}
    vectors = np.load(w2v_dir / 'w2v.npy')

    return vocab, vectors


def evaluate(corpus: T.List[T.List[WSDToken]],
             wsd_func: T.Callable[..., Synset],
             *func_args, **func_kwargs) -> T.Optional[float]:
    try:
        correct, total = 0, 0
        for sentence in tqdm(corpus, desc=(fn := wsd_func.__name__),
                             leave=False):
            for i, token in enumerate(sentence):
                if token.synsets and len(wn.synsets(token.lemma)) > 1:
                    predicted_sense = wsd_func(sentence, i, *func_args,
                                               **func_kwargs)
                    if predicted_sense.name() in token.synsets:
                        correct += 1
                    total += 1

        tqdm.write(f'{fn}: {(acc := correct / total):.1%}')
        return acc
    except NotImplementedError:
        return None


def batch_evaluate(corpus: T.Iterable[T.Collection[WSDToken]],
                   wsd_func: T.Callable[..., T.List[T.List[Synset]]],
                   *func_args, batch_size: int = 32, **func_kwargs) \
        -> T.Optional[float]:
    corpus = sorted(corpus, key=len)
    try:
        correct, total = 0, 0
        for batch_n in trange(0, len(corpus), batch_size,
                              desc=(fn := wsd_func.__name__), leave=False):
            batch = corpus[batch_n:batch_n + batch_size]
            idxs = [[i for i, token in enumerate(sentence)
                     if token.synsets and len(wn.synsets(token.lemma)) > 1]
                    for sentence in batch]
            idxs, batch = zip(*[(s_idx, sent)
                                for s_idx, sent in zip(idxs, batch) if s_idx])

            total += sum(map(len, idxs))
            preds = wsd_func(batch, idxs, *func_args, **func_kwargs)
            print(preds)
            correct += sum(sum(pred.name() in sent[w_idx].synsets
                               for pred, w_idx in zip(s_pred, s_idx))
                           for s_pred, sent, s_idx in zip(preds, batch, idxs))

        tqdm.write(f'{fn}: {(acc := correct / total):.1%}')
        return acc
    except NotImplementedError:
        return None
