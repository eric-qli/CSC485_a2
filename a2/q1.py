# STUDENT NAME: Eric Li 
# STUDENT NUMBER: 1007654307
# UTORID: lieric19

'''
This code is provided solely for the personal and private use of students
taking the CSC485H/2501H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Jinman Zhao, Jingcheng Niu, Ruiyu Wang, Gerald Penn

All of the files in this directory and all subdirectories are:
Copyright (c) 2025 University of Toronto
'''

from collections import Counter
from typing import *

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

import numpy as np
from numpy.linalg import norm

from q0 import stop_tokenize
from wsd import evaluate, load_eval, load_word2vec, WSDToken


def mfs(sent: Sequence[WSDToken], target_index: int) -> Synset:
    """Most frequent sense of a word.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. See the WSDToken class in wsd.py
    for the relevant class attributes.

    Args:
        sent (list of WSDToken): The sentence containing the word to be
            disambiguated.
        target_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The most frequent sense for the given word.
    """
    word  = sent[target_index].lemma
    synsets = wn.synsets(word)

    return synsets[0] if synsets else None


def lesk(sent: Sequence[WSDToken], target_index: int) -> Synset:
    """Simplified Lesk algorithm.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sent (list of WSDToken): The sentence containing the word to be
            disambiguated.
        target_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    best_sense = mfs(sent, target_index)
    best_score = 0
    # context = set([tok.lemma for tok in sent])

    all_sense = wn.synsets(sent[target_index].lemma)

    for sense in all_sense:
        text = sense.definition() + " " + " ".join(sense.examples())
        signature = set(stop_tokenize(text))
        context = set(stop_tokenize(" ".join(
        tok.lemma for i, tok in enumerate(sent) if i != target_index)))

        score = len(signature.intersection(context))
        if score > best_score:
            best_score = score
            best_sense = sense

    return best_sense


def lesk_ext(sent: Sequence[WSDToken], target_index: int) -> Synset:
    """Extended Lesk algorithm.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sent (list of WSDToken): The sentence containing the word to be
            disambiguated.
        target_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    best_sense = mfs(sent, target_index)
    best_score = 0
    # context = set([tok.lemma for tok in sent])

    all_sense = wn.synsets(sent[target_index].lemma)

    for sense in all_sense:
        text = sense.definition() + " " + " ".join(sense.examples())

        for r in sense.hyponyms():
            text += " " + r.definition()
            text += " " + " ".join(r.examples())

        for r in sense.member_holonyms():
            text += " " + r.definition()
            text += " " + " ".join(r.examples())

        for r in sense.part_holonyms():
            text += " " + r.definition()
            text += " " + " ".join(r.examples())

        for r in sense.substance_holonyms():
            text += " " + r.definition()
            text += " " + " ".join(r.examples())

        for r in sense.member_meronyms():
            text += " " + r.definition()
            text += " " + " ".join(r.examples())

        for r in sense.part_meronyms():
            text += " " + r.definition()
            text += " " + " ".join(r.examples())

        for r in sense.substance_meronyms():
            text += " " + r.definition()
            text += " " + " ".join(r.examples())
        
        signature = set(stop_tokenize(text))
        context = set(stop_tokenize(" ".join(
        tok.lemma for i, tok in enumerate(sent) if i != target_index)))

        score = len(signature.intersection(context))
        if score > best_score:
            best_score = score
            best_sense = sense

    return best_sense


def lesk_cos(sent: Sequence[WSDToken], target_index: int) -> Synset:
    """Extended Lesk algorithm using cosine similarity.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sent (list of WSDToken): The sentence containing the word to be
            disambiguated.
        target_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    best_sense = mfs(sent, target_index)
    best_score = 0
    # context = set([tok.lemma for tok in sent])

    all_sense = wn.synsets(sent[target_index].lemma)

    for sense in all_sense:
        text = sense.definition() + " " + " ".join(sense.examples())

        for r in sense.hyponyms():
            text += " " + r.definition()
            text += " " + " ".join(r.examples())

        for r in sense.member_holonyms():
            text += " " + r.definition()
            text += " " + " ".join(r.examples())

        for r in sense.part_holonyms():
            text += " " + r.definition()
            text += " " + " ".join(r.examples())

        for r in sense.substance_holonyms():
            text += " " + r.definition()
            text += " " + " ".join(r.examples())

        for r in sense.member_meronyms():
            text += " " + r.definition()
            text += " " + " ".join(r.examples())

        for r in sense.part_meronyms():
            text += " " + r.definition()
            text += " " + " ".join(r.examples())

        for r in sense.substance_meronyms():
            text += " " + r.definition()
            text += " " + " ".join(r.examples())
        
        signature = set(stop_tokenize(text))
        context = set(stop_tokenize(" ".join(
        tok.lemma for i, tok in enumerate(sent) if i != target_index)))

        all_unique = signature.union(context)
        unique_length = len(all_unique)

        signature_count  = Counter(signature)
        context_count = Counter(context)
        
        context_vec = [context_count[w] for w in all_unique]
        signature_vec = [signature_count[w] for w in all_unique]

        dot = np.dot(signature_vec, context_vec)
        sign_norm = np.linalg.norm(signature_vec)
        context_norm = np.linalg.norm(context_vec)

        denominator = sign_norm * context_norm
        if denominator == 0:
            score = 0
        else:
            score = dot / (sign_norm * context_norm)
            if score > best_score:
                best_score = score
                best_sense = sense

    return best_sense


def lesk_cos_onesided(sent: Sequence[WSDToken], target_index: int) -> Synset:
    """Extended Lesk algorithm using one-sided cosine similarity.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    Args:
        sent (list of WSDToken): The sentence containing the word to be
            disambiguated.
        target_index (int): The index of the target word in the sentence.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    best_sense = mfs(sent, target_index)
    best_score = 0
    # context = set([tok.lemma for tok in sent])

    all_sense = wn.synsets(sent[target_index].lemma)

    for sense in all_sense:
        text = sense.definition() + " " + " ".join(sense.examples())

        for r in sense.hyponyms():
            text += " " + r.definition()
            text += " " + " ".join(r.examples())

        for r in sense.member_holonyms():
            text += " " + r.definition()
            text += " " + " ".join(r.examples())

        for r in sense.part_holonyms():
            text += " " + r.definition()
            text += " " + " ".join(r.examples())

        for r in sense.substance_holonyms():
            text += " " + r.definition()
            text += " " + " ".join(r.examples())

        for r in sense.member_meronyms():
            text += " " + r.definition()
            text += " " + " ".join(r.examples())

        for r in sense.part_meronyms():
            text += " " + r.definition()
            text += " " + " ".join(r.examples())

        for r in sense.substance_meronyms():
            text += " " + r.definition()
            text += " " + " ".join(r.examples())
        
        signature = set(stop_tokenize(text))
        context = set(stop_tokenize(" ".join(
        tok.lemma for i, tok in enumerate(sent) if i != target_index)))

        signature = set([w for w in signature if w in context])

        all_unique = signature.union(context)
        unique_length = len(all_unique)

        signature_count  = Counter(signature)
        context_count = Counter(context)
        
        context_vec = [context_count[w] for w in all_unique]
        signature_vec = [signature_count[w] for w in all_unique]

        dot = np.dot(signature_vec, context_vec)
        sign_norm = np.linalg.norm(signature_vec)
        context_norm = np.linalg.norm(context_vec)

        denominator = sign_norm * context_norm
        if denominator == 0:
            score = 0
        else:
            score = dot / (sign_norm * context_norm)
            if score > best_score:
                best_score = score
                best_sense = sense

    return best_sense


def lesk_w2v(sent: Sequence[WSDToken], target_index: int,
             vocab: Mapping[str, int], word2vec: np.ndarray) -> Synset:
    """Extended Lesk algorithm using word2vec-based cosine similarity.

    **IMPORTANT**: when looking up the word in WordNet, make sure you use the
    lemma of the word, *not* the wordform. For other cases, such as gathering
    the context words, use the wordform. See the WSDToken class in wsd.py for
    the relevant class attributes.

    To look up the vector for a word, first you need to look up the word's
    index in the word2vec matrix, which you can then use to get the specific
    vector. More directly, you can look up a string s using word2vec[vocab[s]].

    To look up the vector for a *single word*, use the following rules:
    * If the word exists in the vocabulary, then return the corresponding
      vector.
    * Otherwise, if the lower-cased version of the word exists in the
      vocabulary, return the corresponding vector for the lower-cased version.
    * Otherwise, return a vector of all zeros. You'll need to ensure that
      this vector has the same dimensions as the word2vec vectors.

    But some wordforms are actually multi-word expressions and contain spaces.
    word2vec can handle multi-word expressions, but uses the underscore
    character to separate words rather than spaces. So, to look up a string
    that has a space in it, use the following rules:
    * If the string has a space in it, replace the space characters with
      underscore characters and then follow the above steps on the new string
      (i.e., try the string as-is, then the lower-cased version if that
      fails), but do not return the zero vector if the lookup fails.
    * If the version with underscores doesn't yield anything, split the
      string into multiple words according to the spaces and look each word
      up individually according to the rules in the above paragraph (i.e.,
      as-is, lower-cased, then zero). Take the mean of the vectors for each
      word and return that.
    Recursion will make for more compact code for these.

    Args:
        sentence (list of WSDToken): The sentence containing the word to be
            disambiguated.
        target_index (int): The index of the target word in the sentence.
        vocab (dictionary mapping str to int): The word2vec vocabulary,
            mapping strings to their respective indices in the word2vec array.
        word2vec (np.ndarray): The word2vec word vectors, as a VxD matrix,
            where V is the vocabulary and D is the dimensionality of the word
            vectors.

    Returns:
        Synset: The prediction of the correct sense for the given word.
    """
    eps = 1e-12
    D = word2vec.shape[1]

    
    def w2v_one(s: str) -> np.ndarray:
        """Single-token lookup with lowercase fallback; zero vec if OOV."""
        if s in vocab:
            return word2vec[vocab[s]]
        s_low = s.lower()
        if s_low in vocab:
            return word2vec[vocab[s_low]]
        return np.zeros(D, dtype=word2vec.dtype)

    
    def w2v_string(s: str) -> np.ndarray:
        """
        String lookup with MWE handling
        """
        if " " not in s:
            return w2v_one(s)

        with_us = s.replace(" ", "_")
        if with_us in vocab:
            return word2vec[vocab[with_us]]
        with_us_low = with_us.lower()
        if with_us_low in vocab:
            return word2vec[vocab[with_us_low]]

        parts = [p for p in s.split() if p]
        if not parts:
            return np.zeros(D, dtype=word2vec.dtype)
        vecs = [w2v_one(p) for p in parts]

        nz = [v for v in vecs if v.any()]
        return np.mean(nz, axis=0) if nz else np.zeros(D, dtype=word2vec.dtype)

    def mean_vec(tokens: set[str]) -> np.ndarray:
        """Mean of non-zero vectors for a token set; zero if none."""
        if not tokens:
            return np.zeros(D, dtype=word2vec.dtype)
        vecs = [w2v_string(t) for t in tokens]
        nz = [v for v in vecs if v.any()]
        return np.mean(nz, axis=0) if nz else np.zeros(D, dtype=word2vec.dtype)

    context_tokens = set(stop_tokenize(" ".join(tok.lemma for i, tok in enumerate(sent)
                                            if i != target_index)))
    context_vec = mean_vec(context_tokens)
    context_norm = float(np.linalg.norm(context_vec))

    best_sense = mfs(sent, target_index)
    best_score = -1.0 
    if context_norm < eps:
        return best_sense

    lemma = sent[target_index].lemma
    candidates = wn.synsets(lemma)

    for sense in candidates:
        text_parts = [sense.definition(), *sense.examples()]
        # hyponyms
        for r in sense.hyponyms():
            text_parts += " " + r.definition()
            text_parts += " " + " ".join(r.examples())
        # holonyms
        for r in sense.member_holonyms() + sense.part_holonyms() + sense.substance_holonyms():
            text_parts += " " + r.definition()
            text_parts += " " + " ".join(r.examples())
        # meronyms
        for r in sense.member_meronyms() + sense.part_meronyms() + sense.substance_meronyms():
            text_parts += " " + r.definition()
            text_parts += " " + " ".join(r.examples())

        sig_tokens = set(stop_tokenize(" ".join(text_parts)))
        if not sig_tokens:
            continue

        sig_vec = mean_vec(sig_tokens)
        sig_norm = float(np.linalg.norm(sig_vec))
        if sig_norm < eps:
            continue

        score = float(np.dot(sig_vec, context_vec) / (sig_norm * context_norm + eps))

        if score > best_score:
            best_score = score
            best_sense = sense

    return best_sense

        


if __name__ == '__main__':
    np.random.seed(1234)
    eval_data = load_eval()
    for wsd_func in [mfs, lesk, lesk_ext, lesk_cos, lesk_cos_onesided]:
        evaluate(eval_data, wsd_func)

    evaluate(eval_data, lesk_w2v, *load_word2vec())
