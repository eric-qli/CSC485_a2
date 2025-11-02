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

import typing as T
from string import punctuation

from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize


def deepest():
    """Find and print the synset with the largest maximum depth along with its
    depth on each of its hyperonym paths.

    Returns:
        None
    """
    max_depth = -1
    deepest_syn = None
    depths_for_deepest = []

    for s in wn.all_synsets():
        paths = s.hypernym_paths()
        depths = [len(p) - 1 for p in paths]
        dmax = max(depths)
        if dmax > max_depth:
            max_depth = dmax
            deepest_syn = s
            depths_for_deepest = depths

    print(f"Deepest synset: {deepest_syn.name()} — {deepest_syn.definition()}")
    print(f"Depths on each path: {depths_for_deepest}")
    print(f"Maximum depth: {max_depth}")



def superdefn(synset: str) -> T.List[str]:
    """Get the "superdefinition" of a synset. (Yes, superdefinition is a
    made-up word. All words are made up.)

    We define the superdefinition of a synset to be the list of word tokens,
    here as produced by word_tokenize, in the definitions of the synset, its
    hyperonyms, and its hyponyms.

    Args:
        synset (str): The name of the synset to look up

    Returns:
        list of str: The list of word tokens in the superdefinition of s

    Examples:
        >>> superdefn('toughen.v.01')
        ['make', 'tough', 'or', 'tougher', 'gain', 'strength', 'make', 'fit']
    """
    s = wn.synset(synset)

    defs = [s.definition()]
    defs += [h.definition() for h in s.hypernyms()]
    defs += [h.definition() for h in s.hyponyms()]

    return word_tokenize(' '.join(defs))


def stop_tokenize(s: str) -> T.List[str]:
    """Word-tokenize and remove stop words and punctuation-only tokens.

    Args:
        s (str): String to tokenize

    Returns:
        list[str]: The non-stopword, non-punctuation tokens in s

    Examples:
        >>> stop_tokenize('The Dance of Eternity, sir!')
        ['Dance', 'Eternity', 'sir']
    """
    s_clean = "".join(c for c in s if c not in punctuation)
    return [w for w in word_tokenize(s_clean) if w.lower() not in stopwords.words('english')]


if __name__ == '__main__':
    import doctest
    doctest.testmod()
