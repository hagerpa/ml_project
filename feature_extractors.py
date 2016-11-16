"""This file contains all the feature extractors that I implemented. A feature extractor
will reseve a vocabulary, i.e. a set of strings, and a example, i.e. a list of stirngs
and will return a dictionary, where keys are the names of the features and the values the
specific feature value for that example.
"""

def simple_features(vocab, words):
    """The simplies feature extractor. It simply checks whether a word of the given vocabulary
    is contained in the given example or not.
    INPUT:
    - vocab: a set of strings
    - words: a list of strings
    """
    features = {}
    for v in vocab:
        features[v] = v in words
    return features