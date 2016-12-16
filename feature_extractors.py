"""This file contains all the feature extractors that I implemented. A feature extractor
will reseve a term_sapce, i.e. a list of strings, and the documents, for which numerical
feautres will be assigned. Documents are in the form of list of -either list of strings
-or dictionarries containing the key 'words' reffering to a list of strings. All the
methods cotain an option parameter which dicriminates between 'featuering' the training
set or the test set. """

import numpy as np
    
def general_model(documents, term_space, feat):
    if type(documents[0]) == dict:
        if 'words' in documents[0].keys():
            return np.array([ [ feat(d['words'], t) for d in documents ] for t in term_space ])
        else:
            raise Exception("When documents are provided as dicts, they must contain the key 'words'.")
    elif type(documents[0]) == list:
         return np.array([ [ feat(d, t) for d in documents ] for t in term_space ])
    else:
        raise Exception("Documents must either be dictionaries containing the key 'words' or directly lists.")

def bernoulli_model(documents, term_space):
    """ The ith component of the feature vector is 1 if the ith word of the term-space is
    contained in the document, otherwise its set to 0. """
    
    def feat(d, t): return t in d
    return general_model(documents, term_space, feat)
    
def multinomial_model(documents, term_space):
    """ The ith component of the feature vector is k if the ith word in the document is
    the kth word in the term-space. If the ith word of the document is not contained
    in the term-space, it is skipped. """
    
    def feat(d, t): return d.count(t)
    return general_model(documents, term_space, feat)

def tfidf(documents, term_space, *args):
    """ In training mode this functions returns the documents indexed by tfidf, in testing
    mode it simply returns the ducuments indexed multinomial."""
    
    Y = multinomial_model(documents, term_space).T
    if len(args) > 0:
        Y = Y * args[0]
        X = Y.T
        norm = np.linalg.norm(Y, axis=1) # normalizing where norm isn't zero.
        X[:,norm>0] = Y[norm>0,:].T / norm[norm>0]
        return X
    if len(args) == 0:
        weights = np.log( len(documents) / np.sum(Y>0, axis=0) )
        return tfidf(documents, term_space, weights), weights