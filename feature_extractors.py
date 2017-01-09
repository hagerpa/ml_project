"""This file contains all the feature extractors that I implemented. A feature extractor
will reseve a term_sapce, i.e. a list of strings, and the documents, for which numerical
feautres will be assigned. Documents must be passed in form of an intarable with list
elements."""

import numpy as np
from sklearn.preprocessing import normalize
from scipy import sparse

def general_model(*args):
    documents, term_space, feat = args[0], args[1], args[-1]
    term_space_inv = {term_space[i]: i for i in range(len(term_space))}
    
    if type(documents[0]) == list:
        X = sparse.lil_matrix((len(documents), len(term_space)))
        for d, i in zip(documents, range(len(documents))):
            for t in d:
                if t in term_space_inv:
                    X[i, term_space_inv[t]] = feat(d, t)
        return X.tocsr()
    else:
        raise Exception("Documents must be an itarable, with list elements.")

def bernoulli_model(*args):
    """ The ith component of the feature vector is 1 if the ith word of the term-space is
    contained in the document, otherwise its set to 0. """
    if (type(args[0]) == sparse.csr.csr_matrix):
        return args[0] > 0
    elif len(args) in [2,3]:
        def feat(d, t): return t in d
        return general_model(*args, feat)
    else:
        raise Exception("Argument not understood. Either to iterables, or sparse csr matrix")
    
def multinomial_model(*args):
    """ The ith component of the feature vector is k if the ith word in the document is
    the kth word in the term-space. If the ith word of the document is not contained
    in the term-space, it is skipped. """
    if (type(args[0]) == sparse.csr.csr_matrix):
        return args[0]
    elif len(args) in [2,3]:
        def feat(d, t): return d.count(t)
        return general_model(*args, feat)
    else:
        raise Exception("Argument not understood. Either to iterables, or sparse csr matrix")

def tfidf(*args):
    """ In training mode this functions returns the documents indexed by tfidf, in testing
    mode it simply returns the ducuments indexed multinomial."""
    
    if type(args[0]) == sparse.csr.csr_matrix:
        X = args[0]
        check = (len(args) == 2)
        if len(args) > 2: raise Exception("too much arguments. ",len(args)," were given.")
    elif len(args) in [2,3]:
        X = multinomial_model(args[0], args[1])
        check = (len(args) == 3)
    else:
        raise Exception("If first matrix is not an sparse csr type, the 2 itrables have to be passed.")
    n, d = X.shape
    
    if check:
        w = sparse.diags(args[-1]).tocsr()
        X = X.dot(w)
        return normalize(X, norm='l2', axis=1)
    else:
        w = np.log( n / (X>0).sum(axis=0) )
        w = np.array(w).flatten()
        return tfidf(X, w), w