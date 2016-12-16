"""This file contains several methods that select a reduced vocabulary from the term-space
of the training-set of a corpus.

gerneral INPUT:
- corpus: a corpus object, see the corpus class.
- M: a parameter by which degree the method should reduce. M has a diffrent meaning for
    the diffrent methods.
"""

import itertools
import pickle
import numpy as np

def most_common(corpus, M=100):
    """This is a simple vocabulary builder that takes the M most common words for
    each category, i.e. those words that appeared most often in the corpus
    for each categroy. M must be an integer.
    """
    vocab = set()
    
    for cat_name in corpus.cats.all_names():
        words = [w for w,_ in corpus.frequencies[cat_name].most_common(M)]
        vocab = vocab.union(words)
    
    return vocab
    
def most_common_reduced(corpus, M=100, S=10, MS=100):
    """This builder is similar to the "most_common" vocabulary builder, only it
    it removes words that appear in the MS most common words of more then S categories.
    """
    vocab = most_common(corpus, M=M)
    
    stopwords = set()
    for cat_names in itertools.combinations(corpus.cats.all_names(), S):
        cat_names = iter(cat_names)
        sub_stops = set([w for w, f in corpus.frequencies[next(cat_names)].most_common(MS)])
        for cat_name in cat_names:
            sub_stops = sub_stops.intersection( set([w for w, f in corpus.frequencies[cat_name].most_common(MS)]) )
        stopwords = stopwords.union(sub_stops)
    
    return vocab.symmetric_difference(stopwords)
    
def ig_based(corpus, M=100, read_from_file=False):
    """This method builds a vocabulary by selecting M words form the overall corpus
    vocabulary with the best "information-gain" index. Therefore the information gain index
    is first calculated, or if so specified read from file.
    """
    ig = calculate_ig_values(corpus, read_from_file)
    best_term_ids = (np.argsort(-ig, axis=0)[:M,:]).flatten()
    return corpus.all_terms[ best_term_ids ]

def ig_based_non_uniform(corpus, M=1000, read_from_file=False):
    """ This method selects a vocabulary by choosing M words form the term-space of the training set
    with the best "information-gain" values. """
    if M > len(corpus.all_terms):
        M = len(corpus.all_terms)
        raise Warning("M was greater then available terms.")
    ig = calculate_ig_values(corpus, read_from_file)
    best_term_ids = np.argsort( np.sort(-ig, axis=1)[:,0] )[:M]
    return corpus.all_terms[ best_term_ids ]

def calculate_ig_values(corpus, read_from_file):
    """ This method calculates the information gain for all terms in the training set of a given corpus.
    It returns a numpy array with ig-entries as term x category. """
    if read_from_file:
        return np.load("information_gain.npy")
    else:
        X, t, c = corpus.freqMatrix, corpus.freqVecTerms, corpus.freqVecCats
        ig = np.log( ((X.T / t).T)**(X*c) )
        ig += np.log( ( ((1-X).T / (1-t)).T)**(c - X*c) )
        np.save("information_gain",ig)
        return ig

def xi_square_based(corpus, M=100, read_from_file=False):
    return cc_based(corpus, M=M, read_from_file=read_from_file, xi_square=True)

def xi_square_based_overall(corpus, M=1000, read_from_file=False):
    return cc_based(corpus, M=M, read_from_file=read_from_file, xi_square=True)

def cc_based(corpus, M=100, read_from_file=False, xi_square=False):
    """This method builds a vocabulary by selecting M words form the overall corpus
    vocabulary with the best "correlation coefficient" index. Therefore the xi-squre index
    is first calculated, or if so specified read from file.
    
    """ 
    p = 2 if xi_square else 1
    
    categories = corpus.cats
    cc = calculate_cc_values(corpus, read_from_file)
    
    # picking those M terms with the best cc vlaues, for each category
    vocab = set()
    for cat in categories.all_names():
        lis = [ (cc[(w,cat)]**p, w) for w in corpus.frequencies['all']]
        lis.sort()
        vocab = vocab.union([w for _, w in lis[-M:]])
    
    return vocab

def cc_based_overall(corpus, M=1000, read_from_file=False, xi_square=False):
    """This method builds a vocabulary by selecting M words form the overall corpus
    vocabulary with the best "xi-square" index. Therefore the xi-squre index
    is first calculated, or if so specified read from file.

    """
    p = 2 if xi_square else 1
    
    categories = corpus.cats
    cc = calculate_cc_values(corpus, read_from_file)

    # picking those M terms with the best xi-square values, which is just cc^2
    lis = [ (cc[(w,cat)]**p, w) for w in corpus.frequencies['all'] for cat in categories.all_names() ]
    lis.sort()
    
    return set( [w for _, w in lis[-M:]] )

def calculate_cc_values(corpus, read_from_file):
    cc = {}
    cat_freq = corpus.cats.frequencies
    t_freq = corpus.frequencies["all"]
    
    # caluculating/reading the correlation coefficient (cc) index
    if (not read_from_file):
        
        for w in corpus.frequencies['all']:
            for cat in corpus.cats.all_names():
                cc[(w, cat)] = 0
                
                if (cat_freq.freq(cat) <= 0) | (cat_freq.freq(cat) >= 1): continue
                
                a = ( corpus.frequencies[cat].freq(w) * cat_freq.freq(cat) )
                b = ( ( 1 - t_freq.freq(w) ) - (1 - corpus.frequencies[cat].freq(w)) * cat_freq.freq(cat) ) / (1 - cat_freq.freq(cat))
                c = 1 - b
                d = 1 - a
                e = t_freq.freq(w) * (1 - t_freq.freq(w)) * cat_freq.freq(cat) * (1 - cat_freq.freq(cat))
                g = len(corpus.te_set)
                
                cc[(w, cat)] = np.sqrt(g)*(a*b - c*d) / np.sqrt(e)
                
        ## Saving into pickle files
        cc_file = open('cc_values.pkl', 'wb+')
        pickle.dump(cc, cc_file)
    else:
        ## Loading from pickle files
        cc_file = open('cc_values.pkl', 'rb')
        cc = pickle.load(cc_file)
    
    return cc

def calculate_xi_sqaure_values(corpus, read_from_file):
    xi_square = {}
    cat_freq = corpus.cats.frequencies
    t_freq = corpus.frequencies["all"]
    
    # caluculating/reading the xi-square-gain index
    if (not read_from_file):
        
        for w in corpus.frequencies['all']:
            for cat in corpus.cats.all_names():
                xi_square[(w, cat)] = 0
                
                if (cat_freq.freq(cat) <= 0) | (cat_freq.freq(cat) >= 1): continue
                
                a = ( corpus.frequencies[cat].freq(w) * cat_freq.freq(cat) )
                b = ( ( 1 - t_freq.freq(w) ) - (1 - corpus.frequencies[cat].freq(w)) * cat_freq.freq(cat) ) / (1 - cat_freq.freq(cat))
                c = 1 - b
                d = 1 - a
                e = t_freq.freq(w) * (1 - t_freq.freq(w)) * cat_freq.freq(cat) * (1 - cat_freq.freq(cat))
                g = len(corpus.te_set)
                
                xi_square[(w, cat)] = g*(a*b - c*d)**2 / e
                
        ## Saving into pickle files
        xi_square_file = open('xi_square.pkl', 'wb+')
        pickle.dump(xi_square, xi_square_file)
    else:
        ## Loading from pickle files
        xi_square_file = open('xi_square.pkl', 'rb')
        xi_square = pickle.load(xi_square_file)
    
    return xi_square