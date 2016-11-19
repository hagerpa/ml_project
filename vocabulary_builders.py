"""This file contains several methods that build a vocabulary.
A vocabulary is a list of words/strings that is somehow extracted from the corpus
and can later on be used to extract features. The standart method recieves as input
a corpus reprecented as a list of dictionries containing the pre-filtered words
(as a list) and the corresponding category. The output must be a set of words.
It also neeads the categoires used.

gerneral INPUT:
- corpus :a list of dictionries containing the pre-filtered words (as a list) and
the corresponding category
- frequencies: a nltk.FreqDist object that contatians the freqeuneces of al words
from the entier corpus
- categories: a list of the category names
"""

import itertools
import progressbar
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
    categories = corpus.cats
    information_gain = calculate_ig_values(corpus.frequencies, categories.frequencies, categories, read_from_file)
    
    # picking those M terms with the best ig-index
    vocab = set()
    for cat in categories.all_names():
        lis = [(information_gain[(w,cat)], w) for w in corpus.frequencies['all']]
        lis.sort()
        vocab = vocab.union([w for _, w in lis[-M:]])
    
    return vocab

def ig_based_non_uniform(corpus, M=1000, read_from_file=False):
    """This method builds a vocabulary by selecting M words form the overall corpus
    vocabulary with the best "information-gain" index. Therefore the information gain index
    is first calculated, or if so specified read from file.
    
    """
    categories = corpus.cats
    information_gain = calculate_ig_values(corpus.frequencies, categories.frequencies, categories, read_from_file)
    
    # picking those M terms with the best ig-index
    vocab = set()
    for cat in categories.all_names():
        lis = [(information_gain[(w,cat)], w) for w in corpus.frequencies['all']]
        lis.sort()
        m = int(corpus.cats.frequencies.freq(cat) * M)
        if m == 0: m = 1
        vocab = vocab.union([w for _, w in lis[-m:]])
    
    return vocab

def calculate_ig_values(frequencies, cat_frequencies, categories, read_from_file):
    information_gain = {}
    # caluculating/reading the information-gain index
    if (not read_from_file):
        
        for w in frequencies['all']:
            for cat in categories.all_names():
                information_gain[(w, cat)] = 0
                
                if frequencies[cat].freq(w) > 0:
                    information_gain[(w, cat)] = frequencies[cat].freq(w) * cat_frequencies.freq(cat) \
                                            * np.log( frequencies[cat].freq(w) / frequencies['all'].freq(w) )
                if frequencies[cat].freq(w) < 1:
                    information_gain[(w, cat)] += (1 - frequencies[cat].freq(w)) * cat_frequencies.freq(cat) \
                                            * np.log( (1 - frequencies[cat].freq(w)) / (1 -frequencies['all'].freq(w)) )
        ## Saving into pickle files
        ig_file = open('information_gain.pkl', 'wb+')
        pickle.dump(information_gain, ig_file)
    else:
        ## Loading from pickle files
        ig_file = open('information_gain.pkl', 'rb')
        information_gain = pickle.load(ig_file)
    
    return information_gain
    
    
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
    lis = [ (cc[(w,cat)]**p, w) for w in corpus.frequencies['all'] for cat in categories.all_names()].sort()
    
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
