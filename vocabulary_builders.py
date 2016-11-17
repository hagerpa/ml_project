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

def most_common(frequencies, categories, M=100):
    """This is a simple vocabulary builder that takes the M most common words for
    each category, i.e. those words that appeared most often in the corpus
    for each categroy. M must be an integer.
    """
    vocab = set()
    
    for cat_name in categories:
        words = [w for w,_ in frequencies[cat_name].most_common(M)]
        vocab = vocab.union(words)
    
    return vocab
    
def most_common_reduced(frequencies, categories, M=100, S=10, MS=100):
    """This builder is similar to the "most_common" vocabulary builder, only it
    it removes words that appear in the MS most common words of more then S categories.
    """
    vocab = most_common(frequencies, categories, M=M)
    
    stopwords = set()
    for cat_names in itertools.combinations(categories, S):
        cat_names = iter(cat_names)
        sub_stops = set([w for w, f in frequencies[next(cat_names)].most_common(MS)])
        for cat_name in cat_names:
            sub_stops = sub_stops.intersection( set([w for w, f in frequencies[cat_name].most_common(MS)]) )
        stopwords = stopwords.union(sub_stops)
    
    return vocab.symmetric_difference(stopwords)
    
def ig_based(frequencies, cat_frequencies, categories, M=100, read_from_file=False):
    """This method builds a vocabulary by selecting M words form the overall corpus
    vocabulary with the best "information-gain" index. Therefore the information gain index
    is first calculated, or if so specified read from file.
    
    """
    information_gain = calculate_ig_values(frequencies, cat_frequencies, categories, read_from_file)
    # picking those M terms with the best ig-index
    vocab = set()
    for cat in categories:
        lis = [(information_gain[(w,cat)], w) for w in frequencies['all']]
        lis.sort()
        vocab = vocab.union([w for _, w in lis[-M:]])
    
    return vocab

def ig_based_non_uniform(frequencies, cat_frequencies, categories, M=1000, read_from_file=False):
    """This method builds a vocabulary by selecting M words form the overall corpus
    vocabulary with the best "information-gain" index. Therefore the information gain index
    is first calculated, or if so specified read from file.
    
    """
    information_gain = calculate_ig_values(frequencies, cat_frequencies, categories, read_from_file)
    # picking those M terms with the best ig-index
    vocab = set()
    for cat in categories:
        lis = [(information_gain[(w,cat)], w) for w in frequencies['all']]
        lis.sort()
        m = int(cat_frequencies.freq(cat) * M)
        vocab = vocab.union([w for _, w in lis[-m:]])
    return vocab


def calculate_ig_values(frequencies, cat_frequencies, categories, read_from_file):
    information_gain = {}
    # caluculating/reading the information-gain index
    if (not read_from_file):
        bar = progressbar.ProgressBar()
        
        for w in bar(frequencies['all']):
            for cat in cats.all_names():
                information_gain[(w, cat)] = 0
                
                if frequencies[cat].freq(w) > 0:
                    information_gain[(w, cat)] = frequencies[cat].freq(w) * cat_frequencies.freq(cat) \
                                            * np.log( frequencies[cat].freq(w) / vocabulary['all'].freq(w) )
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