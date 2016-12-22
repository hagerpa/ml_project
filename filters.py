import re
from nltk import snowball
from nltk.corpus import stopwords
from nltk import word_tokenize

""" This file contains only filters, those are methods that have as a input and output either a
list of strings, or just a string. Such methods are ment to remove unnecessaire elements, or filter 
information in another way (e.g. by stemming).
"""

def run_filters(raw_documents, sentence_filters, word_filters):
    documents = []
    for d in raw_documents:
        if type(d) == str:
            for filt in sentence_filters: d = filt(d)
            d = word_tokenize(d)
        if type(d) == list:
            for filt in word_filters: d = filt(d)
        documents += [d]
    return documents

def punctuation_filter(sentence):
    """A simple filter that essentialy substitutes punctuation symbols by whitespaces and at the same
    time gets track of seqqences of severeal whitespaces. Also substitutes '&' by 'und'. It returns a
    sting.
    
    Key Argument:
    sentence -- must be a single string
    """
    filtered_sentence = sentence
    regexps = [(r"[.!?,\-()\[\]\\]", " "),
                (r"&", "und"),
                (r" {2,}", " ")]
    for find, replace in regexps:
        filtered_sentence = re.sub(find, replace, filtered_sentence)
    
    return filtered_sentence
    
def small_word_filter(words, min_=1):
    """This filter removes very small words given list of words. Words are small if they contain less
    or equal then min. 
    """
    new_words = []
    for w in words:
        if(len(w) > min_):
            new_words += [w]
    return new_words
    
def year_tracker(words):
    """This filter tries to find numbersequence that represent years, hower it ristricts itself to find
    years in the range 1700-2199."""
    new_words = []
    for w in words:
        new_word = re.sub(r"^[1][789][0-9]{2}$", "jahreszahl", w) # for 1700-1999
        new_word = re.sub(r"^[2][01][0-9]{2}$", "jahreszahl", new_word) # for 2000-2199
        new_words += [new_word]
    return new_words
    
def stemming_filter(words):
    """This filter essentally executes stemming which is adapted from the snwoball package of nltk.
    Not that as input one has to hand in a already tokenized sentence, i.e. a list of strings, rather
    than a single string."""
    stemmer = snowball.GermanStemmer()
    return [stemmer.stem(w) for w in words]

def stopword_filter(words):
    """Thise method removes all stopwords/functionwords contained in nltk.corpus.stopwords.words('german')"""
    new_words = []
    for w in words:
        if w in stopwords.words("german"): continue
        else: new_words += [w]
    return new_words
    