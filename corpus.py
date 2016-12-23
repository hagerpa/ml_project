import numpy as np
import csv
from collections import Counter
import pickle
from feature_extractors import tfidf, multinomial_model
from vocabulary_builders import ig_based_non_uniform as ig_nonun
from sklearn.preprocessing import normalize
from filters import run_filters, std_filters
from scipy import sparse
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

def load_from_file():
    with open("corpus.pkl", "rb") as f:
        return pickle.load(f)

class corpus:
    def __init__(self, categories):
        self.file_loaded = False
        self.processed = False
        self.made_feautres = False
        self.cats = categories
        self.word_filters = []
        self.sentence_filters = []
        self.feature_extractor = None
        self.freqVecCats = np.zeros(len(categories))
        self.in_cv_split = False
        self.in_simple_split = False
    
    def save(self):
        with open("corpus.pkl", "wb+") as f:
            pickle.dump(self, f)
    
    def load(self, filename_questions, filename_categories):
        ### Reding category assignments ###
        with open(filename_categories, 'r') as qcatfile:
            qcatreader = csv.reader(qcatfile); next(qcatreader) # skipping column discription
            
            q_to_c = {}
            for qcat in qcatreader:
                c_id = self.cats[ qcat[1] ]
                q_id = int(qcat[2])
                q_to_c[ q_id ] = c_id
                self.freqVecCats[ c_id ] += 1
        
        self.freqVecCats = self.freqVecCats / np.sum(self.freqVecCats)
        
        ### Reading questions ###
        with open(filename_questions, 'r') as qfile:
            qreader = csv.reader(qfile); next(qreader)
            questions, y = [], []
            for row in qreader:
                if len(row) != 21: continue # skipping rows with wrong collum length
                if row[15] != "0": continue # skipping questions marked as deleted
                if int(row[0]) in q_to_c:
                    questions += [ row[4].lower() ]
                    y += [ q_to_c[ int(row[0]) ] ]
            
            self.y = np.array(y, dtype=int)
            self.questions = np.array(questions)
        
        self.file_loaded = True 
        return self
    
    def process(self, sentence_filters=None, word_filters=None, corpus_size=-1):
        """ This method runs given filters on the raw set of documents. One can choose a stratified
        subset of the documents by specifying corpus_size, note however that this set size can not
        simply be extended. One needs to reload the corpus."""
        
        if self.processed:
            raise Warning("Corpus is already processed. This might creat problems if corpus_size is diffrent now.")
        
        if not type(corpus_size) == int:
            raise ValueError("Corpus size must be passed as interger.")
        elif corpus_size == 0:
            raise ValueError("Corpus size can not be zero.")
        elif corpus_size < 0:
            questions = self.questions
        elif corpus_size > len(self.questions):
            raise Warning("Corpus size was greater then avalible documents. Took all avalibales.")
            questions = self.questions
        else:
            questions = self.questions[:corpus_size]
            self.y = self.y[:corpus_size]
        
        questions = np.array( run_filters(questions, sentence_filters, word_filters) )
        self.questions = questions
        self.sentence_filters += sentence_filters
        self.word_filters += word_filters
        
        self.processed = True
        return self
    
    def simple_split(self, test_size=0):
        m = len(self.cats) # number of categories
        
        if test_size==0:
            tr_set, self.y_tr = self.questions[:-1], self.y[:-1]
            te_set, self.y_te = self.questions[-1:], self.y[-1:]
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
            tr_set_ids, te_set_ids = next( sss.split(self.y, self.y) )
        
            tr_set, self.y_tr = self.questions[tr_set_ids], self.y[tr_set_ids]
            te_set, self.y_te = self.questions[te_set_ids], self.y[te_set_ids]
        
        frequencies = count_freqs(tr_set, self.y_tr, m)
        self.all_terms = np.array( list( frequencies['all'] ) )
        self.freqVecTerms, self.freqMatrix = make_freq_vec(frequencies, self.all_terms, m)
        
        self.X_all_tr = multinomial_model(tr_set, self.all_terms)
        self.X_all_te = multinomial_model(te_set, self.all_terms)
        
        freqVecCats = np.array( [ np.sum(self.y_tr==i) for i in range(m) ] )
        self.freqVecCats = freqVecCats / sum(freqVecCats)
        
        self.in_simple_split=True
        return self
    
    def cv_split(self, n_folds):
        """ This mehtod provides an efficient way to creat n_folds on the corpus for cross validation.
        It Counts term_frequencies for each folds seperatly so that they can simply be merged. After
        running this method, the corpus becomes an iterable object, which for each iteration creats a new
        traing-/ test-set split. """
        if not (type(n_folds) == int):
            raise Value_Error("Number of folds must be integer.")
        elif not n_folds > 2:
            raise Value_Error("Number of folds must greater then 2.")
        
        fold_freqs = {}
        skf = StratifiedKFold(n_splits=n_folds, shuffle=False)
        for i, (_, fold) in zip(range(n_folds), skf.split(self.y,self.y)):
            docs = self.questions[fold]
            labels = self.y[fold]
            fold_freqs[i] = count_freqs(docs, labels, len(self.cats))
        
        self.fold_freqs = fold_freqs
        self.n_folds = n_folds
        self.current_fold = 0
        self.skf = skf
        self.skf_iter = skf.split(self.y,self.y)
        
        self.in_cv_split = True
        return self
    
    def reset(self): # if the itteration should be repeated with same folds
        if self.in_cv_split:
            self.current_fold = 0
            self.skf_iter = self.skf.split(self.y,self.y)
        return self
    
    def __iter__(self):
        if not self.in_cv_split:
            raise Exception("Can not iterate befor running .cv_split(n_folds)!")
        return self
    
    def __next__(self):
        if not self.in_cv_split:
            raise Exception("Can not iterate befor running .cv_split(n_folds)!")
        if self.current_fold > self.n_folds:
            raise StopIteration
        
        m = len(self.cats)
        
        train, test = next(self.skf_iter)
        tr_set, self.y_tr = self.questions[train], self.y[train]
        te_set, self.y_te = self.questions[test], self.y[test]
        
        frequencies = {i: Counter() for i in range(m)}
        frequencies['all'] = Counter()
        for i in range(self.n_folds):
            if not (i == self.current_fold):
                frequencies['all'] += self.fold_freqs[i]['all']
                for c in range(m):
                    frequencies[c] += self.fold_freqs[i][c]
        
        self.all_terms = np.array( list( frequencies['all'] ) )
        
        self.freqVecTerms, self.freqMatrix = make_freq_vec(frequencies, self.all_terms, m)
        
        self.X_all_tr = multinomial_model(tr_set, self.all_terms)
        self.X_all_te = multinomial_model(te_set, self.all_terms)
        
        freqVecCats = np.array( [ np.sum(self.y_tr==i) for i in range(m) ] )
        self.freqVecCats = freqVecCats / sum(freqVecCats)
        
        self.current_fold += 1
        return self
        
    def make_features(self, M=-1, vocabulary_builder=ig_nonun, feature_extractor=tfidf):
        """ Creats sparse (csr) feature matricies for the training- and test-set applying the 
        given feature models for feature selcetion and extraction. """
        self.vocabulary_builder = vocabulary_builder, M
        self.feature_extractor = feature_extractor
        
        tmsp_ids = vocabulary_builder(self, M)
        self.term_space = self.all_terms[tmsp_ids]
        
        self.X_tr = self.X_all_tr[:,tmsp_ids]
        self.X_te = self.X_all_te[:,tmsp_ids]
        
        out = feature_extractor(self.X_tr)
        
        if type(out) == tuple:
            self.X_tr, self.term_space_extras = out
        else:
            self.X_tr, self.term_space_extras = out, None
        
        self.X_te = feature_extractor(self.X_te, self.term_space_extras)
        self.made_feautres = True
        return self
    
    def process_example(self, raw_documents):
        """ Pass this method an iterable of documents (strings) and it will process the docuemnts
        acordingly to the training set. It will return a sparse (csr) feature matrx."""
        
        if not self.made_feautres:
            raise Exception("First run .make_features(), to make features for the training set.")
        documents = run_filters(raw_documents, self.sentence_filters, self.word_filters)
        return self.feature_extractor(documents, self.term_space, self.term_space_extras)
    
    def size(self):
        if self.in_simple_split | self.in_cv_split:
            n, _ = self.X_all_tr.shape
            m, _ = self.X_all_te.shape
            return n, m
        else: return None
    
    def __str__(self):
        out = "";
        out += "{0} categories. \n".format(len(self.cats))
        
        if self.file_loaded:
            out += "{0} docuemnts loaded from file. \n".format(len(self.questions))
            out += "processed: {0} \n".format(self.processed)
        else:
            out += "loaded from file: False"
            return out
        if self.processed:
            out += "\t Training-set, Test-set size: {0} \n".format(self.size())
            out += "\t\t sentence_filters: {0} \n".format([f.__name__ for f in self.sentence_filters])
            out += "\t\t word_filters: {0} \n".format([f.__name__ for f in self.word_filters])
        else:
            return out
        out += "made numeric features: {0} \n".format(self.made_feautres)
        if self.made_feautres:
            out += "\t vocabulary_builder, M: {0}, {1} \n".format(self.vocabulary_builder[0].__name__, self.vocabulary_builder[1])
            out += "\t feature_extractor: {0} \n".format(self.feature_extractor.__name__)
        return out

def count_freqs(documents, lables, n_cats):
    frequencies = {i: Counter() for i in range(n_cats)}
    frequencies['all'] = Counter()
    for d, y in zip(documents, lables):
        frequencies[y] += Counter(d)
        frequencies['all'] += Counter(d)
    return frequencies

def make_freq_vec(frequencies, terms, n_cats):
    freqVecTerms = np.array([ frequencies['all'][t] for t in terms])
    freqVecTerms = freqVecTerms / sum(freqVecTerms)
    
    freqMatrix = np.array([ [frequencies[c][t] for c in range(n_cats)] for t in terms], dtype='float64')
    freqMatrix = normalize(freqMatrix, norm='l1', axis=0)
    
    return freqVecTerms, freqMatrix