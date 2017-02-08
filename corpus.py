import numpy as np
import csv
from collections import Counter
import pickle
from feature_extractors import tfidf, multinomial_model
from vocabulary_builders import ig_based_non_uniform as ig_nonun
from sklearn.preprocessing import normalize
from filters import run_filters_sentence, run_filters_words, std_filters, stopword_filter
from scipy import sparse
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import spell_checker as spell_checker_class
from scipy.sparse import find

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
        self.removed_duplicates = False
    
    def save(self):
        with open("corpus.pkl", "wb+") as f:
            pickle.dump(self, f)
    
    def load(self, filename_questions, filename_categories):
        ### Reding category assignments ###
        with open(filename_categories, 'r') as qcatfile:
            qcatreader = csv.reader(qcatfile); next(qcatreader) # skipping column discription
            
            q_to_c = {}
            for qcat in qcatreader:
                c_id = int( self.cats[ qcat[1] ] )
                q_id = int( qcat[2] )
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
    
    def process(self, sentence_filters=None,word_filters=None, corpus_size=1.0, test_corpus=False, random_state=None):
        """ This method runs given filters on the raw set of documents. One can choose a stratified
        subset of the documents by specifying corpus_size, note however that this set size can not
        simply be extended. One needs to reload the corpus. If test_corpus is True and corpus_size
        smaller then the overall corpus, then these question will be saved in corpus.test_corpus and
        not be contained in train or test-set. """
        
        if self.processed:
            raise Warning("Corpus is already processed. This might creat problems if corpus_size is diffrent now.")
        if corpus_size == 0:
            raise ValueError("Corpus size can not be zero.")
        elif (corpus_size < 0)|(corpus_size >= 1):
            questions = self.questions
        else:
            if not random_state:
                random_state = np.random.randint(2**32 - 1)
            
            sss = StratifiedShuffleSplit(n_splits = 1, test_size=None, train_size=corpus_size, random_state=random_state)
            y_parent = self.cats.get_parent( self.y )
            tr, te = next(sss.split( y_parent, y_parent ))
            
            if test_corpus:
                self.test_corpus = (self.questions[te], self.y[te])
            
            questions = self.questions[tr]
            self.y = self.y[tr]
        
        questions = run_filters_sentence(questions, sentence_filters)
        questions = np.array( run_filters_words(questions, word_filters) )
        
        all_terms = set( w for d in questions for w in d )
        all_terms = np.array( list(all_terms) )
        X = multinomial_model(questions, all_terms)
        
        keep, _ = find_duplicates(X, key=-self.freqVecCats[self.y])
        X = X[keep]
        self.y = self.y[keep]
        
        self.questions = questions
        self.sentence_filters += sentence_filters
        self.word_filters += word_filters
        self.X_all = X
        self.all_terms = all_terms
        self.processed = True
        return self
    
    def simple_split(self, test_size=0, random_seed=None, remove_duplicates=True):
        """ Devides the processed corpus into a training- and test-set. The percentage of examples from the
        corpus which form the test-set is specified by test_size. If it is set to 0 the entier corpus will
        form the training set. Note that the terms-space is build only by words accuring in the training set.
        """
        m = len(self.cats) # number of categories
        
        if random_seed:
            self.random_seed = random_seed
        else:
            self.random_seed = np.random.randint(2**32 - 1)
        
        if test_size == 0:
            X_tr, y_tr = self.X_all[:-1], self.y[:-1]
            X_te, y_te = self.X_all[-1:], self.y[-1:]
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.random_seed)
            y_parent = self.cats.get_parent( self.y )
            tr, te = next( sss.split( y_parent, y_parent ) )
            X_tr, y_tr = self.X_all[ tr ], self.y[tr]
            X_te, y_te = self.X_all[ te ], self.y[te]
        
        I, J, _ = find(X_tr); J = list(set(J));
        X_tr, X_te = X_tr[:, J], X_te[:, J]
        terms = self.all_terms[J]
        
        self.X_tr, self.y_tr = X_tr, y_tr
        self.X_te, self.y_te = X_te, y_te
        self.term_space = terms
        self.in_simple_split = True
        self.in_cv_split = False
        return self
    
    def cv_split(self, n_folds, random_seed=None):
        """ This mehtod provides an efficient way to creat n_folds on the corpus for cross validation.
        It Counts term_frequencies for each folds seperatly so that they can simply be merged. After
        running this method, the corpus becomes an iterable object, which for each iteration creats a new
        traing-/ test-set split. """
        
        if not (type(n_folds) == int):
            raise ValueError("Number of folds must be integer.")
        elif not n_folds > 2:
            raise ValueError("Number of folds must greater then 2.")
        
        fold_freqs = {}
        
        if random_seed:
            self.random_seed = random_seed
        else:
            self.random_seed = np.random.randint(2**32 - 1)
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        y_parent = self.cats.get_parent( self.y )
        
        self.n_folds = n_folds
        self.current_fold = 0
        self.skf = skf.split( y_parent, y_parent )
        self.in_cv_split = True
        self.in_simple_split = False
        return self
    
    def reset(self):
        """ If its necessary to reapet the iteration of fold in the cross-validation, then run
        this method first and it will set the corpus to the initial state with same folds.
        """
        if self.in_cv_split:
            self.current_fold = 0
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
            y_parent = self.cats.get_parent( self.y )
            self.skf = skf.split( y_parent, y_parent )
        return self
    
    def __iter__(self):
        if not self.in_cv_split:
            raise Exception("Can not iterate befor running .cv_split(n_folds)!")
        return self
    
    def __next__(self):
        """ Iterate through folds when corpus is in cv-split. """
        if not self.in_cv_split:
            raise Exception("Can not iterate befor running .cv_split(n_folds)!")
        if self.current_fold > self.n_folds:
            raise StopIteration
        
        tr, te = next(self.skf)
        X_tr, y_tr = self.X_all[ tr ], self.y[ tr ]
        X_te, y_te = self.X_all[ te ], self.y[ te ]
        
        I, J, _ = find(X_tr); J = list(set(J));
        X_tr, X_te = X_tr[:, J], X_te[:, J]
        terms = self.all_terms[J]
        
        self.X_tr, self.y_tr = X_tr, y_tr
        self.X_te, self.y_te = X_te, y_te
        self.term_space = terms
        self.current_fold += 1
        return self
        
    def process_example(self, raw_documents):
        """ Pass this method an iterable of documents (strings) and it will process the docuemnts
        acordingly to the training set. It will return a sparse (csr) feature matrx."""
        
        if type(raw_documents)==str:
            with open(raw_documents, 'r') as qfile:
                qreader = csv.reader(qfile); next(qreader)
                questions = []
                for row in qreader:
                    if len(row) != 21: continue # skipping rows with wrong collum length
                    if row[15] != "0": continue # skipping questions marked as deleted
                    questions += [ row[4].lower() ]
            raw_documents = questions
        
        documents = run_filters_sentence(raw_documents, self.sentence_filters)
        documents = np.array( run_filters_words(documents, self.word_filters) )
        return multinomial_model(documents, self.term_space)
    
    def size(self):
        """Returns a tuple, where the first component correspondes to the training-set size and
        the second to the test-set size."""
        if self.in_simple_split | self.in_cv_split:
            n, _ = self.X_all_tr.shape
            m, _ = self.X_all_te.shape
            return n, m
        else: return None
    
    def __str__(self):
        out = "";
        out += "{0} categories. \n".format(len(self.cats))
        
        
        out += "- loaded from file: " + str(self.file_loaded) + "\n"
        if self.file_loaded:
            out += "\t {0} docuemnts loaded from file. \n".format(len(self.questions))
        else:
            return out
        
        out += "- processed: " + str(self.processed) + "\n"
        if self.processed:
            out += "\t sentence_filters: {0} \n".format([f.__name__ for f in self.sentence_filters])
            out += "\t word_filters: {0} \n".format([f.__name__ for f in self.word_filters])
        else:
            return out
        
        if self.in_simple_split:
            out += "- corpus in simple split:" + "\n"
            out += "\t Training-set, Test-set size: {0} \n".format(self.size())
        elif self.in_cv_split:
            out += "- corpus in cv-split:" + "\n"
            out += "\t fold {0} / {1} \n".format(self.current_fold, self.n_folds)
            out += "\t Training-set, Test-set size: {0} \n".format(self.size())
        else:
            return out
        
        out += "- made numeric features: " + str(self.made_feautres) + "\n"
        if self.made_feautres:
            out += "\t vocabulary_builder, M: {0}, {1} \n".format(self.vocabulary_builder[0].__name__, self.vocabulary_builder[1])
            out += "\t feature_extractor: {0} \n".format(self.feature_extractor.__name__)
        else:
            return out
        
        return out

def find_duplicates(X, key):
    remove = np.array([], dtype=int)
    keep = np.array([], dtype=int)
    n, _ = X.shape
    
    for i in range(n):
        if i in np.concatenate((keep,remove)):
            continue
        _, J, _ = find( X[i,:] )
        if len(J) == 0:
            #remove = np.concatenate((remove, [i]))
            keep = np.concatenate((keep, [i]))
        else:
            I, _, _ = find( X[i:,J].sum(axis=1) >= len(J) )
            I = np.array([k for k in (I + i) if abs(X[i,:] - X[k,:]).sum() == 0])
            I = I[ np.argsort( key[I] ) ]
            remove = np.concatenate((remove, I[1:]))
            keep = np.concatenate((keep, I[:1]))
            
    return keep, remove
    
"""
if stopword_filter in word_filters:
    questions = run_filters_words(questions, [stopword_filter])

self.spell_checker = {}
questions = np.array( questions )
for i in range(len(self.cats)):
    exmpl_mask = ( self.y == i )
    self.spell_checker[i] = spell_checker_class.spell_checker( questions[exmpl_mask] )
    if len(self.spell_checker[i].vocabulary) < 2000:
        questions[exmpl_mask] = [ self.spell_checker[i].correct(q) for q in questions[exmpl_mask] ]

#self.spell_checker = spell_checker_class.spell_checker(questions)
#qestions = [ self.spell_checker.correct(q) for q in questions] """