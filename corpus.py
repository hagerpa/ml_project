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
    
    def process(self, sentence_filters=None,word_filters=None,
    corpus_size=-1, test_corpus=False):
        """ This method runs given filters on the raw set of documents. One can choose a stratified
        subset of the documents by specifying corpus_size, note however that this set size can not
        simply be extended. One needs to reload the corpus. If test_corpus is True and corpus_size
        smaller then the overall corpus, then these question will be saved in corpus.test_corpus and
        not be contained in train or test-set. """
        
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
            if test_corpus:
                self.test_corpus = (self.questions[corpus_size:], self.y[corpus_size:])
            questions = self.questions[:corpus_size]
            self.y = self.y[:corpus_size]
        
        questions = run_filters_sentence(questions, sentence_filters)
        
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
        
        questions = np.array( run_filters_words(questions, word_filters) )
        
        self.questions = questions
        self.sentence_filters += sentence_filters
        self.word_filters += word_filters
        
        self.processed = True
        return self
    
    def simple_split(self, test_size=0, random_seed=None):
        """ Devides the processed corpus into a training- and test-set. The percentage of examples from the
        corpus which form the test-set is specified by test_size. If it is set to 0 the entier corpus will
        form the training set. Note that the terms-space is build only by words accuring in the training set.
        """
        m = len(self.cats) # number of categories
        
        if random_seed:
            self.random_seed = random_seed
        else:
            self.random_seed = np.random.randint(2**32 - 1)
        
        if test_size==0:
            tr_set, self.y_tr = self.questions[:-1], self.y[:-1]
            te_set, self.y_te = self.questions[-1:], self.y[-1:]
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.random_seed)
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
        self.in_simple_split = False
        return self
    
    def reset(self):
        """ If its necessary to reapet the iteration of fold in the cross-validation, then run
        this method first and it will set the corpus to the initial state with same folds.
        """
        if self.in_cv_split:
            self.current_fold = 0
            self.skf_iter = self.skf.split(self.y,self.y)
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
        
    def make_features(self, M=-1, vocabulary_builder=ig_nonun, feature_extractor=multinomial_model):
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
        
        return self.feature_extractor(documents, self.term_space, self.term_space_extras)
    
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