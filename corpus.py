import csv, pickle, re
from collections import Counter
from tempfile import TemporaryFile

from feature_extractors import multinomial_model
from filters import run_filters_sentence, run_filters_words

import numpy as np
from scipy import sparse
from scipy.sparse import find
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import nltk.data

def load_from_file():
    with open("corpus.pkl", "rb") as f:
        return pickle.load(f)

class corpus:
    def __init__(self, categories, freeze_random=False, random_seed=None, n_folds=3):
        self.file_loaded = False
        self.processed = False
        self.cats = categories
        self.word_filters = []
        self.sentence_filters = []
        self.RANDOM_SEED = random_seed
        self.FREEZE_RANDOM = freeze_random
        self.N_FOLDS = n_folds
    
    def save(self):
        with open("corpus.pkl", "wb+") as f:
            pickle.dump(self, f)
    
    def load(self, filename_questions, filename_categories):
        ### Reding category assignments ###
        with open(filename_categories, 'r') as qcatfile:
            qcatreader = csv.reader(qcatfile); next(qcatreader) # skipping column titles
            
            q_to_c = {}
            for qcat in qcatreader:
                c_id = self.cats[ qcat[1] ]
                q_id = int( qcat[2] )
                q_to_c[ q_id ] = c_id
        
        ### Reading questions ###
        with open(filename_questions, 'r') as file:
            file_content = file.read()    
            regexps = [(r"\\\"", "'")]
            for find, replace in regexps:
                file_content, n = re.subn(find, replace, file_content)
        
        with TemporaryFile("w+") as qfile:
            qfile.write(file_content)
            qfile.seek(0)
            
            qreader = csv.reader(qfile)
            next(qreader)
            questions, y = [], []
            for row in qreader:
                if len(row) != 21: continue # skipping rows with wrong collum length
                if row[15] != "0": continue # skipping questions marked as deleted
                if int(row[0]) in q_to_c:
                    questions += [ row[4].lower() ]
                    y += [ q_to_c[ int(row[0]) ] ]
        
        ### Add the example questions from the category-descriptions  ###
        tokenizer = nltk.data.load('tokenizers/punkt/german.pickle')
        descriptions = self.cats.descriptions_
        
        for c, d in zip(descriptions.keys(), descriptions.values()):
            sentences = tokenizer.tokenize( d )
            sentences = [s for s in sentences if s[-1]=="?"]
            labels = [ self.cats[ str(c) ] for s in sentences ]
            questions  += sentences
            y += labels
        
        y = np.array(y, dtype=int)
        questions = np.array(questions)
        
        self.questions = questions
        self.y = y
        self.freqVecCats = np.bincount(y)/len(y)
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
            y = self.y
        else:
            if not random_state:
                random_state = np.random.randint(2**32 - 1)
            
            sss = StratifiedShuffleSplit(n_splits = 1, test_size=None, train_size=corpus_size, random_state=random_state)
            y_parent = self.cats.get_parent( self.y )
            tr, te = next(sss.split( y_parent, y_parent ))
            
            if test_corpus:
                self.test_corpus = (self.questions[te], self.y[te])
            
            questions = self.questions[tr]
            y = self.y[tr]
        
        questions = run_filters_sentence(questions, sentence_filters)
        questions = np.array( run_filters_words(questions, word_filters) )
        
        term_space = set( w for d in questions for w in d )
        term_space = np.array( list(term_space) )
        X = multinomial_model(questions, term_space)
        
        keep, _ = find_duplicates(X, key=-self.freqVecCats[ y ])
        X, y = X[keep], y[keep]
        _, J, _ = find(X); J = list(set(J));
        X = X[:,J]
        
        self.questions = None
        self.sentence_filters += sentence_filters
        self.word_filters += word_filters
        self.X_all = X
        self.y = y
        self.term_space = term_space
        self.SUPPORT = J
        self.processed = True
        return self
    
    def simple_split(self, test_size=0, random_seed=None, remove_duplicates=True):
        """ Devides the processed corpus into a training- and test-set. The percentage of examples from the
        corpus which form the test-set is specified by test_size. If it is set to 0 the entier corpus will
        form the training set. Note that the terms-space is build only by words accuring in the training set.
        """
        if not(self.FREEZE_RANDOM & bool(self.RANDOM_SEED)):
            self.RANDOM_SEED = np.random.randint(2**32 - 1)
        
        if test_size == 0:
            X_tr, y_tr = self.X_all[:-1], self.y[:-1]
            X_te, y_te = self.X_all[-1:], self.y[-1:]
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.RANDOM_SEED)
            y_parent = self.cats.get_parent( self.y )
            tr, te = next( sss.split( y_parent, y_parent ) )
            X_tr, y_tr = self.X_all[ tr ], self.y[tr]
            X_te, y_te = self.X_all[ te ], self.y[te]
        
        I, J, _ = find(X_tr); J = list(set(J));
        X_tr, X_te = X_tr[:, J], X_te[:, J]
        
        self.SUPPORT = J
        self.X_tr, self.y_tr = X_tr, y_tr
        self.X_te, self.y_te = X_te, y_te
        
        return (X_tr, y_tr), (X_te, y_te)
    
    def __iter__(self):
        """ This mehtod provides an efficient way to creat n_folds on the corpus for cross validation.
        The diffrenrence towards a standart stratified split is that it removes terms form the feature-
        space that only appear in the test set, which results in a more realistic generalization. """
        
        if not(self.FREEZE_RANDOM & bool(self.RANDOM_SEED)):
            self.RANDOM_SEED = np.random.randint(2**32 - 1)
        
        skf = StratifiedKFold(n_splits=self.N_FOLDS, shuffle=True, random_state=self.RANDOM_SEED)
        y_parent = self.cats.get_parent( self.y )
        self.skf = skf.split( y_parent, y_parent )
        
        self.current_fold_ = 0
        return self
    
    def __next__(self):
        """ Iterate through folds for a Stratified split,
        returning pairs of (X_tr, y_tr), (X_te, y_te).
        """
        self.current_fold_ += 1
        if self.current_fold_ > self.N_FOLDS:
            raise StopIteration
        
        tr, te = next(self.skf)
        X_tr, y_tr = self.X_all[ tr ], self.y[ tr ]
        X_te, y_te = self.X_all[ te ], self.y[ te ]
        
        I, J, _ = find(X_tr); J = list(set(J));
        X_tr, X_te = X_tr[:, J], X_te[:, J]
        
        return (X_tr, y_tr), (X_te, y_te)
        
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
        documents = run_filters_words(documents, self.word_filters)
        X = multinomial_model(documents, self.term_space)
        
        return X[:, self.SUPPORT]
    
    def size(self):
        """Returns a tuple, where the first component correspondes to the training-set size and
        the second to the test-set size."""
        return self.X_all.shape
    
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