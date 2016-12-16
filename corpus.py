import numpy as np
import csv
from nltk import word_tokenize
from collections import Counter
import pickle
from feature_extractors import tfidf
from sklearn.preprocessing import normalize

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
            self.questions, y = [], []
            for row in qreader:
                if len(row) != 21: continue # skipping rows with wrong collum length
                if row[15] != "0": continue # skipping questions marked as deleted
                if int(row[0]) in q_to_c:
                    self.questions += [ row[4].lower() ]
                    y += [ q_to_c[ int(row[0]) ] ]
            
            self.y = np.array(y)
        
        self.file_loaded = True 
        return self
    
    def process(self, sentence_filters, word_filters, tr_set_size=-1, te_set_size=-1, reprocessing=False):
        if reprocessing:
            if self.processed:
                raw_questions = np.append(self.tr_set, self.te_set)
                y = np.append(self.y_tr, self.y_te)
            else:
                raise Exception("Reprocessing question befor processing them is not possible.")
        else:
            if self.file_loaded:
                raw_questions = self.questions
                y = self.y
            else:
                raise Exception("Processing question befor loading them from file is not possible.")
        
        if tr_set_size == 0: raise ValueError("training set size cant be zero.")
        elif tr_set_size < 0: pass
        elif te_set_size < 0: pass
        else:
            raw_questions = raw_questions[:tr_set_size + te_set_size]
            y = y[:tr_set_size + te_set_size]
        
        questions = []
        for q in raw_questions:
            if type(q) == str:
                for filt in sentence_filters: q = filt(q)
                q = word_tokenize(q)
            if type(q) == list:
                for filt in word_filters: q = filt(q)
            questions += [q]
        questions = np.array(questions)
        
        if not reprocessing:
            Qy = np.array([questions, y]).T
            np.random.shuffle(Qy)
            questions, y = Qy.T
            y = np.array(y, dtype=int)
        
        self.sentence_filters = self.sentence_filters + sentence_filters
        self.word_filters = self.word_filters + word_filters
        
        self.tr_set, self.y_tr = questions[:tr_set_size], y[:tr_set_size]
        self.te_set, self.y_te = questions[tr_set_size:], y[tr_set_size:]
        
        frequencies["all"] = Counter()
        for i in range(len(self.cats)):
            frequencies[i] = Counter()
            for q in self.tr_set[self.y_tr == i]:
                frequencies[i] += Counter(q)
            frequencies["all"] += frequencies[i]
        
        self.all_terms = np.array( list( frequencies['all'] ) )
        
        freqVecTerms = np.array([ frequencies['all'][t] for t in self.all_terms ])
        self.freqVecTerms = freqVecTerms / sum(freqVecTerms)
        
        freqMatrix = np.array([ [frequencies[c][t] for c in range(len(self.cats))] for t in self.all_terms])
        self.freqMatrix = normalize(freqMatrix, norm='l1', axis=0)
        
        self.processed = True
        return self
        
    def make_features(self, term_space, feature_extractor=tfidf):
        """ Creats features for the training and test-set applying the given feature model. """
        self.feature_extractor = feature_extractor
        self.term_space = term_space
        
        out = feature_extractor(self.tr_set, term_space)
        
        if type(out) == tuple:
            self.X_tr, extras = out
            self.X_te = feature_extractor(self.te_set, term_space, extras)
            self.term_space_extras = extras
        else:
            self.X_tr = out
            self.term_space_extras = None
            self.X_te = feature_extractor(self.te_set, term_space)
        
        self.made_feautres = True
        return self
    
    def process_example(self, raw_questions):
        if not self.made_feautres:
            raise Exception("First run .make_features(), to make features for the training set.")
        X = np.zeros((len(self.term_space), len(raw_questions)))
        for q, i in zip(raw_questions, range(len(raw_questions))):
            sentence = q.lower()
            for filt in self.sentence_filters:
                sentence = filt(sentence)
            
            words = word_tokenize(sentence)
            for filt in self.word_filters:
                words = filt(words)
            
            if self.term_space_extras is None:
                X[:,i] = self.feature_extractor([words], self.term_space)
            else:
                X[:,i] = self.feature_extractor([words], self.term_space, self.term_space_extras).flatten()
        return X