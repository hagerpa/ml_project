import numpy as np
import csv
from collections import Counter
import pickle
from feature_extractors import tfidf, multinomial_model
from vocabulary_builders import ig_based_non_uniform as ig_nonun
from sklearn.preprocessing import normalize
from filters import run_filters
from scipy import sparse

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
        """ Run this method after loading documents form file. It will apply the given sentence and word
        filters to the documents and count frequencies. This method must be excuted befor making features."""
        
        if reprocessing:
            if self.processed:
                raw_questions = np.append(self.tr_set, self.te_set)
                y = np.append(self.y_tr, self.y_te)
            else:
                raise Exception("Reprocessing documents befor processing them is not possible.")
        else:
            if self.file_loaded:
                raw_questions = self.questions
                y = self.y
            else:
                raise Exception("Processing documents befor loading them is not possible. Run .load(-filenames-) first.")
        
        if tr_set_size == 0: raise ValueError("training set size cant be zero.")
        elif tr_set_size < 0: pass
        elif te_set_size < 0: pass
        else:
            raw_questions = raw_questions[:tr_set_size + te_set_size]
            y = y[:tr_set_size + te_set_size]
        
        questions = np.array( run_filters(raw_questions, sentence_filters, word_filters) )
        self.sentence_filters = self.sentence_filters + sentence_filters
        self.word_filters = self.word_filters + word_filters
        
        if not reprocessing:
            Qy = np.array([questions, y]).T
            np.random.shuffle(Qy)
            questions, y = Qy.T
            y = np.array(y, dtype=int)
        
        self.tr_set, self.y_tr = questions[:tr_set_size], y[:tr_set_size]
        self.te_set, self.y_te = questions[tr_set_size:], y[tr_set_size:]
        
        frequencies = {i: Counter() for i in range(len(self.cats))}
        frequencies['all'] = Counter()
        for q, c in zip(self.tr_set, self.y_tr):
            frequencies[c] += Counter(q)
            frequencies['all'] += Counter(q)
        
        self.all_terms = np.array( list( frequencies['all'] ) )
        
        freqVecTerms = np.array([ frequencies['all'][t] for t in self.all_terms ])
        self.freqVecTerms = freqVecTerms / sum(freqVecTerms)
        
        freqMatrix = np.array([ [frequencies[c][t] for c in range(len(self.cats))] for t in self.all_terms], dtype='float64')
        self.freqMatrix = normalize(freqMatrix, norm='l1', axis=0)
        
        self.X_all_tr = multinomial_model(self.tr_set, self.all_terms)
        self.X_all_te = multinomial_model(self.te_set, self.all_terms)
        
        self.processed = True
        return self
        
    def make_features(self, M=-1, vocabulary_builder=ig_nonun, feature_extractor=tfidf):
        """ Creats sparse (csr) feature matricies for the training- and test-set applying the 
        given feature models for feature selcetion and extraction. """
       
        self.vocabulary_builder = ig_nonun, M
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
        if self.processed: return len(self.tr_set), len(self.te_set)
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