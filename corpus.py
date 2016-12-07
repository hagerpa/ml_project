import numpy as np
import csv
import nltk
from nltk import word_tokenize
import pickle
import time

def load_from_file():
    with open("corpus.pkl", "rb") as f:
        return pickle.load(f)

class corpus:
    def __init__(self, categories):
        self.file_loaded = False
        self.processed = False
        self.cats = categories
        self.frequencies = {}
        self.word_filters = []
        self.sentence_filters = []
        
    def load(self, filename_questions, filename_categories):
        ### Reding category assignments ###
        qcatfile = open(filename_categories, 'r')
        qcatreader = csv.reader(qcatfile)
        
        next(qcatreader) # skipping column discription

        qid_to_catid = {} # mapping from question_id to the parent category_id
        category_frequency = nltk.FreqDist() # maps the category_name to its frequency

        for qcat in qcatreader:
            cat_id = int(qcat[1])
            pcat_id = self.cats.parent_id(cat_id)
            q_id = int(qcat[2])
    
            qid_to_catid[ q_id ] = pcat_id
            category_frequency[ self.cats.name(cat_id) ] += 1
        
        self.cats.frequencies = category_frequency
        
        ### Reading questions ###
        qfile = open(filename_questions, 'r')
        qreader = csv.reader(qfile)
        
        col_names = [col for col in next(qreader)]
        questions = []
        
        for row in qreader:
            if len(row) != 21: continue # skipping rows with wrong collum length
            if row[15] != "0": continue # skipping questions marked as deleted
            
            if int(row[0]) in qid_to_catid.keys(): # checking whether the question was actually assigned to a category
                cat_id = int(row[0])
                sentence = row[4].lower()
                questions += [{"sentence":sentence, "category": self.cats.name( qid_to_catid[cat_id] )}]
        
        ### Saving into pickle files ###
        q_file = open('questions.pkl', 'wb+')
        pickle.dump(questions, q_file)
        q_file.close()
        
        self.file_loaded = True
        
        return self
        
    def process(self, sentence_filters, word_filters, tr_set_size=-1, te_set_size=-1, reprocessing=False):
        if reprocessing:
            # we are running filters on already filtered sentences
            raw_questions = self.tr_set + self.te_set
        else:
            if(not self.file_loaded):
                raise Exception("Processing question befor loading them from file is not possible.")
            ## Loading raw questions from pickle file
            q_file = open('questions.pkl', 'rb')
            raw_questions = pickle.load(q_file)
        
        if tr_set_size == 0: raise ValueError("training set size cant be zero.")
        elif tr_set_size < 0: pass
        elif te_set_size < 0: pass
        else: raw_questions = raw_questions[:tr_set_size] + raw_questions[tr_set_size:tr_set_size + te_set_size]
        
        questions = []
        for q in raw_questions:
            # if we are not reprocessing: sentences might be first filtered and then tokenized
            if not reprocessing:
                sentence = q['sentence']
                # running a sequence of filters on the raw question string 
                for filt in sentence_filters:
                    sentence = filt(sentence)
                words = word_tokenize(sentence)
            else:
                words = q["words"]
            
            # running a sequence of filtes on the already tokenized sentence
            for filt in word_filters:
                words = filt(words)
            
            questions += [{"words": words, "category": q["category"]}]
            
        if not reprocessing:
            np.random.shuffle(questions)
        
        self.sentence_filters = self.sentence_filters + sentence_filters
        self.word_filters = self.word_filters + word_filters
        
        self.frequencies = {cat_name: nltk.FreqDist() for cat_name in self.cats.all_names()}
        self.frequencies["all"] = nltk.FreqDist()
        
        self.tr_set = questions[:tr_set_size]
        self.te_set = questions[tr_set_size:]
        
        for q in self.tr_set: # Now we count the frequencies, but only for words in the training set
            self.frequencies[ q["category"] ] += nltk.FreqDist( q["words"] )
            self.frequencies["all"] += nltk.FreqDist( q["words"] )
        
        self.processed = True
        
        id_to_term = np.array( list( self.frequencies['all'] ) )
        id_to_cat = np.array( self.cats.all_names() )
        
        self.freqVecTerms = np.array([ self.frequencies['all'].freq(term) for term in id_to_term ])
        self.freqVecCats = np.array([ self.cats.frequencies.freq(cat) for cat in id_to_cat ])
        self.freqMatrix = np.array( [ [self.frequencies[c].freq(t) for c in id_to_cat] for t in id_to_term] )
        
        self.id_to_term, self.id_to_cat = id_to_term, id_to_cat
        
        return self
        
    def make_features(self, term_space):
        """
        Recieves the (reduced) term-space as an np.array and returns a
        feature matrix vor the training set.
        """
        self.term_space = term_space
        
        self.X_tr = np.array([ [(t in q['words']) for q in self.tr_set] for t in term_space ])
        self.y_tr = np.array([ self.cats.internal_id( q["category"] ) for q in self.tr_set])
        
        self.X_te = np.array([ [(t in q['words']) for q in self.te_set] for t in term_space ])
        self.y_te = np.array([ self.cats.internal_id( q["category"] ) for q in self.te_set])
        return self
    
    def save(self):
        with open("corpus.pkl", "wb+") as f:
            pickle.dump(self, f)
    
    def process_example(self, raw_questions):
        X = np.zeros((len(self.term_space), len(raw_questions)))
        for q, i in zip(raw_questions, range(len(raw_questions))):
            sentence = q.lower()
            for filt in self.sentence_filters:
                sentence = filt(sentence)
            
            words = word_tokenize(sentence)
            for filt in self.word_filters:
                words = filt(words)
            print(words)
            X[:,i] = np.array( [ (t in words) for t in self.term_space ] )
        return X