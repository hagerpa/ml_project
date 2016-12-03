import numpy as np
import csv
import nltk
from nltk import word_tokenize
import pickle

class corpus:
    def __init__(self, categories):
        self.file_loaded = False
        self.processed = False
        self.cats = categories
        self.frequencies = {}
        self.applied_word_filers = []
        self.applied_sentence_filers = []
        
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
    
    def process(self, sentence_filters, word_filters, tr_set_size=-1, te_set_size=0, reprocessing=False):
        if reprocessing:
            # we are running filters on already filtered sentences
            raw_questions = self.tr_set + self.te_set
        else:
            if(not self.file_loaded):
                raise.Exception("Processing question befor loading them from file is not possible.")
            ## Loading raw questions from pickle file
            q_file = open('questions.pkl', 'rb')
            raw_questions = pickle.load(q_file)
        
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
        
        self.frequencies = {cat_name: nltk.FreqDist() for cat_name in self.cats.all_names()}
        self.frequencies["all"] = nltk.FreqDist()
        
        self.tr_set = questions[:tr_set_size]
        self.te_set = questions[-tr_set_size:]
        
        for q in self.tr_set: # Now we count the frequencies, but only for words in the training set
            self.frequencies[ q["category"] ] += nltk.FreqDist( q["words"] )
            self.frequencies["all"] += nltk.FreqDist( q["words"] )
        
        self.processed = True
        
        
        n_of_terms, n_of_cats = len(self.frequencies['all']), len(self.cats.all_names)
        self.index_to_term, self.term_to_index = {}, {}
        self.index_to_cat, self.cat_to_index = {}, {}
        self.freqMatrix = np.zeros(n_of_terms, n_of_cats)
        self.freqVecTerms = np.zeros(n_of_terms)
        self.freqVecCats = np.zeros(n_of_cats)
        
        for c_index, cat in zip(range(n_of_cats), self.cats.all_names()):
            self.index_to_cat[index] = cat
            self.cat_to_index[cat] = index
            self.freqVecCats[index] = self.cats.frequencies[cat]
        
        for t_index, term in zip(range(n_of_terms), self.frequencies['all']):
            self.index_to_term[t_index] = term
            self.term_to_index[term] = t_index
            
            self.freqMatrix[t_index, :] = [ frequencies[ index_to_cat[c_index] ].freq(term) for c_index in range(n_of_cats) ]
            self.freqVecTerms = self.frequencies['all'].freq(term)
            