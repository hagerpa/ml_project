from nltk.corpus import stopwords
import re
from collections import Counter

class spell_checker:
    def __init__(self, start_vocabulary=None):
        if start_vocabulary is None:
            self.vocabulary = Counter( [" "] )
        else:
            self.vocabulary = start_vocabulary
    
    def correct(self, sentence):
        new_sentence = []
        for w in sentence:
            new_sentence += [self.correction(w)]
        self.vocabulary += Counter(new_sentence)
        return new_sentence
    
    def P(self, word):
        return self.vocabulary[word] / sum(self.vocabulary.values())

    def correction(self, word):
        return max(self.candidates(word), key=self.P)
    
    def candidates(self, word):
        return (self.known([word]) or self.known(self.edits1(word)) or [word]) # or self.known(self.edits2(word))
    
    def known(self, words):
        return set(w for w in words if w in self.vocabulary)
    
    def edits1(self, word):
        letters    = 'abcdefghijklmnopqrstuvwxyzäüöß'
        splits     = [(word[:i], word[i:])      for i in range(len(word) + 1)]
        deletes    = [L + R[1:]                 for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:]   for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]             for L, R in splits if R for c in letters]
        inserts    = [L + c + R                 for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)
        
    def edits2(self, word):
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))