from nltk.corpus import stopwords
import re
from collections import Counter
import numpy as np

class spell_checker:
    def __init__(self, documents):
        self.vocabulary = Counter()
        for d in documents:
            self.vocabulary += Counter(d)
        self.num_corrected = 0
        
    def correct(self, sentence):
        new_sentence = []
        for w in sentence:
            corrected_word, c_p = self.correction(w)
            if corrected_word != w: self.num_corrected += 1
            new_sentence += [corrected_word]
        return new_sentence
    
    def P(self, wp):
        word, p = wp
        pp = self.vocabulary[word] / sum(self.vocabulary.values())
        return p*pp
        
    def correction(self, word):
        return max(candidates(word), key=self.P)


def candidates(word):
    wp = (word, 1)
    return set([ wp ]).union( edits1(*wp) )#.union( edits2(*wp) )

def edits1(word, p):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    
    splits     = [(word[:i], word[i:])      for i in range(len(word) + 1)]
    deletes    = [(L + R[1:],               p*p_forget(L, R)) for L, R in splits if R]
    replaces   = [(L + c + R[1:],           p*0.7)  for L, R in splits if R for c in old_phone(R[0])]
    #transposes = [(L + R[1] + R[0] + R[2:], p*0.5)  for L, R in splits if len(R)>1]
    #inserts    = [(L + c + R,               p*p_forget(L, R)) for L, R in splits for c in letters]
    
    return set(replaces + deletes)# + inserts + transposes)
    
def p_forget(L, R):
    return 1.1 - 1/(1+len(L)/10)
    
def edits2(word, p):
    return set(e2 for e1 in edits1(word, p) for e2 in edits1(*e1))

def old_phone(char):
    btns = ["abcä", "def", "ghi", "jkl", "mnoö", "pqrsß", "tuvü", "wxyz"]
    for btn in btns:
        if char in btn:
            return re.sub(char, "", btn)
    return char