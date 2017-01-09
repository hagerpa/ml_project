import corpus as corpus_class
import categories, filters, vocabulary_builders
from feature_extractors import multinomial_model, tfidf
from filters import std_filters

import numpy as np
from scipy import sparse
import itertools as it

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score as f1_scorer_org
from sklearn.metrics import confusion_matrix
    
def f1_scorer(y_true, y_pred):
    return f1_scorer_org(y_true, y_pred, average="macro")
    
def CV(corpus, clf_class, clf_name, clf_params, feat_params, n_folds=3, scorer=f1_scorer, skipping_rule=None):
    corpus.cv_split(n_folds)
    
    best_score, best_clf, best_f_params = -1, None, None
    
    for c_par, f_par in it.product(clf_params, feat_params):
        if not (skipping_rule is None):
            if skipping_rule(c_par, f_par):
                continue
        
        sco = 0
        for coprus in corpus:
            corpus.make_features(**f_par)
            
            clf = clf_class(**c_par)
            clf.fit(corpus.X_tr, corpus.y_tr)
            y_pred = clf.predict(corpus.X_te)
            
            sco += scorer(corpus.y_te, y_pred)
        corpus.reset()
        
        sco = sco/n_folds
        if sco > best_score:
            best_score = sco
            best_clf = clf
            best_f_params = f_par
    
    return best_clf, best_f_params

def MultinomialNB_params(corpus):
    clf_class = MultinomialNB
    clf_name = "MultinomialNB"
    clf_params = [{"alpha":M} for M in np.logspace(-5,1,7)]

    M_max = np.log2(len(corpus.all_terms))-1
    feat_params = [{"M":int(M)} for M in np.logspace(6,M_max,7, base=2)]
    feat_params += [{"M":-1}]

    return clf_class, clf_name, clf_params, feat_params

def LogisticRegression_params():
    clf_class = LogisticRegression
    clf_name = "LogisticRegression"
    clf_params = [{"C": round(C, 2), "penalty": 'l1', "solver": 'liblinear'} for C in np.logspace(-1,2,16, base=2)]
    #clf_params = [{"C":M, "penalty": 'l1', "solver": 'liblinear'} for M in np.linspace(2,4,5)]
    feat_params = [{"M":-1, "feature_extractor":multinomial_model}]
    
    return clf_class, clf_name, clf_params, feat_params

def RandomForest_params(corpus):
    clf_class = RandomForestClassifier
    clf_name = "RandomForestClassifier"
    #clf_params = [{"max_features": int(C), "n_estimators": 20} for C in np.logspace(0,8,9, base=2)]
    clf_params = [{"max_features": int(C), "n_estimators": 20, "criterion": 'entropy'} for C in np.logspace(0,4,5, base=2)]
    
    M_max = np.log2(len(corpus.all_terms))-1
    feat_params = [{"M": int(M), "feature_extractor":tfidf} for M in np.logspace(6,M_max,7, base=2)]
    feat_params += [{"M":-1, "feature_extractor":tfidf}]
    
    return clf_class, clf_name, clf_params, feat_params

def RF_skipping_rule(c_par, f_par):
    # returns True if this pair of parameter should be skipped
    if f_par["M"] == -1:
        return False
    else:
        return c_par["max_features"] > f_par["M"]
        
def train_apply_classifier(classifier = 'NaiveBayes', qfile_train = 'question_train.csv',
    qcatfile_train = 'question_category_train.csv', catfile = 'category.csv', qfile_test = 'question_test.csv', subcats = False):
    """This method performs a parameter tuning using cross validation for the specified classfier.
    After the hyper-parameter(s) are selected it returns the predicted labes for the given test-set.
    Following 3 classifiers are known to the method:
        - "NaiveBayes" (default)
        - "LogisticRegression"
        - "RandomForest"
    """
    # initalizing corpus
    corpus = corpus_class.corpus( categories.categories() )
    corpus.load(qfile_train, qcatfile_train)
    filts = std_filters()
    corpus.process(corpus_size=-1, **filts)
    corpus.simple_split(0)
    
    #corpus = corpus_class.load_from_file()
    #corpus.simple_split(0)
    
    if classifier == 'NaiveBayes':
        clf_par = MultinomialNB_params(corpus)
        clf, feat_params = CV(corpus, *clf_par, n_folds=3)
    elif classifier == 'LogisticRegression':
        clf_par = LogisticRegression_params()
        clf, feat_params = CV(corpus, *clf_par, n_folds=3)
    elif classifier == 'RandomForest':
        clf_par = RandomForest_params(corpus)
        clf, feat_params = CV(corpus, *clf_par, n_folds=3, skipping_rule=RF_skipping_rule)
    else:
        raise ValueError("The given classfier is not known to this method. Look up the doc to see which classfiers work.")
    
    # making the fit for the entier traing set
    corpus.simple_split(0)
    corpus.make_features(**feat_params)
    clf.fit(corpus.X_tr, corpus.y_tr)
    
    X_te = corpus.process_example(qfile_test)
    
    return clf.predict(X_te)

if __name__ == "__main__":
    train_apply_classifier()