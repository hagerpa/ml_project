from scipy.sparse import find
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from numpy.random import randint as RandInt
import numpy as np

class TrainingSupport:
    """ Discards terms form freautres-space that are not contained in any
    example of the training-set. Works like a sklearn transformer.
    """
    def __init__(self):
        self.J = None
    
    def fit(self, X, y=None):
        _, J, _ = find( X )
        self.J = list(set(J))
        return self
    
    def transform(self, X, y=None):
        return X[:,self.J]
    
    def fit_transfrom(self, X, y=None):
        _, J, _ = find( X )
        self.J = list(set(J))
        return X[:,j]
    
    def get_params(self, deep):
        return {}

class ProbabilitySpace:
    """ Objects of this class essentally concatenate probabilty outputs form a list of
    classifiers to form a new feature-space. See RandForestPS for usage examples.
    """
    def __init__(self, estimators, n_folds=6, random_seed=None, bootstrap=False):
        self.N_FOLDS = n_folds
        self.BOOTSTRAP = bootstrap
        self.estimators_ = estimators
        if random_seed:
            self.RANDOM_SEED = random_seed
        else:
            self.RANDOM_SEED = RandInt(2**32 - 1)
        
    def fit_transform(self, X, y):
        if self.BOOTSTRAP: 
            self.skf_generator = StratifiedShuffleSplit(n_splits=self.N_FOLDS, random_state=self.RANDOM_SEED)
        else:
            self.skf_generator = StratifiedKFold(n_splits=self.N_FOLDS, shuffle=True, random_state=self.RANDOM_SEED)
        skf = self.skf_generator.split( y, y )
        
        n, _ = X.shape
        d = len(self.estimators_)*len(set(y))
        Xp = np.zeros((n, d))
        yp = np.zeros(n)
        
        P, lP = [], []
        for tr, te in skf:
            for clf in self.estimators_:
                clf.fit(X[ tr ], y[ tr ])
            A = [ clf.predict_proba( X[ te ] ) for clf in self.estimators_]
            Xp[te, :] = np.concatenate(A, axis=1)
            yp[te] = y[ te ]
        
        ## fitting all the classifiers on the entire available set ##
        for clf in self.estimators_:
            clf.fit(X, y)
        
        return Xp, yp
        
    def transform(self, X, y=None):
        A = [ clf.predict_proba( X ) for clf in self.estimators_]
        return np.concatenate(A, axis=1), y

from sklearn import svm
    
class LinearSVC( svm.LinearSVC ):
    def predict_proba(self, X):
        """returns the absolute value of the decision boundary.
    """
        return abs(self.decision_function(X))

from sklearn.ensemble import RandomForestClassifier

class RandForestPS():
    """ This classifier fist builds a features space concistiong of predicted probabilties from
    a list of classiefiers and then trains ont then space. One can use it like a classifier
    form the sklearn package.
    """
    def __init__(self, estimators=None, *args, n_folds=8, bootstrap=False, **kwargs):
        self.RF = RandomForestClassifier(*args, **kwargs)
        self.estimators = estimators
        self.n_folds = n_folds
        self.pbb_space = ProbabilitySpace(estimators, n_folds = n_folds, bootstrap = bootstrap)
    
    def fit(self, X, y):
        Xp, yp = self.pbb_space.fit_transform(X, y)
        self.RF.fit(Xp, yp)
        return self
    
    def predict(self, X, y=None):
        Xp, yp = self.pbb_space.transform(X, y)
        return self.RF.predict(Xp)
        
    def predict_proba(self, X):
        Xp, _ = self.pbb_space.transform(X)
        return self.RF.predict_proba(Xp)
    
    def set_params(self, **kwargs):
        self.RF.set_params(**kwargs)
        return self
    
    def get_params(self, *args, **kwargs):
        params = self.RF.get_params(*args, **kwargs)
        params['estimators'] = self.estimators
        params['n_folds'] = self.n_folds
        return params
    
    def score(self, X, y):
        Xp, yp = self.pbb_space.transform(X, y)
        return self.RF.score(Xp, yp)
        
        
        
        
        
        
        
        