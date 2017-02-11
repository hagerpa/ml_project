from scipy.sparse import find

class TrainingSupport:
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