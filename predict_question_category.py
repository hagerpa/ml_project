import csv, pickle, re
from tempfile import TemporaryFile

def predict_question_category(qfile = 'question.csv'):
    """This method returns the results of the prediction form the ensemble classifier.
    The documents are obtained from the cv_file. It returns a list of dictionaries where
    each dictonary has the following keys:
        - major_category: predicted mayor-category ids
        - minor_category: predicted minor-category ids
        - confidence_major_cat: convidence value of prediction
        - confidence_minor_cat: --//--
    """
    with open("corpus_main.pkl", "rb") as file:
        corpus_main = pickle.load(file)
        cat_main = corpus_main.cats
    with open("corpus_sub.pkl", "rb") as file:
        corpus_sub = pickle.load(file)
        cat_sub = corpus_sub.cats
    
    with open("final_method_main.pkl", "rb") as file:
        final_method_main = pickle.load(file);
    with open("final_method_sub.pkl", "rb") as file:
        final_method_sub = pickle.load(file);
    
    ### Reading questions ###
    with open(qfile, 'r') as file_:
        file_content = file_.read()    
        regexps = [(r"\\\"", "'")]
        for find, replace in regexps:
            file_content, n = re.subn(find, replace, file_content)
    
    test_documents = []
    with TemporaryFile("w+") as file_:
        file_.write(file_content)
        file_.seek(0)
        
        qreader = csv.reader(file_)
        next(qreader)
        
        for row in qreader:
            test_documents += [ row[4].lower() ]
    
    X_main = corpus_main.process_example(test_documents)
    X_sub = corpus_sub.process_example(test_documents)
    
    y_main = final_method_main.predict(X_main)
    y_sub = final_method_main.predict(X_sub)
    
    PA_main = final_method_main.predict_proba(X_main)
    PA_sub = final_method_main.predict_proba(X_sub)
    
    p_main = [PA_main[i, int(y_main[i]) ] for i in range(len(y_main))]
    p_sub = [PA_sub[i, int(y_sub[i]) ] for i in range(len(y_sub))]
    
    df = [{
        'major_category': cat_main[ int(y_main[i]) ],
        'minor_category': cat_sub[ int(y_sub[i]) ],
        'confidence_major_cat': p_main[i],
        'confidence_minor_cat': p_sub[i]
    } for i in range(len(y_main))]
    
    return df

if __name__ == "__main__":
    predict_question_category()