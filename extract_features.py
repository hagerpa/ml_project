import corpus, filters
import categories as categorie_class
from vocabulary_builders import ig_based_non_uniform
import pickle
from feature_extractors import simple_features
import numpy as np

def extract_features(qfile="question_train.csv", qcatfile="question_category_train.csv", catfile="category.csv", subcats=False, outfile="features.npz"):
    # loading the categories
    cats = categorie_class.categories()
    # initalizing corpus
    corp = corpus.corpus(cats)
    # loading questions into corpus
    corp.load(qfile, qcatfile)
    # running filers on the raw questions
    sentence_filters = [filters.punctuation_filter]
    word_filters = [filters.small_word_filter, filters.stopword_filter, filters.stemming_filter]
    corp.process(sentence_filters, word_filters)
    # saving corpus into pickle
    # pickle.dump(corp, "corpus.pkl")
    # selecting the term-space
    term_space = ig_based_non_uniform(corp, M=2500, read_from_file=True)
    d = len(term_space)
    # create mapping form features names to new ids and inverse
    term_to_feature = {}
    feature_to_term = {}
    for term, i in zip(term_space,range(d)):
        term_to_feature[term] = i
        feature_to_term[i] = term
    # creating features and lable arrays
    n = len(corp.tr_set)
    features = np.zeros((d, n))
    categoryids = np.zeros(n)
    
    # we define new ids vor the parent categories, which will be coherent with ones assigned in categoryids 
    number_of_cats = len(corp.cats.all_names())
    new_category_ids = {c: i for c, i in zip(corp.cats.all_names(), range(number_of_cats))}
   
    for q, j in zip(corp.tr_set, range(n)):
        fe = simple_features(term_space, q["words"])
        for term, value in zip(fe.keys(), fe.values()):
            i = term_to_feature[term]
            features[i, j] = value
            categoryids[j] = new_category_ids[ q["category"] ]
    
    categories = {i: c for c, i in zip(new_category_ids.keys(), new_category_ids.values())}
    
    featurenames = [feature_to_term[i] for i in range(d)]
    
    np.savez(outfile, features=features, featurenames=featurenames, categoryids=categoryids, categories=categories)
    
if __name__ == "__main__":
    extract_features()
