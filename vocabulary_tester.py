from feature_extractors import simple_features 
from nltk import NaiveBayesClassifier, ConfusionMatrix
import time
import csv

def test(vocabularies, corpus, feature_extractor=simple_features, tr_set_size=8000, te_set_size=2000, csv_file_write=None):
    """This is a simple method to evaluate the perfomance of diffrent kinds of
    vocabulary on a "NavieBayesClassifier" (from the nltk package). We understand
    the "vocabulary" as a set of words from wich we can build features.
    In the default case we use the "simple_features" extractor wich simply checks
    for a training example whether a word from the vocabulary is contained or not.
    The results will be written in a CSV-files at ./vocabulary_test_results/, the
    name of the files contains the current datum and time.
    
    Key Inputs:
    - vacublaries: a dictionary which as keys has the name of the vocabulary and as
                    values sets of strings.
    - corpus: a list of tuples of the form (words, category), where words must be a
                list of words. Pay attantion that thes words are processed such that
                the fit with the vocabulary.
    - csv_file_writer: if you arleady open a CSV-file you can pass the CSV-write to 
        this method and it will use your writer.
    """

    classifiers = {}
    results = []
    
    if csv_file_writer == None:
        file_name = time.strftime("vacabulary_test_results/evaluation_%d-%m-%Y_%H-%M.csv", time.gmtime())
        res_file = open(file_name, 'w+')
        res_writer = csv.writer(res_file)
    else:
        res_writer = csv_file_write
    
    res_writer.writerow(["vocabulary_type", "vocabulary_length", "tr_set_size", "te_set_size", "standard_accuracy", "uniform_accuracy"])
    
    # training calssifiers
    for vocab_key in vocabularies.keys():
        print("training '{0}' classifier...".format(vocab_key))
        vocab = vocabularies[vocab_key]
    
        feature_set = [ (feature_extractor(vocab, q), cat) for q, cat in corpus ]
        te_max = max(tr_set_size + te_set_size, len(feature_set))
        train_set, test_set = feature_set[:tr_set_size], feature_set[tr_set_size:tr_set_size + te_set_size]
    
        classifiers[vocab_key] = NaiveBayesClassifier.train(train_set)
        print(" --- classifier is trained.")
    
    print("")
    
    # evaluating calssifiers
    for cf in classifiers.keys():
        print("testing '{0}' classifier: ".format(cf))
        print(" --- vocabulary volume: ", len(vocabularies[cf]))
    
        res = [classifiers[cf].classify(q) for q, _ in test_set]
        indeed = [c for _, c in test_set]
        cm = ConfusionMatrix(indeed, res)
        
        res_writer.writerow([cf, len(vocabularies[cf]), tr_set_size, te_set_size, standard_accuracy(cm, set(indeed) ), uniform_accuracy(cm, set(indeed) )])
        
        labels_te_set = set(indeed)
        print(" --- standart accurcy:", standard_accuracy(cm, labels_te_set))
        print(" --- uniform accuracy:", uniform_accuracy(cm, labels_te_set))
    
        print("")
    
    if csv_file_write == None:
        res_file.close()
    
    print("\n test are all finshed and saved into file!")
    #print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=12))

def print_accuracy_list(cm, categories):
    for cat in categories:
        total_examples = sum([cm.__getitem__((cat, cat_)) for cat_ in categories])
        accuracy = cm.__getitem__((cat, cat)) / total_examples
        print("accuracy within category {0:23}: {1:2.2}".format(cat,accuracy))

def uniform_accuracy(cm, categories):
    uni_accuracy = 0
    for cat in categories:
        total_examples = sum([cm.__getitem__((cat, cat_)) for cat_ in categories])
        accuracy = cm.__getitem__((cat, cat)) / total_examples
        uni_accuracy += accuracy
    return uni_accuracy / len(categories)

def standard_accuracy(cm, categories):
    right_examples = 0
    all_examples = 0
    for cat in categories:
        all_examples += sum([cm.__getitem__((cat, cat_)) for cat_ in categories])
        right_examples += cm.__getitem__((cat, cat))
    return right_examples/all_examples