from feature_extractors import simple_features 
from nltk import NaiveBayesClassifier, ConfusionMatrix
import numpy as np
import itertools
import time
import csv

def test(vocabulary_builder_name, vocabulary_builder, arguments, corpus, feature_extractor=simple_features, csv_file_writer=None):
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
    
    # creating a CSV file where evaluations will be saved
    if csv_file_writer == None:
        file_name = "vacabulary_test_results/evaluation_" + vocabulary_builder_name
        file_name += time.strftime("_%d-%m-%Y_%H-%M.csv", time.gmtime())
        res_file = open(file_name, 'w+')
        res_writer = csv.writer(res_file)
    else:
        res_writer = csv_file_writer
    
    if csv_file_writer == None: # if this is a new file than write the header
        res_writer.writerow(["vocabulary_builder", "comment", "arguments", "vocabulary_length", "tr_set_size", "te_set_size", "standard_accuracy", "uniform_accuracy"])
    
    for comment, args in arguments:
        print("{0} vocabulary builder, arguments {1}, {2}".format(vocabulary_builder_name, str(args), comment))
        vocabulary = vocabulary_builder(corpus, **args)
        tr_set = [ (feature_extractor(vocabulary, q["words"]), q["category"]) for q in corpus.tr_set ]
        te_set = [ (feature_extractor(vocabulary, q["words"]), q["category"]) for q in corpus.te_set ]
        
        print(" --- vocabulary volume: ", len(vocabulary) )
        print(" --- training classifier...")
        
        classifier = NaiveBayesClassifier.train( tr_set )
        print(" --- ...classifier is trained.")
        
        # evaluating calssifiers
        print(" --- testing classifier: ")
        
        res = [classifier.classify(q) for q, _ in te_set]
        indeed = [c for _, c in te_set]
        cm = ConfusionMatrix(indeed, res)
        
        labelset_te_set = set(indeed) # in case not all categories accured in the test set
        
        st_accuracy = standard_accuracy(cm, labelset_te_set )
        un_accuracy = uniform_accuracy(cm, labelset_te_set)
        
        res_writer.writerow([vocabulary_builder_name,
                            comment,
                            str(args),
                            len(vocabulary),
                            len(tr_set),
                            len(te_set),
                            st_accuracy,
                            un_accuracy
                        ])
    
        labels_te_set = set(indeed)
        print(" --- standart accurcy:", standard_accuracy(cm, labels_te_set))
        print(" --- uniform accuracy:", uniform_accuracy(cm, labels_te_set))

        print("")

    if csv_file_writer == None:
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