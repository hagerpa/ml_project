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
        res_writer.writerow(["vocabulary_builder",
                            "comment",
                            "arguments",
                            "term-space-dimension",
                            "tr_set_size", 
                            "te_set_size",
                            "microavaraged_recall",
                            "macroavaraged_recall",
                            "macroavarged_precission"])
    
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
        
        mic_recall = microavaraged_recall(cm, labelset_te_set )
        mac_recall = macroavaraged_recall(cm, labelset_te_set)
        mac_prec = macroavaraged_precision(cm, labelset_te_set)
        
        res_writer.writerow([vocabulary_builder_name,
                            comment,
                            str(args),
                            len(vocabulary),
                            len(tr_set),
                            len(te_set),
                            mic_recall,
                            mac_recall,
                            mac_prec
                        ])
        
        print(" --- microavaraged recall:", mic_recall)
        print(" --- macroavaraged recall:", mac_recall)
        print(" --- macroavaraged precision:", mac_prec)
        print("")

    if csv_file_writer == None:
        res_file.close()
    
    print("Tests are all finshed and saved into file!")
    #print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=12))

def macroavaraged_recall(cm, categories):
    """This method returns the macroavarged recall."""
    mac_recall = 0
    for cat in categories:
        # counting all examples that where provided for this category in the test-set
        total_examples = sum([cm.__getitem__((cat, cat_)) for cat_ in categories])
        # couting True-Positives and divide by the number of all examples in this category
        accuracy = cm.__getitem__((cat, cat)) / total_examples
        # adding up the "local"-recalls
        mac_recall += accuracy
    return mac_recall / len(categories)
    
def macroavaraged_precision(cm, categories):
        """This method returns the macroavarged precission."""
        if len(categories) == 0: return 0
        mac_prec = 0
        for cat in categories:
            # counting all the examples that where assigned to this category (TP + FP)
            total_examples = sum([cm.__getitem__((cat_, cat)) for cat_ in categories])
            if total_examples == 0: continue
            # couting True-Positives and divide by the number of all examples in this category
            accuracy = cm.__getitem__((cat, cat)) / total_examples
            # adding up the "local"-recalls
            mac_prec += accuracy
        return mac_prec / len(categories)

def microavaraged_recall(cm, categories):
    """This method returns the microavarage recall."""
    true_positives = 0
    all_examples = 0
    for cat in categories:
        all_examples += sum([cm.__getitem__((cat, cat_)) for cat_ in categories])
        true_positives += cm.__getitem__((cat, cat))
    return true_positives/all_examples
    
def print_accuracy_list(cm, categories):
    for cat in categories:
        total_examples = sum([cm.__getitem__((cat, cat_)) for cat_ in categories])
        accuracy = cm.__getitem__((cat, cat)) / total_examples
        print("accuracy within category {0:23}: {1:2.2}".format(cat,accuracy))