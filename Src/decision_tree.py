from collections import Counter
from sklearn import metrics
import argparse
from decimal import *
import decimal
import sys
import copy
from random import shuffle
from math import log
import numpy as np
import operator
import csv


CATEGORIES = ['math', 'cs', 'physics', 'stat']

class Node:
    def __init__(self, abst, leaf=False, root=False, yes=None, no=None, feature=None):
        #list of abstracts at the node
        self.abstracts = abst
        #is node a leaf
        self.leaf = leaf
        #is node root
        self.root = root
        #yes node
        self.yes = yes
        #no node
        self.no = no
        #feature for testing at node
        self.feature = feature
        pass
class Abstract:
    def __init__(self, ID, cat, text):
        self.ID = ID
        self.category = cat
        self.text = text
        self.counts = Counter([w for w in text.split()])

def load_abstracts(abstracts, categories):
    """
        return: list of Abstract objects
        input: path to .csv file containing abstracts and .csv file containing categories
    """
    #load categories
    abstract_list = []
    with open(categories, "r") as cat, open(abstracts, "r") as abst:
        for d in enumerate(zip(cat, abst)):
            if d[0] == 0:
                continue

            cat_data = d[1][0].split(",")
            abs_data = d[1][1].split(",")
            ID = int(cat_data[0].strip())
            category = cat_data[-1].strip()
            text = abs_data[-1]

            if category != "category":
                abstract_list.append(Abstract(ID, category, text))

    return abstract_list

def feature_extract(abstracts, chunk=None):
    """
        return: list of words to use as features for decision nodes.
    """
    wordset = {}
    for a in abstracts:
        #update wordset with abstract's word counts
        for w in a.counts:
            word_counts = wordset.setdefault(w, {'math':0, 'cs':0, 'stat':0, 'physics':0})
            word_counts[a.category] = word_counts[a.category] + a.counts[w]

    top_words = sorted(wordset, reverse=True, key=lambda word: \
            np.sum([wordset[word][cat] for cat in ['physics', 'cs', 'math', 'stat']]))

    if chunk:
        return [w for w in wordset.keys() if w in top_words[:chunk]]
    else:
        return [w for w in wordset.keys()]

def entropy(abstracts):
    """
        return: entropy of given set of abstracts
    """
    h_total = 0.0
    node_categories = Counter([a.category for a in abstracts])
    for c in node_categories:
        #compute probability of a category c
        p_C = node_categories[c] / float(len(abstracts))
        #compute entropy of category c
        h_C = -1 * p_C * log(p_C, 2)
        h_total = h_total + h_C

    return h_total

def information_gain(partitions, D):
    """
        return: information gain of given partition against unpartitioned D
    """
    #calculate entropy without partitioning H(D)
    parent_entropy = entropy(D)
    #for each branching of D, compute the conditional entropy
    partition_entropy = 0
    #return information gain: H(D) - H(D|x
    return round(parent_entropy, 5) - round(partition_entropy, 5)

def partition_data(splitword, abstracts):
    """
        Split the data based on a splitword or feature
    """
    yes_list = []
    no_list = []

    for a in abstracts:
        if splitword in a.counts:
            yes_list.append(a)
        else:
            no_list.append(a)

    return (yes_list, no_list)

def best_partition(node, chunk=200):
    """"
        return: optimal partition of data over all features
    """
    features = feature_extract(node.abstracts, chunk=chunk)
    if len(features) < 1:
        print("FEATS")
        sys.exit()
    #try to partition the data on each feature
    best_partition = None
    best_info_gain = 0
    best_feature = None
    partitions = []
    for f in features:
        yes_partition, no_partition = partition_data(f, node.abstracts)
        info_gain = information_gain([yes_partition, no_partition], node.abstracts)
        partitions.append((len(yes_partition), len(no_partition), info_gain,\
            [a.category for a in node.abstracts], f, [a.ID for a in\
                node.abstracts]))
        if info_gain < 0:
            print("WHOA")
            print(info_gain)
            sys.exit()
        if info_gain >= best_info_gain and len(yes_partition) > 0 and\
        len(no_partition) > 0:
            best_partition = (yes_partition, no_partition)
            best_info_gain = info_gain
            best_feature = f

    if not best_feature:
        return -1

    else:
        return (best_partition, best_feature)

def check_stop(node, stop=0):
    """
        Checks if node should be a leaf
    """
    node_entropy = entropy(node.abstracts)
    if np.isclose(node_entropy, 0.0): 
        return True
    elif node_entropy < stop:
        return True
    else:
        return False

def node_category(node):
    """
        Returns the consensus class of the node
    """
    category_counts = Counter([a.category for a in node.abstracts])
    return max(category_counts, key=category_counts.get)

def print_tree(node, indent=''):
    """
        print tree
    """
    if node.leaf:
        print(indent + '\t' + node.leaf.upper())
        return
    else:
        print(indent + node.feature + "?")

        print(indent + "yes? --> ")
        print_tree(node.yes, indent=indent+'\t')
        print(indent + "no? --> ")
        print_tree(node.no, indent=indent+'\t')
    return
def grow_tree(node, chunk=200, stop=0):
    if check_stop(node, stop=stop):
        #set node as leaf and end recursion
        node.leaf = node_category(node)
        return
    else:
        #get partitions for node's abstracts and the test, add node to tree
        parti = best_partition(node, chunk=chunk)
        if parti == -1:
            node.leaf = node_category(node)
            return
        yes_partition, no_partition = parti[0]
        feature = parti[1]
        node.feature = feature
        #set yes and no branches to new nodes based on best partition
        node.yes = Node(yes_partition)
        node.no = Node(no_partition)

        #tree.append(node)

        grow_tree(node.yes, chunk=chunk, stop=stop)
        grow_tree(node.no, chunk=chunk, stop=stop)
    return node

def classify(abstract, node):
    """
        return: category of abstrcact on given decision tree
    """
    current_node = node
    while True:
        if current_node.leaf:
            return current_node.leaf
        else:
            if current_node.feature in abstract.counts.keys():
                current_node = current_node.yes
            else:
                current_node = current_node.no


def abstract_clean(abstracts):
    unique_abstracts = []
    for i, a in enumerate(abstracts):
        print("%s of %s" % (i, len(abstracts)))
        if a.counts not in unique_abstracts:
            unique_abstracts.append(a)
        else:
            print("DUPLICATE!")
            continue
    return unique_abstracts
def count_nodes(node):
    if node.leaf:
        return 1
    else:
        return count_nodes(node.yes) + count_nodes(node.no)

def validation(validation_set, node):
    """
        return: accuracy on validation set
    """
    correct = 0
    for abstract in validation_set:
        prediction = classify(abstract, node)
        if prediction == abstract.category:
            correct = correct + 1

    return correct / float(len(validation_set))

def k_fold(k, abstracts):
    """
        return: average accuracy on k fold cross validation. shuffles input
    before validation.
    """
    #shuffle a deep copy of abstracts
    shuffled = copy.deepcopy(abstracts)
    shuffle(shuffled)

    slice_size = len(abstracts) / k
    accuracies = []

    for n in np.arange(0, len(abstracts), slice_size):
        print(n)
        training_set = abstracts[n: n + slice_size]
        validation_set = abstracts[:n] + abstracts[n+slice_size:]


        root = Node(training_set, root=True)
        tree = grow_tree(root)

        accuracy = validation(validation_set, tree) 
        accuracies.append(accuracy)

    return np.mean(accuracies)



def prune(node, validation_set):
    if node.leaf:
        return
    else:
        tree_accuracy = validation(validation_set, node)
        maj_cat = node_category(node)
        node_accuracy = len([a for a in validation_set if maj_cat ==\
            a.category]) / float(len(validation_set))
        if node_accuracy > tree_accuracy:
            node.leaf = maj_cat
        else:
            prune(node.yes, validation_set)
            prune(node.no, validation_set)
        return node

def parameters(abstracts, size):
    parameter_dict = {}
    with open("decision_parameters.txt", "w+") as para:
        for s in [0, 0.5, 1, 1.5]:
            print(s)
            for c in np.arange(100, 500, 100):
                print(c)

                train_set = abstracts[:size]
                validation_set = abstracts[size:]
                root = Node(train_set)
                tree = grow_tree(root, stop=s, chunk=c)

                accuracy = validation(validation_set, tree)
                print(accuracy)

                parameter_dict.setdefault(s, {}).setdefault(c, accuracy)
                para.write("%s %s %s" % (s, c, accuracy))

        return parameter_dict

def get_metrics(validation_set, node):
    predictions = []
    trues = []
    for a in validation_set:
        predictions.append(classify(a, node))
        trues.append(a.category)

    with open("DTmetrics.txt", "w+") as m:
        metrix = metrics.classification_report(trues, predictions, \
                labels=CATEGORIES)
        m.write(metrix)

    return 1

def cline():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in", type=str, help="file containing training\
    examples")
    parser.add_argument("-o", "--out", type=str, help="file containing categories")
    parser.add_argument("-f", "--features", type=int, help="number of features to use")
    parser.add_argument("-e", "--entropy", type=float, help="maximal entropy at node")
    parser.add_argument("-s", "--size", type=int, help="size of training set")
    return parser.parse_args()
if __name__ == "__main__":
    abstracts_path = "../Data/train_in_nouns_unique.csv"
    categories_path = "../Data/train_out_unique.csv"
    #get command line arguments
    args = cline()


    print("loading abstracts")
    abstracts = load_abstracts(abstracts_path, categories_path)
    print("abstracts loaded")

    shuffle(abstracts)
    train_set = abstracts[:args.size]
    validation_set = abstracts[args.size:]

    root = Node(train_set)
    tree = grow_tree(root, chunk=args.features, stop=args.entropy)
    get_metrics(validation_set, tree)


