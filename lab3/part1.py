#!/usr/bin/python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from common_utilities import common

MAX_DEPTH = 200
CRITERION = ['gini', 'entropy']
MAX_FEATURES = ['auto', 'sqrt', 'log2', None]
SPLIT_TYPES = ['best', 'random']
PARTS = 10


def step1():
    results = {}
    for i in range(1, MAX_DEPTH + 1):
        results[i] = common.learn_and_count_quantity(DecisionTreeClassifier(max_depth=i), data, PARTS)
    common.show_plot_from_dict(results, "quantity(max_depth)", "correct predictions", "max_depth")
    print(common.find_key_of_max_value(results))


def step2():
    results2 = {}
    for depth in range(1, MAX_DEPTH + 1):
        for features in MAX_FEATURES:
            for split_type in SPLIT_TYPES:
                for criterion in CRITERION:
                    classifier = DecisionTreeClassifier(criterion=criterion, splitter=split_type, max_features=features,
                                                        max_depth=depth)
                    results2[(depth, features, split_type, criterion)] = common.learn_and_count_quantity(classifier,
                                                                                                         data, PARTS)

    print("max_depth", "max_features", "splitter", "criterion", "quantity")
    for element in results2.keys():
        print(element, results2.get(element))
    print("best", common.find_key_of_max_value(results2))


if __name__ == '__main__':
    raw_data = np.loadtxt("../glass.csv", delimiter=",", dtype=np.str)
    data = np.array(raw_data[1:, 2:], dtype=np.float)
    step1()
    step2()
