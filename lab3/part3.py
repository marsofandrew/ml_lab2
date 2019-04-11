#!/usr/bin/python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from common_utilities import common

SUBSTITUTES = ['"y"', '"n"']
MAX_DEPTH = 50
CRITERION = ['gini', 'entropy']
MAX_FEATURES = ['auto', 'sqrt', 'log2', None]
SPLIT_TYPES = ['best', 'random']
PARTS = 10


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
    return common.find_key_of_max_value(results2)


if __name__ == '__main__':
    raw_data = np.loadtxt("../spam7.csv", delimiter=",", dtype=np.str)
    data = common.replace_text_data(raw_data[1:, :], SUBSTITUTES)
    data = np.array(data, dtype=np.float)
    best = step2()
    print("best = ", best)
    classifier = DecisionTreeClassifier(max_depth=best[0], max_features=best[1], splitter=best[2], criterion=best[3])
