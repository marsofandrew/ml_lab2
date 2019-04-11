#!/usr/bin/python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from common_utilities import common

PARTS = 4

if __name__ == '__main__':
    raw_data = np.loadtxt("../Lenses.txt", delimiter=",", dtype=np.int)
    data = np.array(raw_data[:, 1:], dtype=np.int)

    # TODO: show classifier
    classifier = DecisionTreeClassifier()
    quantity = common.learn_and_count_quantity(classifier, data, PARTS)
    print(quantity)
    print(classifier.predict([[2, 1, 2, 1]]))
