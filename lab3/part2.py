#!/usr/bin/python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from lab3 import helper
from common_utilities import common

PARTS = 4

if __name__ == '__main__':
    raw_data = np.loadtxt("../Lenses.txt", delimiter=",", dtype=np.int)
    data = np.array(raw_data[:, 1:], dtype=np.int)
    classifier = DecisionTreeClassifier()
    quantity = common.learn_and_count_quantity(classifier, data, PARTS)
    helper.create_graph_png(classifier, "part2.png")
    print(quantity)
    print(classifier.predict([[2, 1, 2, 1]]))
