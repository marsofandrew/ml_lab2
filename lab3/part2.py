#!/usr/bin/python
import os
os.environ["PATH"] += 'C:\Program Files (x86)\Graphviz2.38\bin\dot.exe'
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from common_utilities import common
import graphviz


PARTS = 4

if __name__ == '__main__':
    raw_data = np.loadtxt("../Lenses.txt", delimiter=",", dtype=np.int)
    data = np.array(raw_data[:, 1:], dtype=np.int)
    classifier = DecisionTreeClassifier()
    quantity = common.learn_and_count_quantity(classifier, data, PARTS)
    dot_data = tree.export_graphviz(classifier)
    graph = graphviz.Source(dot_data)
    graph.save("graph", "..")
    #graph.render()
    print(quantity)
    print(classifier.predict([[2, 1, 2, 1]]))
