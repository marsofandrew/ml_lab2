#!/usr/bin/python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric
import common
from matplotlib import pyplot as plot


MIN_NEIGHBORS = 1
MAX_NEIGHBORS = 100


if __name__ == '__main__':
    raw_data_learn = np.loadtxt("svmdata4.txt", delimiter="\t", dtype=np.str)
    raw_data_test = np.loadtxt("svmdata4test.txt", delimiter="\t", dtype=np.str)
    data_learn, oe = common.get_transformed_data(raw_data_learn[1:, 1:])
    data_test, oe = common.get_transformed_data(raw_data_test[1:, 1:])

    results = {}
    for i in range(MIN_NEIGHBORS, MAX_NEIGHBORS + 1):
        classifier = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
        common.learn(classifier, data_learn)
        results[i] = common.count_quantity(classifier, data_test)
    print(results)