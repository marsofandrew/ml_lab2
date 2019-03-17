#!/usr/bin/python
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import common

PARTS = 10
MAX_NEIGHBOURS = 100


def get_results(data, n_neighbours):
    divided_data = common.divide_into_parts(data, PARTS)
    results = []
    for i in range(PARTS):
        learn, test = common.get_learn_and_test_data(divided_data, [i])
        k_neighbour = KNeighborsClassifier(n_neighbours, n_jobs=-1)
        x = learn[:, :-1]
        y = learn[:, -1:].T[0]
        k_neighbour.fit(x, y)
        predicted = k_neighbour.predict(test[:, :-1])
        results.append(common.count_quality_of_predictions(test[:, -1:], predicted))
    return sum(results) / len(results)


raw_data = np.loadtxt("glass.csv", delimiter=",", dtype=np.str)
data = raw_data[1:, 2:]
data = common.get_transformed_data(data[1:, 2:])
for k in range(1, MAX_NEIGHBOURS + 1):
    print({k: get_results(data, k)})
