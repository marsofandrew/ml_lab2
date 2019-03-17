#!/usr/bin/python

import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import common

MIN_PARTS = 2
MAX_PARTS = 100


def get_results(data, parts):
    divided_data = common.divide_into_parts(data, parts)
    results = []
    for i in range(parts):
        learn, test = common.get_learn_and_test_data(divided_data, [i])
        k_neighbour = KNeighborsClassifier(n_jobs=-1)
        x = learn[:, :-1]
        y = learn[:, -1:].T[0]
        k_neighbour.fit(x, y)
        predicted = k_neighbour.predict(test[:, :-1])
        results.append(common.count_quality_of_predictions(test[:, -1:], predicted))
    return sum(results) / len(results)


def get_learning_results(min_parts: int, max_parts: int, data):
    if min_parts < 1:
        raise ValueError("min parts should be greater or equal 1")
    if max_parts > len(data):
        raise ValueError("max parts must be less or equal data length")
    results = []
    for parts in range(min_parts, max_parts + 1):
        results.append({parts: get_results(data, parts)})
    return results


dataset = np.loadtxt("Tic_tac_toe.txt", delimiter=",", dtype=np.str)
dataset2 = np.loadtxt("spam.csv", delimiter=",", dtype=np.str)

transformed_data = common.get_transformed_data(dataset)
transformed_data2 = common.get_transformed_data(dataset2[1:, 1:])

print("Tic tac toe")
print(get_learning_results(MIN_PARTS, MAX_PARTS, transformed_data))
print("spam")
print(get_learning_results(MIN_PARTS, MAX_PARTS, transformed_data2))
