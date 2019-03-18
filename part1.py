#!/usr/bin/python

import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import common

MIN_PARTS = 2
MAX_PARTS = 25


def get_results(data, parts):
    return common.learn(KNeighborsClassifier(n_jobs=-1), data, parts)


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
