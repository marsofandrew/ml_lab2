#!/usr/bin/python

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from common_utilities import common
from matplotlib import pyplot as plot

MIN_PARTS = 2
MAX_PARTS = 50

RESULTS = ['"nonspam"', '"spam"']


def get_results(data, parts):
    return common.learn_and_count_quantity(KNeighborsClassifier(n_jobs=-1), data, parts)


def get_learning_results(min_parts: int, max_parts: int, data):
    if min_parts < 1:
        raise ValueError("min parts should be greater or equal 1")
    if max_parts > len(data):
        raise ValueError("max parts must be less or equal data length")
    results = {}
    for parts in range(min_parts, max_parts + 1):
        results[parts] = get_results(data, parts)
    return results


def transform_data(data):
    for element in data:
        element[-1:] = RESULTS.index(element[-1:])
    return data


def show_plot1(results_dict: dict, name):
    plot.plot(results_dict.keys(), results_dict.values())
    plot.ylabel("Percent of correct predictions")
    plot.xlabel("parts")
    plot.title("dependencies from parts " + name)
    plot.show()


dataset = np.loadtxt("../Tic_tac_toe.txt", delimiter=",", dtype=np.str)
dataset2 = np.loadtxt("../spam.csv", delimiter=",", dtype=np.str)

transformed_data, oe = common.get_transformed_data(dataset)
transformed_data2 = np.array(transform_data(dataset2[1:, 1:]), dtype=np.float)

print("Tic tac toe")
result1 = get_learning_results(MIN_PARTS, MAX_PARTS, transformed_data)
print("spam")
result2 = get_learning_results(MIN_PARTS, MAX_PARTS, transformed_data2)
show_plot1(result1, "tic tac toe")
show_plot1(result2, "spam")
