#!/usr/bin/python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric
import common

from matplotlib import pyplot as plot

PARTS = 10
MIN_NEIGHBOURS = 1
MAX_NEIGHBOURS = 100


def get_results(data, n_neighbours):
    return common.learn(KNeighborsClassifier(n_neighbours, n_jobs=-1), data, PARTS)


def show_plot1(results_dict: dict):
    plot.plot(results_dict.keys(), results_dict.values())
    plot.ylabel("Percent of correct predictions")
    plot.xlabel("neighbours")
    plot.title("dependencies from K")
    plot.show()


def find_best_k(results_dict):
    best_k = MIN_NEIGHBOURS
    for key in results_dict.keys():
        if results_dict[key] > results_dict[best_k]:
            best_k = key
    return best_k


raw_data = np.loadtxt("glass.csv", delimiter=",", dtype=np.str)
data = np.array(raw_data[1:, 2:], dtype=np.float)


def create_classifiers(min_neighbours: int, max_neighbours: int, classifiers_params: list):
    classifiers_ = []
    for params in classifiers_params:
        for neighbours in range(min_neighbours, max_neighbours + 1):
            classifiers_.append(KNeighborsClassifier(n_neighbors=neighbours,
                                                     n_jobs=-1,
                                                     metric=params.get('metric'),
                                                     metric_params=params.get('metric_params', None)))
    return classifiers_


results = {}
for k in range(MIN_NEIGHBOURS, MAX_NEIGHBOURS + 1):
    results[k] = get_results(data, k)

show_plot1(results)
best_k = find_best_k(results)
print("best prediction:", "k =", best_k, "percent =", results[best_k] * 100)
classifier = KNeighborsClassifier(best_k, n_jobs=-1)
common.learn(classifier, data, PARTS)
predicted = classifier.predict([[1.516, 11.7, 1.01, 1.19, 72.59, 0.43, 11.44, 0.02, 0.1]])
print(predicted)
