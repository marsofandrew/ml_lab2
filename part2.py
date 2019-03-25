#!/usr/bin/python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric
import common

from matplotlib import pyplot as plot

PARTS = 10
MIN_NEIGHBORS = 1
MAX_NEIGHBORS = 20
METRIC = 'metric'
METRIC_PARAMS = 'metric_params'


def get_results(data, n_neighbors):
    return common.learn_and_count_quantity(KNeighborsClassifier(n_neighbors, n_jobs=-1), data, PARTS)


def show_plot1(results_dict: dict):
    plot.plot(results_dict.keys(), results_dict.values())
    plot.ylabel("Percent of correct predictions")
    plot.xlabel("neighbours")
    plot.title("dependencies from K")
    plot.show()


def show_plot2(result: dict):
    pass


def find_best_k(results_dict):
    best_k = MIN_NEIGHBORS
    for key in results_dict.keys():
        if results_dict[key] > results_dict[best_k]:
            best_k = key
    return best_k


def create_classifiers(min_neighbours: int, max_neighbours: int, classifiers_params: list):
    classifiers_ = []
    for params in classifiers_params:
        for neighbours in range(min_neighbours, max_neighbours + 1):
            classifiers_.append(KNeighborsClassifier(n_neighbors=neighbours,
                                                     n_jobs=-1,
                                                     metric=params.get(METRIC),
                                                     metric_params=params.get(METRIC_PARAMS, None)))
    return classifiers_


if __name__ == '__main__':
    raw_data = np.loadtxt("glass.csv", delimiter=",", dtype=np.str)
    data = np.array(raw_data[1:, 2:], dtype=np.float)

    results = {}
    for k in range(MIN_NEIGHBORS, 100 + 1):
        results[k] = get_results(data, k)

    show_plot1(results)

    classifiers = [
        KNeighborsClassifier(n_neighbors=3, n_jobs=-1, metric='euclidean'),
        KNeighborsClassifier(n_neighbors=3, n_jobs=-1, metric="manhattan"),
        KNeighborsClassifier(n_neighbors=3, n_jobs=-1, metric="chebyshev"),
        KNeighborsClassifier(n_neighbors=3, n_jobs=-1, metric="minkowski"),
        KNeighborsClassifier(n_neighbors=3, n_jobs=-1, metric="minkowski", p=4),
        KNeighborsClassifier(n_neighbors=3, n_jobs=-1, metric="minkowski", p=10),
        KNeighborsClassifier(n_neighbors=3, n_jobs=-1, metric="minkowski", p=8),
        KNeighborsClassifier(n_neighbors=3, n_jobs=-1, metric="minkowski", p=1)
    ]
    results2 = {}
    for classifier_item in classifiers:
        results2[classifier_item] = common.learn_and_count_quantity(classifier_item, data, PARTS)
    print(results2)

    best_k = find_best_k(results)
    print("best prediction:", "k =", best_k, "percent =", results[best_k] * 100)
    classifier = KNeighborsClassifier(best_k, n_jobs=-1)
    common.learn_and_count_quantity(classifier, data, PARTS)
    predicted = classifier.predict([[1.516, 11.7, 1.01, 1.19, 72.59, 0.43, 11.44, 0.02, 0.1]])
    print(predicted)
