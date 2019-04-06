#!/usr/bin/python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from common_utilities import common
from matplotlib import pyplot as plot

MIN_NEIGHBORS = 1
MAX_NEIGHBORS = 100
COLORS = ["green", "red"]

COLORS2 = ["orange", "black"]


def show_plot1(results_dict: dict):
    plot.plot(results_dict.keys(), results_dict.values())
    plot.ylabel("Percent of correct predictions")
    plot.xlabel("neighbours")
    plot.title("dependencies from K")
    plot.show()


def show_plots(learn_data, test_data, predictions, name="data"):
    plot.subplot(2, 1, 1)
    for element in learn_data:
        plot.scatter(element[0], element[1], c=COLORS[int(element[2])])
    plot.title(name)
    plot.xlabel("x1")
    plot.ylabel("x2")
    plot.subplot(2, 1, 2)
    for i in range(len(test_data)):
        if predictions[i] == test_data[i, -1]:
            t = 1
        else:
            t = 0
        plot.scatter(test_data[i, 0], test_data[i, 1], c=COLORS2[t])
    plot.xlabel("x1")
    plot.ylabel("x2")
    plot.show()


def transform_data(data):
    for element in data:
        element[-1:] = COLORS.index(element[-1:])
    return data


def find_best_k(results_dict):
    best_k = MIN_NEIGHBORS
    for key in results_dict.keys():
        if key > 15: break
        if results_dict[key] > results_dict[best_k]:
            best_k = key
    return best_k


if __name__ == '__main__':
    raw_data_learn = np.loadtxt("../svmdata4.txt", delimiter="\t", dtype=np.str)
    raw_data_test = np.loadtxt("../svmdata4test.txt", delimiter="\t", dtype=np.str)

    data_learn = np.array(transform_data(raw_data_learn[1:, 1:]), dtype=np.float)
    data_test = np.array(transform_data(raw_data_test[1:, 1:]), dtype=np.float)

    results = {}
    for i in range(MIN_NEIGHBORS, MAX_NEIGHBORS + 1):
        classifier = KNeighborsClassifier(n_neighbors=i, n_jobs=-1)
        common.learn(classifier, data_learn)
        results[i] = common.count_quantity(classifier, data_test)
    show_plot1(results)
    print("best k =", find_best_k(results))
    classifier = KNeighborsClassifier(n_neighbors=find_best_k(results), n_jobs=-1)
    common.learn(classifier, data_learn)
    predictions = classifier.predict(data_test[:, :-1])
    show_plots(data_learn, data_test, predictions)
