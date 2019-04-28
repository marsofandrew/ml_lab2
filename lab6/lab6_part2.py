from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import svm
from common_utilities import common

svc = svm.SVC(probability=True, kernel='linear')
gauss = GaussianNB()
neighbour = KNeighborsClassifier()
tree = DecisionTreeClassifier()
PARTS = 4
N_ESTIMATORS = 200
BASE_ESTIMATOR = [gauss, tree, svc, neighbour]
TITLE = ["GaussianNB", "KNeighbours", "DecisionTree"]


def step0():
    results = {}
    for i in range(1, N_ESTIMATORS + 1):
        results[i] = common.learn_and_count_quantity(BaggingClassifier(n_estimators=i), data, PARTS)
    common.show_plot_from_dict(results, "Quantity of BaggingClassifier", "correct predictions", "n_estimators")
    best_key = common.find_key_of_max_value(results)
    print("best: {}; score: {}".format(best_key, results[best_key]))


def step1():
    results2 = {}
    k = 0
    for base_est in BASE_ESTIMATOR:
        for i in range(1, N_ESTIMATORS + 1):
            results2[i] = common.learn_and_count_quantity(BaggingClassifier(n_estimators=i, base_estimator=base_est),
                                                          data, PARTS)
        common.show_plot_from_dict(results2, TITLE[k], "correct predictions", "n_estimators")
        best_key = common.find_key_of_max_value(results2)
        print("best: {}; score: {}".format(best_key, results2[best_key]))
        k += 1
        results2 = {}


def step2():
    results3 = {}
    for i in range(1, N_ESTIMATORS + 1):
        results3[i] = common.learn_and_count_quantity(RandomForestClassifier(n_estimators=i), data, PARTS)
    common.show_plot_from_dict(results3, "Quantity of RandomForestClassifier", "correct predictions", "n_estimators")
    best_key = common.find_key_of_max_value(results3)
    print("best: {}; score: {}".format(best_key, results3[best_key]))


if __name__ == '__main__':
    raw_data = np.loadtxt("../glass.csv", delimiter=",", dtype=np.str)
    data = np.array(raw_data[1:, 2:], dtype=np.float)
    #common.execute(step0)
    #common.execute(step1)
    common.execute(step2)
