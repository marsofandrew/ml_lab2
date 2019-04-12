#!/usr/bin/python
import numpy as np
from sklearn import svm
from common_utilities import common
from matplotlib import pyplot as plot

SUBSTITUTES = ['red', 'green']


def show_plots(learn_data, test_data, classifier):
    vectors = classifier.support_vectors_
    amount_of_vectors = classifier.n_support_
    index = 0
    support_vectors_points = []
    for support_vectors in amount_of_vectors:
        support_vectors_points.append([])
        for j in range(support_vectors):
            support_vectors_points[-1].append(vectors[index])
            index += 1

    plot.subplot(2, 1, 1)
    plot.title("learn data")
    plot.xlabel("x1")
    plot.ylabel('x2')
    for elem in learn_data:
        plot.scatter(elem[0], elem[1], c=SUBSTITUTES[int(elem[-1])])
    plot.subplot(2, 1, 2)
    plot.title("test data")
    plot.xlabel("x1")
    plot.ylabel('x2')
    for elem in test_data:
        plot.scatter(elem[0], elem[1], c=SUBSTITUTES[int(elem[-1])])
    plot.show()


def get_data_from_file(filename):
    raw_data = np.loadtxt(filename, delimiter='\t', dtype=np.str)
    return np.array(common.replace_text_data(raw_data[1:, 1:], SUBSTITUTES), dtype=np.float)


def main():
    learn_data = get_data_from_file("resources/svmdata1.txt")
    test_data = get_data_from_file("resources/svmdata1test.txt")
    classifier = svm.SVC(kernel='linear')
    common.learn(classifier, learn_data)
    quantity_learn = common.count_quantity(classifier, learn_data)
    quantity_test = common.count_quantity(classifier, test_data)
    print("Quantity on:\nlearn_data: {}\ntest_data: {}".format(quantity_learn, quantity_test))
    print("support vectors:", sum(classifier.n_support_))
    show_plots(learn_data, test_data, classifier)


if __name__ == '__main__':
    main()
