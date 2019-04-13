#!/usr/bin/python
from sklearn import svm
from lab4.helpers import *


def find_best_c(learn_data, test_data, expected_quantity=1.0, min_parameter=0.01, max_parameter=None, step=0.01):
    """
    :param learn_data: data for learning classifier
    :type numpy.ndarray
    :param test_data: data for testing classifier
    :type numpy.ndarray
    :param expected_quantity: expected quantity of classifier
    :param min_parameter: minimal parameter for search
    :type float
    :param max_parameter: maximal parameter for search
    :type float
    :param step: step for finding parameter c
    :return: parameter c and classifier
    """

    def should_continue(current_quantity, current_c):
        if max_parameter is None:
            return current_quantity < expected_quantity
        else:
            return (current_quantity < expected_quantity) and (current_c < max_parameter)

    c = min_parameter
    classifier = svm.SVC(kernel='linear', C=c)
    common.learn(classifier, learn_data)
    quantity = common.count_quantity(classifier, test_data)
    while should_continue(quantity, c):
        c += step
        classifier = svm.SVC(kernel='linear', C=c)
        common.learn(classifier, learn_data)
        quantity = common.count_quantity(classifier, test_data)
    return c, classifier


def part1():
    learn_data = get_data_from_file("resources/svmdata1.txt")
    test_data = get_data_from_file("resources/svmdata1test.txt")
    classifier = svm.SVC(kernel='linear')
    common.learn(classifier, learn_data)
    quantity_learn = common.count_quantity(classifier, learn_data)
    quantity_test = common.count_quantity(classifier, test_data)
    print("Quantity on:\nlearn_data: {}\ntest_data: {}".format(quantity_learn, quantity_test))
    print("support vectors:", sum(classifier.n_support_), ':', classifier.n_support_)
    show_plots(learn_data, test_data, classifier)


def part2():
    learn_data = get_data_from_file("resources/svmdata2.txt")
    test_data = get_data_from_file("resources/svmdata2test.txt")
    print("find the best parameter c for learn data")
    c, classifier = find_best_c(learn_data, learn_data)
    print("parameter c: {}\nquantity on learn data:{}\nquantity on test data: {}".format(c, common.count_quantity(
        classifier, learn_data), common.count_quantity(classifier, test_data)))
    print("support vectors:", sum(classifier.n_support_), ':', classifier.n_support_)
    print("find the best parameter c for test data")
    c, classifier = find_best_c(learn_data, test_data, min_parameter=1)
    print("parameter c: {}\nquantity on learn data:{}\nquantity on test data: {}".format(c, common.count_quantity(
        classifier, learn_data), common.count_quantity(classifier, test_data)))
    print("support vectors:", sum(classifier.n_support_), ':', classifier.n_support_)
    show_plots(learn_data, test_data, classifier)


def part3():
    learn_data = get_data_from_file("resources/svmdata3.txt")
    test_data = get_data_from_file("resources/svmdata3test.txt")


if __name__ == '__main__':
    # part1()
    part2()
