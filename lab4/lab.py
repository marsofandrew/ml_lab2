#!/usr/bin/python
from sklearn import svm
from lab4.helpers import *
import math

KERNELS = ['poly', 'rbf', 'sigmoid']
GAMMAS = ['auto', 'scale']
MAX_GAMMA = 100
STEP_GAMMA = 1
MAX_DEGREE = 15
STEP_DEGREE = 0.1
MIN_GAMMA = STEP_GAMMA


def exit_command():
    exit(0)


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

    def show_best_c(title, learn_data_param, test_data_param, min_parameter):
        print(title)
        c, classifier = find_best_c(learn_data_param, test_data_param, min_parameter=min_parameter)
        print("parameter c: {}\nquantity on learn data:{}\nquantity on test data: {}".format(c, common.count_quantity(
            classifier, learn_data), common.count_quantity(classifier, test_data_param)))
        print("support vectors:", sum(classifier.n_support_), ':', classifier.n_support_)
        show_plots(learn_data, test_data, classifier)

    show_best_c("find the best parameter c for learn data", learn_data, learn_data, 0.01)
    show_best_c("find the best parameter c for test data", learn_data, test_data, 1)


def part3():
    learn_data = get_data_from_file("resources/svmdata3.txt")
    test_data = get_data_from_file("resources/svmdata3test.txt")
    results = {}
    for kernel in KERNELS:
        classifier = svm.SVC(kernel=kernel)
        common.learn(classifier, learn_data)
        results[kernel] = common.count_quantity(classifier, test_data)
    print(results)
    kernel = common.find_key_of_max_value(results)
    print("best:", kernel)
    classifier = svm.SVC(kernel=kernel)
    common.learn(classifier, learn_data)
    show_plots(learn_data, test_data, classifier)

    results2 = {}
    i = 0
    while i <= MAX_DEGREE:
        classifier = svm.SVC(kernel='poly', degree=i, gamma='scale')
        common.learn(classifier, learn_data)
        results2[i] = common.count_quantity(classifier, test_data)
        i += STEP_DEGREE
    print(results2)
    common.show_plot_from_dict(results2, "quantity(degree)", 'quantity', 'degree')


def part4():
    learn_data = get_data_from_file("resources/svmdata4.txt")
    test_data = get_data_from_file("resources/svmdata4test.txt")

    classifiers = [
        svm.SVC(kernel=KERNELS[0], gamma='scale'),
        svm.SVC(kernel=KERNELS[1], gamma='scale'),
        svm.SVC(kernel=KERNELS[2], gamma='scale')
    ]
    results = {}
    for classifier in classifiers:
        common.learn(classifier, learn_data)
        results[classifier] = common.count_quantity(classifier, test_data)
    print(results)

    best_classifier = common.find_key_of_max_value(results)
    print("best classifier: {}; quantity: {}".format(best_classifier, results[best_classifier]))
    show_plots(learn_data, test_data, best_classifier)


def part5():
    learn_data = get_data_from_file("resources/svmdata5.txt")
    test_data = get_data_from_file("resources/svmdata5test.txt")
    results = {}
    full_results = {}
    t_results = {}

    for kernel in KERNELS:
        gamma = MIN_GAMMA
        while gamma <= MAX_GAMMA:
            classifier = svm.SVC(kernel=kernel, gamma=gamma)
            common.learn(classifier, learn_data)
            learn_quantity = common.count_quantity(classifier, learn_data)
            test_quantity = common.count_quantity(classifier, test_data)
            t_results[(kernel, gamma)] = test_quantity
            results[(kernel, gamma)] = math.fabs(learn_quantity - test_quantity)
            full_results[(kernel, gamma)] = {'learn_data': learn_quantity, 'test_data': test_quantity}
            gamma += STEP_GAMMA
    print(results)
    print(full_results)

    kernel_res, gamma_res = common.find_key_of_max_value(results)
    best_kernel, best_gamma = common.find_key_of_max_value(t_results)

    def visualization(kernel, gamma, title):
        print(title)
        print(kernel, gamma)
        classifier = svm.SVC(kernel=kernel, gamma=gamma)
        common.learn(classifier, learn_data)
        show_plots(learn_data, test_data, classifier)

    visualization(kernel_res, gamma_res, "retraining")
    visualization(best_kernel, best_gamma, "normal")


if __name__ == '__main__':
    functions = {
        "part1": part1,
        "part2": part2,
        "part3": part3,
        "part4": part4,
        "part5": part5,
        "exit": exit_command
    }

    while True:
        print("\n\n*******************************************\n")
        print("Write command, you can chose one of the following:\n{}".format(functions.keys()))
        command = input("Input command: ").strip()
        ex_command = functions.get(command, None)
        if ex_command is not None:
            common.executor(ex_command)
        else:
            print("You input unsupported command")
        print("****************************************")
