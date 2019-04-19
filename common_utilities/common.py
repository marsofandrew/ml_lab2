#!/usr/bin/python

import random
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from matplotlib import pyplot as plot


def get_transformed_data(data, ordinal_encoder=preprocessing.OrdinalEncoder()):
    """
    Transform provided by ordinal encoder
    :param data: dataset for transforming
    :param ordinal_encoder: encoder, as default preprocessing.OrdinalEncoder()
    :return: transformed data, ordinal_encoder
    """
    ordinal_encoder.fit(data)
    return ordinal_encoder.transform(data), ordinal_encoder


def divide_into_parts(data, parts, mix=True):
    data_set = np.array(data)
    data_list = data_set.tolist()
    data_length = len(data)
    if parts > data_length:
        raise ValueError("parts must be less or equal data length")

    if mix:
        for i in range(data_length * 10):
            i1 = int(random.uniform(0, data_length))
            i2 = int(random.uniform(0, data_length))
            data_list[i1], data_list[i2] = data_list[i2], data_list[i1]

    data_set = np.array(data_list)
    part_size = data_length // parts
    divided_data = []
    for i in range(parts - 1):
        divided_data.append(data_set[0:part_size])
        data_set = np.delete(data_set, slice(part_size), axis=0)
    divided_data.append(data_set)
    return divided_data


def get_learn_and_test_data(divided_data, test_indexes: list):
    test = []
    learn = []
    for i in range(len(divided_data)):
        part = np.array(divided_data[i]).tolist()
        if i in test_indexes:
            test += part
        else:
            learn += part
    return np.array(learn), np.array(test)


def count_quality_of_predictions(expected, predicted):
    conf_matrix = metrics.confusion_matrix(expected, predicted)
    main_diagonal = 0
    for i in range(len(conf_matrix)):
        main_diagonal += conf_matrix[i, i]
    amount = conf_matrix.sum()
    return main_diagonal / amount


def learn_and_count_quantity(classifier, data, parts):
    divided_data = divide_into_parts(data, parts)
    results = []
    for i in range(parts):
        learn_data, test = get_learn_and_test_data(divided_data, [i])
        learn(classifier, learn_data)
        results.append(count_quantity(classifier, test))
    return sum(results) / len(results)


def learn(classifier, data):
    x = data[:, :-1]
    y = data[:, -1:].T[0]
    classifier.fit(x, y)


def count_quantity(classifier, test_data):
    predicted = classifier.predict(test_data[:, :-1])
    return count_quality_of_predictions(test_data[:, -1:], predicted)


def show_plot_from_dict(results_dict: dict, name, y_name, x_name):
    plot.plot(results_dict.keys(), results_dict.values())
    plot.ylabel(y_name)
    plot.xlabel(x_name)
    plot.title(name)
    plot.show()


def find_key_of_max_value(dict_for_search: dict):
    if len(dict_for_search.keys()) <= 0:
        raise ValueError("Not enough data")
    best_key = list(dict_for_search.keys())[0]
    for key in dict_for_search.keys():
        if dict_for_search[key] > dict_for_search[best_key]:
            best_key = key
    return best_key


def replace_text_data(data, substitutes: list, substituted_key=-1):
    for element in data:
        element[substituted_key] = substitutes.index(element[substituted_key])
    return data


def executor(function):
    if not callable(function):
        raise ValueError('function must be callable and without parameters')
    print("{} is started".format(function.__name__))
    function()
    print("{} is finished".format(function.__name__))
