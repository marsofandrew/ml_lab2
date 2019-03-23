#!/usr/bin/python

import random
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from matplotlib import pyplot as plot

TEST = 'test'
LEARN = 'learn'
CORRECT = 'correct'


def get_transformed_data(data):
    oe = preprocessing.OrdinalEncoder()
    oe.fit(data)
    return oe.transform(data), oe


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
    main_diagonal = conf_matrix[0, 0] + conf_matrix[1, 1]
    amount = conf_matrix.sum()
    return main_diagonal / amount


def learn(classifier, data, parts):
    divided_data = divide_into_parts(data, parts)
    results = []
    for i in range(parts):
        learn, test = get_learn_and_test_data(divided_data, [i])
        x = learn[:, :-1]
        y = learn[:, -1:].T[0]
        classifier.fit(x, y)
        predicted = classifier.predict(test[:, :-1])
        results.append(count_quality_of_predictions(test[:, -1:], predicted))
    return sum(results) / len(results)

