#!/usr/bin/python
from common_utilities import common
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plot
from lab7.custom_reducer import Reducer

MAX_EPSILON_P8 = 1.5
EPSILON_P8_STEP = 0.1


def __get_data(file_path, delimiter):
    dataset = np.loadtxt(file_path, dtype=np.str, delimiter=delimiter)
    return dataset[1:, :]


def part1():
    raw_data = __get_data("resources/cygage.txt", '\t')
    data = np.array(raw_data, dtype=np.float)
    regressor = LinearRegression()
    regressor.fit(np.reshape(data[:, 1], (len(data[:, 1]), 1)), data[:, 0], data[:, 2])
    x = np.reshape(data[:, 1], (len(data[:, 1]), 1))
    res = regressor.score(x, data[:, 0], data[:, 2])
    plot.plot(data[:, 1], data[:, 0], c='black', label='data')
    predicted = regressor.predict(x)
    plot.plot(data[:, 1], predicted, 'r--', label='regression')
    print("score:", res)
    plot.legend()
    plot.show()


def part2():
    raw_data = __get_data("resources/reglab.txt", '\t')
    data = raw_data[1:, :]
    y = np.array(data[:, 0], dtype=np.float)
    x = np.array(data[:, 1:], dtype=np.float)
    regressors = [
        ("Linear", LinearRegression)
    ]
    for regressor in regressors:
        reducer = Reducer(regressor[1], mean_squared_error)
        print("regressor: {}, removed indexes: {}".format(regressor[0], reducer.reduce(x, y)))


def part3():
    raw_data = __get_data("resources/longley.csv", ',')
    data = np.array(raw_data, dtype=np.float)
    data = np.delete(data, -3, axis=1)
    divided_data = common.divide_into_parts(data, 2)
    learn = divided_data[0]
    test = divided_data[1]
    results_learn = []
    results_test = []
    for i in range(0, 26):
        lam = 10 ** (-3 + 0.2 * i)
        regressor = Ridge(alpha=lam)
        regressor.fit(learn[:, :-1], learn[:, -1])
        test_predict = regressor.predict(test[:, :-1])
        learn_predict = regressor.predict(learn[:, :-1])
        learn_error = mean_squared_error(learn[:, -1], learn_predict)
        test_error = mean_squared_error(test[:, -1], test_predict)
        results_learn.append(learn_error)
        results_test.append(test_error)
    plot.plot(results_test, c='r', label='test_error')
    plot.plot(results_learn, c='b', label='learn error')
    plot.legend()
    plot.show()


def part4():
    def plot_part(data_element, color, label, regressor_color, size_x=1):
        plot.plot(data_element, c=color, label=label)
        regressor = LinearRegression()
        x = np.reshape(range(len(data_element)), (len(data_element), size_x))
        regressor.fit(x, data_element)
        predictions = regressor.predict(x)
        plot.plot(predictions, regressor_color, label='regression of {}'.format(label))

    data = pd.read_csv("resources/eustock.csv")
    plot_part(data["DAX"], 'black', 'DAX', 'k--')
    plot_part(data["SMI"], 'red', 'SMI', 'r--')
    plot_part(data["CAC"], 'blue', 'CAC', 'b--')
    plot_part(data["FTSE"], 'green', 'DAX', 'g--')
    converted_data = np.multiply(0.25, (
            np.array(data["DAX"]) + np.array(data["SMI"]) + np.array(data["CAC"]) + np.array(data["FTSE"])))
    plot_part(converted_data, 'cyan', 'total', 'c--')
    plot.legend()
    plot.show()


def part5():
    data = pd.read_csv('resources/JohnsonJohnson.csv')
    data_q = [[] for _ in range(4)]
    for row in data.iterrows():
        if row[1][0].endswith("Q1"):
            x1 = int(row[1][0].split(" ")[0])
            data_q[0].append([x1, row[1][1]])
        if row[1][0].endswith("Q2"):
            x1 = int(row[1][0].split(" ")[0])
            data_q[1].append([x1, row[1][1]])
        if row[1][0].endswith("Q3"):
            x1 = int(row[1][0].split(" ")[0])
            data_q[2].append([x1, row[1][1]])
        if row[1][0].endswith("Q4"):
            x1 = int(row[1][0].split(" ")[0])
            data_q[3].append([x1, row[1][1]])

    x = np.array(data['index'])
    for i in range(len(x)):
        x[i] = x[i].replace(" Q1", '.0')
        x[i] = x[i].replace(" Q2", '.25')
        x[i] = x[i].replace(" Q3", '.50')
        x[i] = x[i].replace(" Q4", '.75')
    x = np.array(x, dtype=np.float)
    x = np.reshape(x, (len(x), 1))
    plot.plot(x, data['value'], c='black', label='data')
    regressors = [('linear', LinearRegression(), 'b--'), ('svr', SVR(gamma='scale'), 'g--'),
                  ('random forest', RandomForestRegressor(n_estimators=100), 'r--')]

    for regressor in regressors:
        regressor[1].fit(x, data['value'])
        predictions = regressor[1].predict(x)
        plot.plot(x, predictions, regressor[2], label="total:" + regressor[0])
        pred = regressor[1].predict([[2016.0], [2016.25], [2016.50], [2016.75]])
        print("{}: predictions for 2016 :{}\ttotal: {}".format(regressor[0], pred, sum(pred)))
    plot.legend()
    plot.show()

    results = {}
    for i, data_element in enumerate(data_q):
        data_element = np.array(data_element)
        results[i] = {}
        for regressor in regressors:
            x = np.reshape(data_element[:, 0], (len(data_element[:, 0]), 1))
            plot.plot(x, data_element[:, 1], c='black', label='data Q{}'.format(i))
            regressor[1].fit(x, data_element[:, 1])
            predictions = regressor[1].predict(x)
            plot.plot(x, predictions, regressor[2], label="Q{}: {}".format(i, regressor[0]))
            pred = regressor[1].predict([[2016]])
            results[i][regressor[0]] = pred
        plot.title("Q{}".format(i + 1))
        plot.legend()
        plot.show()

    for element in regressors:
        type = element[0]
        predict = 0
        for q in results.keys():
            predict += results[q][type]
        print(type, 'year', predict)

    print(results)


def part6():
    data = pd.read_csv("resources/sunspot.year.csv")
    plot.plot(data['index'], data['value'], c='b', label="data")
    regressor = LinearRegression()
    x = np.reshape(np.array(data['index']), (len(data['index']), 1))
    regressor.fit(x, data['value'])
    predictions = regressor.predict(x)
    plot.plot(x, predictions, 'r--', label='regression')
    plot.legend()
    plot.show()


def part7():
    data = pd.read_csv('resources/cars.csv')
    plot.plot(data['speed'], data['dist'], c='black', label='data')
    regressors = [('linear', LinearRegression(), 'r--'), ('ridge', Ridge(), 'g--')]
    x = np.reshape(np.array(data['speed']), (len(data['speed']), 1))
    for regressor in regressors:
        regressor[1].fit(x, data['dist'])
        predictions = regressor[1].predict(x)
        plot.plot(x, predictions, regressor[2], label=regressor[0])
        print("{}: predict for 40mps".format(regressor[0]), regressor[1].predict([[40]]))
    plot.legend()
    plot.show()


def part8():
    raw_data = __get_data('resources/svmdata6.txt', '\t')
    data = np.array(raw_data[1:, 1:], dtype=np.float)
    x = np.reshape(data[:, 0], (data.shape[0], 1))
    results = {}
    epsilon = EPSILON_P8_STEP
    while epsilon <= MAX_EPSILON_P8:
        regressor = SVR(epsilon=epsilon, gamma='scale')
        regressor.fit(x, data[:, 1])
        predictions = regressor.predict(x)
        results[epsilon] = mean_squared_error(data[:, 1], predictions)
        epsilon += EPSILON_P8_STEP
    print(results)
    common.show_plot_from_dict(results, 'error(epsilon)', 'error', 'epsilon')


def part9():
    data = pd.read_csv("resources/nsw74psid1.csv")
    y = np.array(data["re78"])
    x = np.array(data)[:, :-1]
    regressors = [
        ("DecisionTree", DecisionTreeRegressor()),
        ("Linear", LinearRegression()),
        ("Ridge", Ridge()),
        ("SVR", SVR(gamma='scale'))
    ]

    for regressor in regressors:
        regressor[1].fit(x, y)
        print("Type: {}; score: {}".format(regressor[0], regressor[1].score(x, y)))


def main():
    functions = {
        'exit': [lambda :exit()],
        'exec part1': [part1],
        'exec part2': [part2],
        'exec part3': [part3],
        'exec part4': [part4],
        'exec part5': [part5],
        'exec part6': [part6],
        'exec part7': [part7],
        'exec part8': [part8],
        'exec part9': [part9],
        'exec all': [part1, part2, part3, part4, part5, part6, part7, part8, part9]
    }
    while True:
        cmd = input("input command:\n")
        commands = functions.get(cmd)
        if commands is None:
            print("Unsupported commands")
        else:
            common.execute_all(commands)


if __name__ == '__main__':
    main()
