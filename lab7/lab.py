#!/usr/bin/python
from common_utilities import common
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plot
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
    res = regressor.score(np.reshape(data[:, 1], (len(data[:, 1]), 1)), data[:, 0], data[:, 2])

    print("score:", res)


def part2():
    pass


def part3():
    raw_data = __get_data("resources/longley.csv", ',')
    data = np.array(raw_data)

    divided_data = common.divide_into_parts(data, 2)
    Ridge()


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
    pass


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
    raw_data =__get_data('resources/svmdata6.txt', '\t')
    data = np.array(raw_data[1:, 1:], dtype=np.float)
    x = np.reshape(data[:,0], (data.shape[0], 1))
    results = {}
    epsilon = EPSILON_P8_STEP
    while epsilon <= MAX_EPSILON_P8:
        regressor = SVR(epsilon=epsilon, gamma='scale')
        regressor.fit(x, data[:,1])
        predictions = regressor.predict(x)
        results[epsilon] = mean_squared_error(data[:, 1], predictions)
        epsilon+=EPSILON_P8_STEP
    print(results)
    common.show_plot_from_dict(results, 'error(epsilon)', 'quantity', 'epsilon')


def main():
    functions = {
        'exit': [exit],
        'exec part1': [part1],
        'exec part2': [part2],
        'exec part3': [part3],
        'exec part4': [part4],
        'exec part5': [part5],
        'exec part6': [part6],
        'exec part7': [part7],
        'exec part8': [part8],
        'exec all': [part1, part2, part3, part4, part5, part6, part7, part8]
    }
    while True:
        cmd = input("input command:\n")
        commands = functions.get(cmd)
        if cmd is None:
            print("Unsupported commands")
        else:
            common.execute_all(commands)


if __name__ == '__main__':
    part8()
