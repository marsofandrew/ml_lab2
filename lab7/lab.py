#!/usr/bin/python
from common_utilities import common
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from matplotlib import pyplot as plot


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
    def plot_part(data_element, color, label, regressor_color):
        plot.plot(data_element, c=color, label=label)
        regressor = LinearRegression()
        x = np.reshape(range(len(data_element)), (len(data_element), 1))
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
    pass


def part7():
    pass


def part8():
    pass


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
    part4()
