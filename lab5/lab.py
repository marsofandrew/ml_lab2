#!/usr/bin/python
import math
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn import metrics
from common_utilities import common
from matplotlib import pyplot as plot
import matplotlib
import pandas as pd
import numpy as np
from lab5.additional_algorithms import kmedoids
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

MIN_ITERATIONS = 1
MAX_ITERATIONS = 5
SCALERS = [StandardScaler(), MinMaxScaler(), MaxAbsScaler()]

COLORS = ['red', 'green', 'blue']
AMOUNT_OF_POINTS_IN_CLUSTER = [250, 250, 250]


def visualization_multidimensional_space(data, labels, vmin, vmax):
    colormap = plot.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    axes = pd.plotting.scatter_matrix(data, color=colormap(norm(labels)))
    plot.show()


def show_dendogram(X):
    Z = linkage(X, method='ward')
    plot.figure()
    dendrogram(Z)
    plot.show()


def part1():
    data = pd.read_csv("resources/pluton.csv")

    def iteration(iterations):
        kmeans = KMeans(n_clusters=3, max_iter=iterations)
        kmeans.fit(data)
        visualization_multidimensional_space(data, kmeans.labels_, 0, 4)
        print(kmeans.n_iter_)
        return metrics.silhouette_score(data, kmeans.labels_)

    results = {}
    for i in range(MIN_ITERATIONS, MAX_ITERATIONS + 1):
        results[i] = iteration(i)
    print(results)


def part2_iteration(data_set, name_of_iteration):
    print(name_of_iteration)

    def count_distances(data, metric):
        distances = []
        for i in range(len(data)):
            element = []
            for j in range(len(data)):
                element.append(metric(data[i], data[j]))
            distances.append(element)
        return np.array(distances)

    def visualization(data_for_learn, labels, labels_vars):
        labels_vars = np.array(labels_vars)
        labels_vars = labels_vars.tolist()
        plot.subplot(2, 2, 1)
        plot.title("source_data")
        for i in range(len(data_set)):
            element = data_set[i]
            plot.scatter(element[:, 0], element[:, 1], c=COLORS[i])
        plot.subplot(2, 2, 4)
        plot.title("results")
        for i in range(len(data_for_learn)):
            plot.scatter(data_for_learn[i][0], data_for_learn[i][1], c=COLORS[labels_vars.index(labels[i])])
        plot.show()

    learn_data = data_set[0]
    for index in range(1, len(data_set)):
        learn_data = np.concatenate((learn_data, data_set[index]))
    for scaler in SCALERS:
        for metric in [distance.euclidean,
                       lambda first, second: math.fabs(first[0] - second[0]) + math.fabs(first[1] - second[1])]:
            normalize_data = scaler.fit_transform(learn_data)
            clusters, curr_medoids = kmedoids.cluster(count_distances(normalize_data, metric))
            print(scaler, metric)
            visualization(learn_data, clusters, curr_medoids)


def part2():
    def generate_2d_data(size, center_x, center_y, dispersion_x, dispersion_y):
        data_x = np.random.normal(center_x, dispersion_x, size)
        data_y = np.random.normal(center_y, dispersion_y, size)
        return np.array([[data_x[i], data_y[i]] for i in range(size)])

    data_set = [generate_2d_data(AMOUNT_OF_POINTS_IN_CLUSTER[0], 3, 50, 1.5, 6),  # example for bad classification
                generate_2d_data(AMOUNT_OF_POINTS_IN_CLUSTER[1], 15, 20, 10, 1.2),
                generate_2d_data(AMOUNT_OF_POINTS_IN_CLUSTER[2], 18, 25, 1, 12)]
    part2_iteration(data_set, "bad classification, has crosses")

    data_set = [generate_2d_data(AMOUNT_OF_POINTS_IN_CLUSTER[0], 3, 50, 11, 1.5),  # example for good classification
                generate_2d_data(AMOUNT_OF_POINTS_IN_CLUSTER[1], 10, 25, 15, 1.5),
                generate_2d_data(AMOUNT_OF_POINTS_IN_CLUSTER[2], 1, 10, 12, 1)]
    part2_iteration(data_set, "can has good classification 1")

    data_set = [generate_2d_data(AMOUNT_OF_POINTS_IN_CLUSTER[0], 50, 3, 1.5, 11),  # example for good classification
                generate_2d_data(AMOUNT_OF_POINTS_IN_CLUSTER[1], 25, 10, 1.5, 15),
                generate_2d_data(AMOUNT_OF_POINTS_IN_CLUSTER[2], 10, 1, 1, 12)]
    part2_iteration(data_set, "can has good classification 2")

    data_set = [generate_2d_data(AMOUNT_OF_POINTS_IN_CLUSTER[0], 50, 3, 11, 1.5),  # example for good classification
                generate_2d_data(AMOUNT_OF_POINTS_IN_CLUSTER[1], 25, 10, 15, 1.5),
                generate_2d_data(AMOUNT_OF_POINTS_IN_CLUSTER[2], 10, 1, 10, 1.2)]
    part2_iteration(data_set, "because of interest 3")


def part3():
    data = pd.read_csv("resources/votes.csv")
    data[np.isnan(data)] = 0
    show_dendogram(pdist(data))


def exit_function():
    exit()


if __name__ == '__main__':
    functions = {
        "execute part1": part1,
        "execute part2": part2,
        "execute part3": part3,
        "exit": exit_function
    }
    while True:
        command = input("input command: ")
        command = functions.get(command)
        if command is not None:
            common.executor(command)
        else:
            print("unsupported command")
        print("\n")
