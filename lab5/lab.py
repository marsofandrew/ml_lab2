#!/usr/bin/python
from sklearn.cluster import KMeans
import numpy as np


def part1():
    raw_data = np.loadtxt("resources/pluton.csv", delimiter=',', dtype=np.str)
    data = np.array(raw_data[1:, :], dtype=np.float)

    def iteration(iterations):
        kmeans = KMeans(n_clusters=3, max_iter=iterations)
        kmeans.fit(data)
        print(kmeans.labels_)

    iteration(100)

if __name__ == '__main__':
    part1()