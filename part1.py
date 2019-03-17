#!/usr/bin/python

import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import common




def get_results(dataset):
    pass

dataset = np.loadtxt("Tic_tac_toe.txt", delimiter=",", dtype=np.str)
dataset2 = np.loadtxt("spam.csv", delimiter=",", dtype=np.str)

transformed_data = common.get_transformed_data(dataset)
transformed_data2 = common.get_transformed_data(dataset2)



