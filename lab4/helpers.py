#!/usr/bin/python

import numpy as np
from common_utilities import common
from matplotlib import pyplot as plot

SUBSTITUTES = ['red', 'green']


def get_data_from_file(filename):
    raw_data = np.loadtxt(filename, delimiter='\t', dtype=np.str)
    return np.array(common.replace_text_data(raw_data[1:, 1:], SUBSTITUTES), dtype=np.float)


def show_plots(learn_data, test_data, classifier):
    vectors = classifier.support_vectors_
    amount_of_vectors = classifier.n_support_
    index = 0
    support_vectors_points = []

    for support_vectors in amount_of_vectors:
        support_vectors_points.append([])
        for j in range(support_vectors):
            support_vectors_points[-1].append(vectors[index])
            index += 1

    xx, yy = make_meshgrid(learn_data[:, 0], learn_data[:, 1])
    fig, sub_plots = plot.subplots(2, 1)
    for title, ax, data in zip(['learn data', 'test_data'], sub_plots, [learn_data, test_data]):
        plot_contours(ax, classifier, xx, yy)
        for elem in data:
            ax.scatter(elem[0], elem[1], c=SUBSTITUTES[int(elem[-1])])
            ax.set_xlabel("x1")
            ax.set_ylabel('x2')
            ax.set_title(title)
            ax.set_xticks(())
            ax.set_yticks(())

    plot.show()


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out
