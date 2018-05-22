import random
from typing import List

import numpy
from numpy import ndarray
from sklearn.mixture import GaussianMixture

from core import Particle
from scratch import visualization


def perform(image: ndarray, cells: List[Particle]):
    means = numpy.empty((len(cells), 3), dtype=numpy.float64)
    for i in range(len(cells)):
        means[i, 0] = cells[i].x
        means[i, 1] = cells[i].y
        means[i, 2] = cells[i].z

    max_intensity = image.max()
    points = []
    iterator = numpy.nditer(image, flags=['multi_index'])
    while not iterator.finished:
        intensity = iterator[0]
        index = iterator.multi_index
        if random.random() < (intensity / max_intensity) / 10:
            # Generate more points for more intense pixels
            points.append([index[2] + random.random(), index[1] + random.random(), index[0] + random.random()])
        iterator.iternext()
    points = numpy.array(points)
    print("Created " + str(len(points)) + " points. Starting fit...")
    #fit the gaussian model
    gmm = GaussianMixture(n_components=len(cells), covariance_type='diag', means_init=means)
    gmm.fit(points)
    print("Done fitting!")

    #visualize
    visualization.visualize_3d_gmm(points, gmm.weights_, gmm.means_.T, numpy.sqrt(gmm.covariances_).T)
