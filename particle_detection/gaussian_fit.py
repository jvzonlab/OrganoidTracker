from sklearn.mixture import GaussianMixture
from typing import List
from numpy import ndarray
from core import Particle


def perform_fit(particles: List[Particle], blurred_image: ndarray):
    gmm=GaussianMixture(n_components=3, covariance_type="tied")
    gmm=gmm.fit(blurred_image)