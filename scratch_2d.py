import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelmax

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle

from tifffile import TiffFile

# 2D Gaussian model
def func(xy, x0, y0, sigma, H):
    x, y = xy
    # x is equal to [0, 1, 2, 3, 4, 5, ..., 128, 1, 2, ..., 129] (len = 130*130)
    # y is equal to [0, 0, 0, 0, 0, 0, 0,   0,   1, 1, ..., 129] (len = 130*130)

    A = 1 / (2 * sigma**2)
    I = H * np.exp(-A * ( (x - x0)**2 + (y - y0)**2))
    # I is a 2D image ravel=ed (see numpy.ravel)
    return I

# Generate 2D gaussian
def generate(x0, y0, sigma, H):

    x = np.arange(0, max(x0, y0) * 2 + sigma, 1)
    y = np.arange(0, max(x0, y0) * 2 + sigma, 1)
    xx, yy = np.meshgrid(x, y)

    I = func((xx, yy), x0=x0, y0=y0, sigma=sigma, H=H)

    return xx, yy, I

def fit(image, with_bounds):

    # Prepare fitting
    x = np.arange(0, image.shape[1], 1)
    y = np.arange(0, image.shape[0], 1)
    xx, yy = np.meshgrid(x, y)

    # Guess intial parameters
    x0 = int(image.shape[0]) # Middle of the image
    y0 = int(image.shape[1]) # Middle of the image
    sigma = max(*image.shape) * 0.1 # 10% of the image
    H = np.max(image) # Maximum value of the image
    initial_guess = [x0, y0, sigma, H]

    # Constraints of the parameters
    if with_bounds:
        lower = [0, 0, 0, 0]
        upper = [image.shape[0], image.shape[1], max(*image.shape), image.max() * 2]
        bounds = [lower, upper]
    else:
        bounds = [-np.inf, np.inf]

    # Input: all positions of the image, output: image as 1D, parameters: description of Gaussian
    pred_params, uncert_cov = curve_fit(func, (xx.ravel(), yy.ravel()), image.ravel(),
                                        p0=initial_guess, bounds=bounds)

    # Get residual
    predictions = func((xx, yy), *pred_params)
    rms = np.sqrt(np.mean((image.ravel() - predictions.ravel())**2))

    print("True params : ", true_parameters)
    print("Predicted params : ", pred_params)
    print("Residual : ", rms)

    return pred_params

def plot(image, params):

    fig, ax = plt.subplots()
    ax.imshow(image, cmap="BrBG", interpolation='nearest', origin='lower')

    ax.scatter(params[0], params[1], s=100, c="red", marker="x")

    circle = Circle((params[0], params[1]), params[2], facecolor='none',
            edgecolor="red", linewidth=1, alpha=0.8)
    ax.add_patch(circle)


# Simulate and fit model
true_parameters = [50, 60, 10, 500]
xx, yy, image = generate(*true_parameters)

# The fit performs well without bounds
params = fit(image, with_bounds=False)
plot(image, params)
plt.show()