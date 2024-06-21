from sklearn.metrics import classification_report, log_loss
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

import numpy


def platt_scaling(predictions, correct):

    # define model
    model = LogisticRegression(random_state=0, penalty=None)

    #
    eps = 10**-10
    predictions = numpy.log(predictions + eps) - numpy.log(1 - predictions + eps)

    # run model
    model.fit(predictions.reshape(-1,1), correct)

    intercept = model.intercept_
    coef = model.coef_[0]

    model = LogisticRegression(random_state=0, penalty=None, fit_intercept=False)
    model.fit(predictions.reshape(-1, 1), correct)

    return intercept[0] , coef[0] ,model.coef_[0]