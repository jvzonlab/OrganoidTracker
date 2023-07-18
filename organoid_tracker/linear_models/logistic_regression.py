from sklearn.metrics import classification_report, log_loss
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

import numpy


def platt_scaling(predictions, correct):

    # define model
    model = LogisticRegression(random_state=0, penalty='none')

    #
    eps = 10**-10
    predictions = numpy.log(predictions + eps) - numpy.log(1 - predictions + eps)

    # run model
    model.fit(predictions.reshape(-1,1), correct)

    intercept = model.intercept_
    coef = model.coef_[0]
    print(intercept)
    print(coef)

    model = LogisticRegression(random_state=0, penalty='none', fit_intercept=False)
    model.fit(predictions.reshape(-1, 1), correct)
    print(model.coef_[0])

    return intercept[0] , coef[0] ,model.coef_[0]