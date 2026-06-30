
from sklearn.linear_model import LogisticRegression


def build_logistic_regression():
    return LogisticRegression(
        max_iter=1000,
        C=1.0,
    )