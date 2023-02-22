import abc
import numpy as np


class Model(metaclass=abc.ABCMeta):
    def __init__(self, copy=True, seed=2023):
        self.copy = copy
        self.rng = np.random.default_rng(seed)
        self.weights = None  # With folded-in bias.

    def _preprocess_for_fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError('Mismatch between X sample count and y label count.')
        if self.copy:
            X = np.copy(X)
            # Add a dummy feature x0 for bias.
        X = np.insert(X, 0, 1, axis=1)
        return X

    def fit(self, X, y):
       pass

    def _preprocess_for_predict(self, X):
        if self.copy:
            X = np.copy(X)
        # Add the dummy x0 variable to each row to fold intercept into weights vector
        X = np.insert(X, 0, 1, axis=1)
        return X

    def predict(self, X):
        pass