import numpy as np
from enum import Enum

class Optimization(Enum):
    NORMAL, GD = 1,2

class LinearRegression:
    def __init__(self, fit_intercept=True, seed=2023, copy=True, optimization=Optimization.NORMAL):
        self.fit_intercept = fit_intercept  # To be used
        self.copy = copy
        self.rng = np.random.default_rng(seed)
        self.weights = None
        self.optimization = optimization

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError('Mismatch between X sample count and y label count. ')
        if self.copy:
            X = np.copy(X)
        # Add the dummy x0 variable to each row to fold intercept into weights vector
        X = np.insert(X, 0, 1, axis=1)
        n, m = X.shape[0], X.shape[1]
        if self.optimization == Optimization.NORMAL:
            inverse = np.linalg.inv(np.dot(X.T, X))
            xy = np.dot(X.T, y)
            self.weights = np.dot(inverse, xy)
        else:
            gradient = self.rng.random((m,))



if __name__ == '__main__':
    linreg = LinearRegression()
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    linreg.fit(X, y)
    assert linreg.weights.all() == np.array([3,1,2]).all()
    print(linreg.weights)
