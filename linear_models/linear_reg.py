import numpy as np


class LinearRegression():
    def __int__(self, fit_intercept=True, seed=2023, copy=True):
        self.fit_intercept = fit_intercept # To be used
        #self.copy = copy
        self.rng = np.random.default_rng(seed)
        self.weights = []
        self.bias = 0

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError('Mismatch between X sample count and y label count. ')

        #if self.copy:
        X = np.copy(X)

        # Add the dummy x0 variable to each row to fold intercept into weights vector
        X = np.insert(X, 0, 1, axis=1)
        n, m = X.shape[0], X.shape[1]
        inverse = np.linalg.inv(np.dot(X.T, X))
        xy = np.dot(X.T, y)
        return np.dot(inverse, xy)


if __name__ == '__main__':
    linreg = LinearRegression()
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    print(linreg.fit(X, y))