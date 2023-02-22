import numpy as np
from enum import Enum
from base import ml_model
from sklearn.linear_model import LinearRegression

class Optimization(Enum):
    NORMAL, GD = 1, 2


class MyLinearRegression(ml_model.Model):
    def __init__(self, fit_intercept=True, seed=2023, copy=True, optimization=Optimization.NORMAL):
        super().__init__(copy, seed)
        self.fit_intercept = fit_intercept  # To be used
        self.optimization = optimization

    def fit(self, X, y):
        X = super()._preprocess_for_fit(X, y)
        n, m = X.shape[0], X.shape[1]
        if self.optimization == Optimization.NORMAL:
            # (XTX)^-1 (XTY)
            inverse = np.linalg.inv(np.dot(X.T, X))
            xy = np.dot(X.T, y)
            self.weights = np.dot(inverse, xy)
        else:
            # Needs to be subtracted from weights. So has to be (m,)
            self.weights = self.rng.normal(loc=0, scale=0.1, size=(m,))

            # Assuming 100 epochs and learning rate = 0.01. TODO: Add convergence criteria
            learning_rate = 0.1
            for _ in range(10000):
                y_predicted = np.dot(X, self.weights)  # (n,m) dot (m,) = (n, )
                gradient = np.dot(X.T, (y_predicted - y)) / n  # (m,n) dot (n, ) = (m, )
                self.weights = self.weights - learning_rate * gradient

    def predict(self, X):
        X = super()._preprocess_for_predict(X)
        return np.dot(X, self.weights)


if __name__ == '__main__':
    linreg = MyLinearRegression(optimization=Optimization.GD)
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    linreg.fit(X, y)
    assert linreg.weights.all() == np.array([3,1,2]).all()

    sklearn_linreg = LinearRegression()
    sklearn_linreg.fit(X, y)

    assert linreg.predict(np.array([[2, 3], [4,5 ]])).all() == sklearn_linreg.predict(np.array([[2, 3], [4,5 ]])).all()

