from base import ml_model
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from scipy.special import expit


class MyLogisticRegression(ml_model.Model):
    def __init__(self, copy=True, seed=2023, tol=0.0001):
        super().__init__(copy=copy, seed=seed)
        self.costs = []
        self.tol = tol

    def _sigmoid(self, X):
        weighted_sum = np.dot(X, self.weights)
        # OPTION 1: Vanilla sigmoid (prone to overflow if weighted_sum < 0)
        # return 1 / 1 + np.exp(-weighted_sum)
        # OPTION 2: Sigmoid that prevents overflow
        return np.where(weighted_sum >= 0,
                        1 / (1 + np.exp(-weighted_sum)),
                        np.exp(weighted_sum) / (1 + np.exp(weighted_sum)))
        # OPTION 3: Scipy's expit
        # return expit(weighted_sum)

    def fit(self, X, y):
        X = super()._preprocess_for_fit(X, y)
        n, m = X.shape[0], X.shape[1]

        self.weights = self.rng.normal(loc=0, scale=5, size=(m,))
        learning_rate = 0.03
        self.costs = []
        while True:
            # z is sigmoid function probabilities
            sigma = self._sigmoid(X)
            gradient = np.dot(X.T, (sigma - y)) / n  # Identical to lin reg
            self.weights -= learning_rate * gradient
            # Shape matching: y:(n,) and sigma:(n,)
            # cost is J(theta) = -l(theta)
            cost = -(np.dot(y.T, np.log(sigma)) + np.dot((1-y).T, np.log(1-sigma))) / n  # Will always be positive
            self.costs.append(cost)
            # Convergence Criterion
            if len(self.costs) >= 2 and abs(self.costs[-2] - self.costs[-2]) < self.tol:
                break

    def predict(self, X):
        X = super()._preprocess_for_predict(X)
        probabilities = self._sigmoid(X)
        probabilities[probabilities > 0.5] = 1
        probabilities[probabilities <= 0.5] = 0
        return probabilities


if __name__ == '__main__':
    my_model = MyLogisticRegression()
    X = np.array([[1, 1], [2, 2], [4, 4], [1, 2], [2, 4], [3, 4]])
    # Feature scaling to ensure circular cost-contours that allow for fast convergence of GD
    normalizer = Normalizer()
    X = normalizer.fit_transform(X)

    # Log reg calculates P(Y=1 | X; theta). So if prob > 0.5 => label is 1
    y = np.array([1, 1, 1, 0, 0, 0])
    my_model.fit(X, y)
    sklearn_model = LogisticRegression()
    sklearn_model.fit(X, y)

    X_test = np.array([[5, 5], [3, 0], [3, 6]])
    X_test = normalizer.transform(X_test)

    print(my_model.predict(X_test))
    print(sklearn_model.predict(X_test))

    assert my_model.predict(X_test).all() == sklearn_model.predict(X_test).all()
