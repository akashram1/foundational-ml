from base import ml_model
import numpy as np


class LogisticRegression(ml_model.Model):
    def __init__(self, copy=True, seed=2023):
        super().__init__(copy=copy, seed=seed)

    def _sigmoid(self, X):
        weighted_sum = np.dot(X, self.weights)
        return np.exp(-weighted_sum) / 1 + np.exp(-weighted_sum)

    def fit(self, X, y):
        X = super()._preprocess_for_fit(X, y)
        n, m = X.shape[0], X.shape[1]

        self.weights = self.rng.normal(loc=0, scale=0.1, size=(m,))

        learning_rate = 0.1
        # TODO: Add a convergence criteria
        for _ in range(10000):
            # z is sigmoid function probabilities
            z = self._sigmoid(X)
            gradient = np.dot(X.T, (z - y)) / n  # Identical to lin reg
            self.weights -= learning_rate * gradient

    def predict(self, X):
        super()._preprocess_for_predict(X)
        probabilities = self._sigmoid(np.dot(X, self.weights))
        # TODO: Convert to labels


if __name__ == '__main__':
    model = LogisticRegression()
