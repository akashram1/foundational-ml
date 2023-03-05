from base import ml_model
import numpy as np


class Perceptron(ml_model.Model):
    def __init__(self, copy=True, seed=2023, tol=0.0001, lr=0.01, max_iter=1000):
        super().__init__(copy=copy, seed=seed)
        self.lr = lr
        # Break after max_iter if perceptron still has not converged
        self.max_iter = max_iter
        self.mistakes_by_iter = []

    @staticmethod
    def activate(p):
        return np.where(p >= 0, 1, -1)  # unit step function

    def fit(self, X, y):
        X = super()._preprocess_for_fit(X, y)
        n, m = X.shape[0], X.shape[1]

        # TODO: What happens when we initialize weight to zero
        self.weights = self.rng.normal(loc=0, scale=0.0001, size=(m,))
        self.mistakes_by_iter = []
        iter_count = 0
        while True:
            # Pass over whole dataset
            mistakes_count = 0
            for i in range(n):
                # Weights : (m, ,). X[i] = (m, )
                linear_output = np.dot(self.weights.T, X[i])
                y_pred = Perceptron.activate(linear_output)
                # Update weights only if a mistake is made
                if y_pred != y[i]:
                    mistakes_count += 1
                    # (m, ) = (m, ) + (m, )
                    # Note in numpy a single row with m cols in a 2d (n,m) matrix is always represented as (m, )
                    # So X[0].shape = X[0].T.shape = (m, )
                    self.weights = self.weights + y[i] * X[i].T  # Same as w = w + y[i] * X[i]
            iter_count += 1
            self.mistakes_by_iter.append(mistakes_count)
            if mistakes_count == 0:
                print(f'Perceptron converged in {iter_count} iterations over the dataset.')
                break
            if iter_count == self.max_iter:
                print('Perceptron did not converge. Data may not be linearly separable or more iters needed.')

    def predict(self, X):
        X = super()._preprocess_for_predict(X)
        linear_output = np.dot(X, self.weights)
        return Perceptron.activate(linear_output)