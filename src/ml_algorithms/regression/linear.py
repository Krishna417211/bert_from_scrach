"""
Linear and Polynomial Regression (From Scratch & Sklearn)
"""
import numpy as np
try:
    from sklearn.linear_model import LinearRegression as SklearnLinearRegression
    from sklearn.preprocessing import PolynomialFeatures as SklearnPoly
except ImportError:
    SklearnLinearRegression = None
    SklearnPoly = None

class LinearRegressionFromScratch:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

class PolynomialRegressionFromScratch:
    def __init__(self, degree=2, lr=0.01, n_iters=1000):
        self.degree = degree
        self.linear_model = LinearRegressionFromScratch(lr=lr, n_iters=n_iters)

    def _transform(self, X):
        X_poly = [X**i for i in range(1, self.degree + 1)]
        return np.hstack(X_poly)

    def fit(self, X, y):
        X_poly = self._transform(X)
        self.linear_model.fit(X_poly, y)

    def predict(self, X):
        X_poly = self._transform(X)
        return self.linear_model.predict(X_poly)

# Revision commit 2026-08-07 #8
