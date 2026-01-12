"""
Regularized Regression: Ridge, Lasso, and Elastic Net (From Scratch)
"""
import numpy as np

class RidgeRegressionFromScratch:
    def __init__(self, alpha=1.0, lr=0.01, n_iters=1000):
        self.alpha = alpha
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * (np.dot(X.T, (y_pred - y)) + self.alpha * self.weights)
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

class LassoRegressionFromScratch:
    def __init__(self, alpha=1.0, n_iters=500):
        self.alpha = alpha
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = np.mean(y)

        for _ in range(self.n_iters):
            for j in range(n_features):
                X_j = X[:, j]
                y_pred = np.dot(X, self.weights) + self.bias - self.weights[j] * X_j
                rho = np.dot(X_j, y - y_pred)

                if rho < -self.alpha:
                    self.weights[j] = (rho + self.alpha) / np.sum(X_j**2 + 1e-10)
                elif rho > self.alpha:
                    self.weights[j] = (rho - self.alpha) / np.sum(X_j**2 + 1e-10)
                else:
                    self.weights[j] = 0.0

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

class ElasticNetFromScratch:
    def __init__(self, alpha=1.0, l1_ratio=0.5, lr=0.01, n_iters=1000):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            l1_penalty = self.alpha * self.l1_ratio * np.sign(self.weights)
            l2_penalty = self.alpha * (1 - self.l1_ratio) * self.weights
            dw = (1 / n_samples) * (np.dot(X.T, (y_pred - y)) + l1_penalty + l2_penalty)
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Revision commit 2026-01-12 #4
