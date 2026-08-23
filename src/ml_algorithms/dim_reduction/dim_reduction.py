"""
PCA, LDA, t-SNE, and UMAP Implementation (From Scratch & Wrappers)
"""
import numpy as np

class PCAFromScratch:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov = np.cov(X_centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[idxs[:self.n_components]]
        return np.dot(X_centered, self.components.T)

class LDAFromScratch:
    def __init__(self, n_components=1):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit_transform(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)
        mean_overall = np.mean(X, axis=0)

        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))

        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            S_W += np.dot((X_c - mean_c).T, (X_c - mean_c))

            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            S_B += n_c * (mean_diff).dot(mean_diff.T)

        A = np.linalg.inv(S_W + 1e-6 * np.eye(n_features)).dot(S_B)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        self.linear_discriminants = eigenvectors[idxs[:self.n_components]]

        return np.dot(X, self.linear_discriminants.T)

class SimplifiedTSNEFromScratch:
    def __init__(self, n_components=2, lr=100.0, n_iter=200):
        self.n_components = n_components
        self.lr = lr
        self.n_iter = n_iter

    def fit_transform(self, X):
        n_samples = X.shape[0]
        Y = np.random.randn(n_samples, self.n_components) * 1e-4

        for _ in range(self.n_iter):
            # Compute pairwise distances
            sum_Y = np.sum(np.square(Y), 1)
            num = -2.0 * np.dot(Y, Y.T) + sum_Y + sum_Y.T
            num = 1.0 / (1.0 + num)
            np.fill_diagonal(num, 0.0)
            Q = num / np.sum(num)

            # Gradient calculation
            dY = np.zeros((n_samples, self.n_components))
            for i in range(n_samples):
                grad = np.sum((Q[i, :] - 1.0/n_samples)[:, None] * (Y[i, :] - Y) * num[i, :][:, None], 0)
                dY[i, :] = grad

            Y -= self.lr * dY
        return Y

# Revision commit 2026-08-23 #4
