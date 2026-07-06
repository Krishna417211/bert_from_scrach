"""
K-Means, Hierarchical, DBSCAN, and GMM Clustering from Scratch
"""
import numpy as np

class KMeansFromScratch:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit_predict(self, X):
        n_samples, n_features = X.shape
        random_idxs = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_idxs]

        for _ in range(self.max_iters):
            clusters = [[] for _ in range(self.k)]
            labels = np.zeros(n_samples, dtype=int)
            for idx, x in enumerate(X):
                centroid_idx = np.argmin(np.sqrt(np.sum((x - self.centroids)**2, axis=1)))
                clusters[centroid_idx].append(idx)
                labels[idx] = centroid_idx

            old_centroids = self.centroids.copy()
            for cluster_idx, cluster in enumerate(clusters):
                if cluster:
                    self.centroids[cluster_idx] = np.mean(X[cluster], axis=0)

            if np.all(old_centroids == self.centroids):
                break

        return labels

class DBSCANFromScratch:
    def __init__(self, eps=1.0, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, X):
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1)
        cluster_id = 0

        for i in range(n_samples):
            if labels[i] != -1:
                continue
            neighbors = self._region_query(X, i)
            if len(neighbors) < self.min_samples:
                labels[i] = -1  # Noise
            else:
                self._expand_cluster(X, labels, i, neighbors, cluster_id)
                cluster_id += 1
        return labels

    def _region_query(self, X, i):
        distances = np.sqrt(np.sum((X - X[i])**2, axis=1))
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, labels, i, neighbors, cluster_id):
        labels[i] = cluster_id
        k = 0
        while k < len(neighbors):
            neighbor = neighbors[k]
            if labels[neighbor] == -1:
                labels[neighbor] = cluster_id
            elif labels[neighbor] == -1:
                labels[neighbor] = cluster_id
                n_neighbors = self._region_query(X, neighbor)
                if len(n_neighbors) >= self.min_samples:
                    neighbors = np.concatenate((neighbors, n_neighbors))
            k += 1

class GaussianMixtureFromScratch:
    def __init__(self, n_components=3, max_iter=100):
        self.n_components = n_components
        self.max_iter = max_iter

    def fit_predict(self, X):
        n_samples, n_features = X.shape
        self.weights = np.full(self.n_components, 1 / self.n_components)
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances = [np.eye(n_features) for _ in range(self.n_components)]

        for _ in range(self.max_iter):
            # Expectation
            responsibilities = np.zeros((n_samples, self.n_components))
            for k in range(self.n_components):
                diff = X - self.means[k]
                norm = 1.0 / (2 * np.pi * np.linalg.det(self.covariances[k])**0.5 + 1e-6)
                exponent = -0.5 * np.sum(diff @ np.linalg.inv(self.covariances[k] + 1e-6 * np.eye(n_features)) * diff, axis=1)
                responsibilities[:, k] = self.weights[k] * norm * np.exp(exponent)

            responsibilities /= (np.sum(responsibilities, axis=1, keepdims=True) + 1e-10)

            # Maximization
            N_k = np.sum(responsibilities, axis=0)
            for k in range(self.n_components):
                self.means[k] = np.sum(responsibilities[:, k:k+1] * X, axis=0) / (N_k[k] + 1e-10)
                diff = X - self.means[k]
                self.covariances[k] = (responsibilities[:, k:k+1] * diff).T @ diff / (N_k[k] + 1e-10)
                self.weights[k] = N_k[k] / n_samples

        return np.argmax(responsibilities, axis=1)

# Revision commit 2026-07-06 #18
