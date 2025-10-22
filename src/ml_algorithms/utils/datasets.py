"""
Synthetic Dataset Generators for ML Benchmarking
"""
import numpy as np

def generate_regression_data(n_samples=100, n_features=1, noise=0.1, random_state=42):
    np.random.seed(random_state)
    X = np.random.uniform(-3, 3, size=(n_samples, n_features))
    true_w = np.random.uniform(1.5, 4.0, size=(n_features, 1))
    true_b = 2.5
    y = X @ true_w + true_b + np.random.normal(0, noise, size=(n_samples, 1))
    return X, y.flatten()

def generate_classification_data(n_samples=150, n_features=2, n_classes=2, random_state=42):
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    weights = np.random.randn(n_features)
    logits = X @ weights
    y = (logits > 0).astype(int)
    return X, y

def generate_clustering_data(n_samples=200, n_clusters=3, random_state=42):
    np.random.seed(random_state)
    centers = [np.array([i*3, i*3]) for i in range(n_clusters)]
    X_list = []
    for center in centers:
        pts = center + np.random.randn(n_samples // n_clusters, 2) * 0.8
        X_list.append(pts)
    return np.vstack(X_list)

def generate_transaction_data(n_transactions=50, n_items=8, random_state=42):
    np.random.seed(random_state)
    items = [f"item_{i}" for i in range(n_items)]
    transactions = []
    for _ in range(n_transactions):
        size = np.random.randint(1, 5)
        basket = list(np.random.choice(items, size=size, replace=False))
        transactions.append(basket)
    return transactions

# Revision commit 2025-10-22 #7
