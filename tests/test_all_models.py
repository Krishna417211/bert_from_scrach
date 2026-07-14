"""
Unit Tests for Machine Learning Algorithms
"""
import numpy as np
from src.ml_algorithms.registry import machine_learning_algorithms, list_categories
from src.ml_algorithms.utils.datasets import generate_regression_data, generate_classification_data
from src.ml_algorithms.regression.linear import LinearRegressionFromScratch
from src.ml_algorithms.classification.linear_classifiers import LogisticRegressionFromScratch

def test_registry():
    assert len(list_categories()) == 5

def test_linear_regression():
    X, y = generate_regression_data(n_samples=50)
    model = LinearRegressionFromScratch(lr=0.01, n_iters=200)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 50

def test_logistic_regression():
    X, y = generate_classification_data(n_samples=50)
    model = LogisticRegressionFromScratch(lr=0.01, n_iters=200)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 50

if __name__ == "__main__":
    test_registry()
    test_linear_regression()
    test_logistic_regression()
    print("All unit tests passed successfully!")

# Revision commit 2026-07-14 #19
