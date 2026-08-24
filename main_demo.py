"""
Main Benchmark Script showcasing all Machine Learning Algorithms
"""
from src.ml_algorithms.registry import machine_learning_algorithms
from src.ml_algorithms.utils.datasets import (
    generate_regression_data, generate_classification_data,
    generate_clustering_data, generate_transaction_data
)
from src.ml_algorithms.regression.linear import LinearRegressionFromScratch
from src.ml_algorithms.classification.linear_classifiers import LogisticRegressionFromScratch
from src.ml_algorithms.clustering.clustering import KMeansFromScratch
from src.ml_algorithms.dim_reduction.dim_reduction import PCAFromScratch
from src.ml_algorithms.association.association_rules import AprioriFromScratch

def main():
    print("=== Machine Learning Algorithms Demonstration ===")
    for cat, algos in machine_learning_algorithms.items():
        print(f"\nCategory: {cat}")
        for algo in algos:
            print(f" - {algo}")

    print("\n[1] Running Linear Regression From Scratch...")
    X_reg, y_reg = generate_regression_data()
    reg = LinearRegressionFromScratch()
    reg.fit(X_reg, y_reg)
    print("Linear Regression fit complete.")

    print("\n[2] Running Logistic Regression From Scratch...")
    X_cls, y_cls = generate_classification_data()
    cls = LogisticRegressionFromScratch()
    cls.fit(X_cls, y_cls)
    print("Logistic Regression fit complete.")

    print("\n[3] Running K-Means Clustering From Scratch...")
    X_cl = generate_clustering_data()
    km = KMeansFromScratch(k=3)
    labels = km.fit_predict(X_cl)
    print("K-Means clustering complete.")

    print("\n[4] Running PCA Dimensionality Reduction From Scratch...")
    pca = PCAFromScratch(n_components=2)
    X_reduced = pca.fit_transform(X_cl)
    print("PCA complete. Reduced shape:", X_reduced.shape)

    print("\n[5] Running Apriori Association Rules...")
    trans = generate_transaction_data()
    ap = AprioriFromScratch(min_support=0.1)
    rules = ap.fit(trans)
    print(f"Found {len(rules)} frequent itemsets.")

if __name__ == "__main__":
    main()

# Revision commit 2026-08-24 #8
