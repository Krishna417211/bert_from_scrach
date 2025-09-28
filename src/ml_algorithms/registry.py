"""
Machine Learning Algorithms Registry
"""

machine_learning_algorithms = {
    "Regression": [
        "Linear Regression",
        "Polynomial Regression",
        "Ridge Regression",
        "Lasso Regression",
        "Elastic Net Regression"
    ],

    "Classification": [
        "Logistic Regression",
        "K-Nearest Neighbors (KNN)",
        "Naive Bayes",
        "Decision Tree",
        "Random Forest",
        "Support Vector Machine (SVM)",
        "Gradient Boosting",
        "XGBoost",
        "LightGBM",
        "CatBoost"
    ],

    "Clustering": [
        "K-Means",
        "Hierarchical Clustering",
        "DBSCAN",
        "Gaussian Mixture Model (GMM)"
    ],

    "Dimensionality Reduction": [
        "Principal Component Analysis (PCA)",
        "Linear Discriminant Analysis (LDA)",
        "t-SNE",
        "UMAP"
    ],

    "Association Rule Learning": [
        "Apriori",
        "FP-Growth"
    ]
}

def list_categories():
    return list(machine_learning_algorithms.keys())

def get_algorithms(category):
    return machine_learning_algorithms.get(category, [])

# Revision commit 2025-09-28 #11
