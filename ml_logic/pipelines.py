from grid_searches import evaluate_model_with_grid_search
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
import random

def complete_grid_search(VARIABLES, thresholds, n_samples, results_path):
    """
    Perform a complete grid search over multiple models with various parameter configurations.

    Parameters:
        VARIABLES (list): List of feature variables.
        thresholds (numpy.ndarray): Array of thresholds for binary classification.
        n_samples (int): Number of random parameter combinations to try for each model.
        results_path (str): Path to store the results.

    Returns:
        None
    """

    upsampling_methods = ['none', 'undersample', 'oversample', 'smote', 'cgan']

    models_params = {
        RandomForestClassifier: {
            'n_estimators': [100, 200, 300],
            'max_depth': [2, 3, 5, 9],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [2, 4],
            'max_features': ['log2', 'sqrt']
        },
        LogisticRegression: [
            {'C': 0.01, 'penalty': 'l2', 'solver': 'liblinear', 'max_iter': 100, 'class_weight': None},
            {'C': 0.1, 'penalty': 'l2', 'solver': 'liblinear', 'max_iter': 100, 'class_weight': 'balanced'},
            {'C': 1, 'penalty': 'l1', 'solver': 'saga', 'max_iter': 200, 'class_weight': 'balanced'},
            {'C': 10, 'penalty': 'elasticnet', 'solver': 'saga', 'l1_ratio': 0.5, 'max_iter': 200, 'class_weight': None},
            {'C': 100, 'penalty': 'none', 'solver': 'newton-cg', 'max_iter': 200, 'class_weight': 'balanced'}
        ],
        KNeighborsClassifier: {
            'n_neighbors': list(range(3, 9)),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2]
        },
        SVC: {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto'],
            'degree': [2, 3, 4, 5]
        },
        xgb.XGBClassifier: {
            'n_estimators': [50, 100, 150, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 7],
            'min_child_weight': [1, 2, 3, 4],
            'gamma': [0, 0.1, 0.2, 0.3],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_lambda': [0.01, 0.1, 1, 10],
            'reg_alpha': [0, 0.1, 1, 10]
        }
    }

    for model_class, params in models_params.items():
        if isinstance(params, list):
            # If params is a list, it means specific parameter configurations are provided
            try:
                evaluate_model_with_grid_search(VARIABLES, model_class, params, upsampling_methods, thresholds, results_path)
            except:
                print(f'error with  {model_class}, {params} ')
        else:
            # Generate random parameter combinations
            random_params = []
            for _ in range(n_samples):
                params_copy = {key: random.choice(value) for key, value in params.items()}
                random_params.append(params_copy)
            try:
                evaluate_model_with_grid_search(VARIABLES, model_class, random_params, upsampling_methods, thresholds, results_path)
            except:
                print(f'error with  {model_class}, {params} ')
