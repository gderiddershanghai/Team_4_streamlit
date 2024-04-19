from upsampling import upsample_data
from preprocessor import preprocess_and_split
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.base import clone
from itertools import product
import numpy as np
from datetime import datetime
from sklearn.preprocessing import Binarizer
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import xgboost as xgb

VARIABLES = ['Age', 'BusinessTravel', 'DailyRate',
             'Department', 'DistanceFromHome', 'Education',
             'EmployeeCount', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
             'JobInvolvement', 'JobRole', 'JobSatisfaction',
             'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
             'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
             'RelationshipSatisfaction', 'StandardHours', 'Shift',
             'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
             'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
             'YearsWithCurrManager']

UPSAMPLING = ['none', 'undersample', 'oversample', 'smote', 'cgan']
THRESHOLDS = np.linspace(0.1,0.9,5)

X_train, X_test, y_train, y_test = preprocess_and_split(columns_to_use=VARIABLES)


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import make_scorer

def f1_weighted(recall, precision, beta):
    """Compute weighted F1 score."""
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

def compute_metrics(y_true, y_pred):
    """Computes specified metrics given true and predicted labels."""
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Custom weighted F1 scores
    f1_w2 = f1_weighted(recall, precision, beta=2)
    f1_w3 = f1_weighted(recall, precision, beta=3)
    f1_w4 = f1_weighted(recall, precision, beta=4)

    return {
        'recall': recall,
        'precision': precision,
        'accuracy': accuracy,
        'f1': f1,
        'f1_weighted_2': f1_w2,
        'f1_weighted_3': f1_w3,
        'f1_weighted_4': f1_w4
    }

def append_results_to_csv(VARIABLES, results, results_path):
    """
    Appends results to a CSV file, ensuring column consistency and preventing overwrites.
    """
    columns_order = ['date', 'model', 'upsampling', 'threshold', 'params','recall', 'precision', 'accuracy', 'f1',
                     'f1_weighted_2', 'f1_weighted_3', 'f1_weighted_4', 'test_recall', 'test_precision',
                     'test_accuracy', 'test_f1', 'test_f1_weighted_2', 'test_f1_weighted_3', 'test_f1_weighted_4',
                     f'{VARIABLES}']

    if os.path.exists(results_path):
        existing_df = pd.read_csv(results_path)
        new_df = pd.DataFrame(results)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(results_path, index=False, columns=columns_order)
    else:
        new_df = pd.DataFrame(results, columns=columns_order)
        new_df.to_csv(results_path, index=False, columns=columns_order)

# def evaluate_model_with_grid_search(X_train, y_train, X_test, y_test, model_class, model_params, upsampling_methods, thresholds, results_path):
#     """
#     Extends the model evaluation to include a grid search over model hyperparameters,
#     different decision thresholds, and upsampling methods.

#     Parameters:
#     - X_train, y_train: Training data and labels.
#     - X_test, y_test: Test data and labels.
#     - model_class: The machine learning model class to train and evaluate.
#     - model_params (list of dict): List of dictionaries, each representing a set of parameters for the model.
#     - upsampling_methods (list): Names of upsampling methods to use.
#     - thresholds (iterable): Decision thresholds to apply.
#     - results_path (str): Path to save the CSV file with the results.
#     """
#     results = []
#     current_date = datetime.now().strftime("%Y-%m-%d")

#     # Generate all combinations of parameters, upsampling methods, and thresholds
#     for params, method, threshold in product(model_params, upsampling_methods, thresholds):
#         X_resampled, y_resampled = upsample_data(X_train, y_train, method=method)

#         # Initialize the model with the current set of parameters
#         model = model_class(**params)
#         binarizer = Binarizer(threshold=threshold)
#         clf = clone(model)

#         # Perform cross-validation and compute metrics
#         y_prob_cv = cross_val_predict(clf, X_resampled, y_resampled, cv=5, method='predict_proba')[:, 1]
#         y_pred_cv = binarizer.transform(y_prob_cv.reshape(-1, 1)).reshape(-1)
#         cv_metrics = compute_metrics(y_resampled, y_pred_cv)

#         # Fit the model on the resampled training data and evaluate on the test set
#         clf.fit(X_resampled, y_resampled)
#         y_prob_test = clf.predict_proba(X_test)[:, 1]
#         y_pred_test = binarizer.transform(y_prob_test.reshape(-1, 1)).reshape(-1)
#         test_metrics = compute_metrics(y_test, y_pred_test)

#         results.append({
#             'date': current_date,
#             'model': model.__class__.__name__,
#             'params': str(params),  # Convert params to string for CSV storage
#             'upsampling': method,
#             'threshold': threshold,
#             **cv_metrics,
#             **{'test_' + k: v for k, v in test_metrics.items()}
#         })

#     # Use the new function to append results to CSV
#     if results_path is None:
#         results_path = '../Data/results_params_df_2.csv'
#     append_results_to_csv(results, results_path)


def evaluate_model_with_grid_search(VARIABLES, model_class, model_params, upsampling_methods, thresholds, results_path):
    """
    Extends the model evaluation to include a grid search over model hyperparameters,
    different decision thresholds, and upsampling methods, optimized by preprocessing upsampling.
    """

    X_train, X_test, y_train, y_test = preprocess_and_split(columns_to_use=VARIABLES)

    results = []
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Preprocess and store upsampling
    resampled_datasets = {}
    for method in upsampling_methods:
        X_resampled, y_resampled = upsample_data(X_train, y_train, method=method)
        resampled_datasets[method] = (X_resampled, y_resampled)

    # Iterate over all combinations
    for params, method, threshold in product(model_params, upsampling_methods, thresholds):
        X_resampled, y_resampled = resampled_datasets[method]

        # Initialize and prepare the model
        model = model_class(**params)
        binarizer = Binarizer(threshold=threshold)
        clf = clone(model)

        # Perform cross-validation
        y_prob_cv = cross_val_predict(clf, X_resampled, y_resampled, cv=5, method='predict_proba')[:, 1]
        y_pred_cv = binarizer.transform(y_prob_cv.reshape(-1, 1)).reshape(-1)
        cv_metrics = compute_metrics(y_resampled, y_pred_cv)

        # Train and evaluate on the test set
        clf.fit(X_resampled, y_resampled)
        y_prob_test = clf.predict_proba(X_test)[:, 1]
        y_pred_test = binarizer.transform(y_prob_test.reshape(-1, 1)).reshape(-1)
        test_metrics = compute_metrics(y_test, y_pred_test)

        # Append results
        results.append({
            'date': current_date,
            'model': model.__class__.__name__,
            'params': str(params),
            'upsampling': method,
            'threshold': threshold,
            **cv_metrics,
            **{'test_' + k: v for k, v in test_metrics.items()}
        })

    # Append results to CSV
    if not results_path:
        results_path = '../Data/results_params_df_2.csv'
    append_results_to_csv(VARIABLES, results, results_path)
