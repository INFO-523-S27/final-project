"""
ds.py

Final Project: From Data to Decisions with TidyTuesday

This module defines a reusable, testable data science pipeline for your
final project. You will:

- Load and summarize a TidyTuesday dataset.
- Clean data and engineer features.
- Prepare feature matrices and target vectors.
- Train and evaluate at least TWO of the following:
    * Regression model(s)
    * Classification model(s)
    * Clustering model(s)

IMPORTANT:
- Do NOT rename any existing functions.
- Do NOT change function signatures.
- You may add small helper functions, but keep the public API unchanged.
- Keep this file free of plotting/printing. Use your notebook for EDA
  and visualization, and call these functions from there.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator


RANDOM_STATE: int = 511


# ============================================================
# Part 1: Data Loading & Overview
# ============================================================

def load_data(path: str) -> pd.DataFrame:
    """
    Load the main TidyTuesday dataset from a CSV file.

    Parameters
    ----------
    path : str
        Path to the dataset CSV (e.g., 'data/my_dataset.csv').

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Notes
    -----
    - Use `pd.read_csv`.
    - Do NOT hardcode the path; always use the `path` argument.
    """
    raise NotImplementedError("Implement load_data().")


def initial_summary(df: pd.DataFrame) -> dict[str, object]:
    """
    Compute a basic summary of the dataset for initial exploration.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict[str, object]
        A dictionary containing:
        - 'n_rows': int, number of rows
        - 'n_cols': int, number of columns
        - 'columns': list[str], all column names
        - 'numeric_columns': list[str], numeric columns
        - 'categorical_columns': list[str], non-numeric columns

    Notes
    -----
    - This function should NOT print anything.
    - Your notebook can display or interpret these results.
    """
    raise NotImplementedError("Implement initial_summary().")


# ============================================================
# Part 2: Cleaning & Feature Engineering
# ============================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning on the dataset.

    Suggested operations (adapt to your project):
    - Drop exact duplicate rows.
    - Handle missing values (drop or impute as justified in your report).
    - Filter to a subset of rows if necessary (e.g., specific years).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Cleaned dataset (new copy; do NOT modify `df` in-place).
    """
    raise NotImplementedError("Implement clean_data().")


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional columns that may be useful for modeling.

    Examples (depending on the dataset):
    - Ratios (e.g., `value_per_capita`)
    - Aggregations or group-level features
    - Categorical encodings (e.g., collapse rare categories)
    - Date features (e.g., year, month, weekday)

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with engineered features added.

    Notes
    -----
    - You may choose to return a subset of columns if you wish.
    - Do NOT perform one-hot encoding here; leave that for modeling
      or preprocessing steps if needed.
    """
    raise NotImplementedError("Implement feature_engineering().")


# ============================================================
# Part 3: Feature/Target Preparation & Splitting
# ============================================================

def prepare_features_and_target(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare the feature matrix X and target vector y for supervised models.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset after cleaning and feature engineering.
    feature_cols : Iterable[str]
        Column names to use as input features.
    target_col : str
        Column name to use as the target.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with columns in `feature_cols`.
    y : pd.Series
        Target vector containing values from `target_col`.

    Notes
    -----
    - This function should NOT modify `df` in-place.
    - Your notebook can inspect X and y for sanity checks.
    """
    raise NotImplementedError("Implement prepare_features_and_target().")


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    stratify: bool = False,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and target into train and test sets.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    test_size : float, default 0.2
        Proportion of data reserved for testing.
    stratify : bool, default False
        If True, use `y` to stratify the split (recommended for classification).
    random_state : int, default RANDOM_STATE
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test

    Notes
    -----
    - Use `sklearn.model_selection.train_test_split`.
    - For regression, `stratify` should remain False.
    """
    raise NotImplementedError("Implement split_train_test().")


# ============================================================
# Part 4: Regression Pipeline
# ============================================================

def train_regression_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> BaseEstimator:
    """
    Train a regression model on the training data.

    You may start with a simple baseline such as:
    - `sklearn.linear_model.LinearRegression`
    - or another regression model justified in your report.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series

    Returns
    -------
    model : BaseEstimator
        Fitted regression model.

    Notes
    -----
    - You may add preprocessing inside a Pipeline if appropriate.
    - Do NOT evaluate inside this function; keep it for training only.
    """
    raise NotImplementedError("Implement train_regression_model().")


def predict_regression(
    model: BaseEstimator,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Generate predictions using a trained regression model.

    Parameters
    ----------
    model : BaseEstimator
        Fitted regression model.
    X_test : pd.DataFrame

    Returns
    -------
    np.ndarray
        Predicted numeric values.
    """
    raise NotImplementedError("Implement predict_regression().")


def regression_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute regression evaluation metrics.

    Suggested metrics:
    - R-squared ('r2')
    - Root Mean Squared Error ('rmse')

    Parameters
    ----------
    y_true : pd.Series
    y_pred : np.ndarray

    Returns
    -------
    dict[str, float]
        Dictionary of metric_name -> value.
    """
    raise NotImplementedError("Implement regression_metrics().")


# ============================================================
# Part 5: Classification Pipeline
# ============================================================

def train_classification_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> BaseEstimator:
    """
    Train a classification model on the training data.

    You may start with:
    - `sklearn.linear_model.LogisticRegression`
    - `sklearn.ensemble.RandomForestClassifier`
    - or another classifier justified in your report.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series

    Returns
    -------
    model : BaseEstimator
        Fitted classification model.
    """
    raise NotImplementedError("Implement train_classification_model().")


def predict_classification(
    model: BaseEstimator,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Generate predicted class labels from a fitted classifier.

    Parameters
    ----------
    model : BaseEstimator
    X_test : pd.DataFrame

    Returns
    -------
    np.ndarray
        Predicted class labels.
    """
    raise NotImplementedError("Implement predict_classification().")


def classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute classification evaluation metrics.

    Suggested metrics:
    - Accuracy ('accuracy')
    - F1-score ('f1')
    - You may add others (e.g., 'precision', 'recall', 'roc_auc')
      if appropriate for your problem.

    Parameters
    ----------
    y_true : pd.Series
    y_pred : np.ndarray

    Returns
    -------
    dict[str, float]
        Dictionary of metric_name -> value.

    Notes
    -----
    - For multiclass problems, be explicit about averaging strategy
      (e.g., macro vs weighted) and document it in your report.
    """
    raise NotImplementedError("Implement classification_metrics().")


# ============================================================
# Part 6: Clustering Pipeline
# ============================================================

def run_clustering_model(
    X: pd.DataFrame,
    n_clusters: int,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """
    Run a clustering algorithm on the feature matrix X.

    You may start with:
    - `sklearn.cluster.KMeans`
    - or another clustering model justified in your report.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (numeric).
    n_clusters : int
        Number of clusters to form.
    random_state : int, default RANDOM_STATE

    Returns
    -------
    np.ndarray
        Cluster labels for each row in X.

    Notes
    -----
    - Do NOT compute metrics here; just return labels.
    - Your notebook can call clustering_metrics(...) after this function.
    """
    raise NotImplementedError("Implement run_clustering_model().")


def clustering_metrics(
    X: pd.DataFrame,
    labels: np.ndarray,
) -> dict[str, float]:
    """
    Compute clustering evaluation metrics.

    Suggested metrics:
    - Silhouette score ('silhouette')
    - Calinski–Harabasz score ('calinski_harabasz')

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (numeric).
    labels : np.ndarray
        Cluster labels.

    Returns
    -------
    dict[str, float]
        Dictionary of metric_name -> value.

    Notes
    -----
    - If there are fewer than 2 clusters in `labels`, you may return
      NaN for the metrics or raise a ValueError. Be consistent.
    """
    raise NotImplementedError("Implement clustering_metrics().")


# ============================================================
# Part 7: High-Level Convenience Wrappers (Optional)
# ============================================================

def full_regression_workflow(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Optional convenience function to run a full regression workflow:

    1. Prepare features and target.
    2. Split into train/test sets.
    3. Train regression model.
    4. Generate predictions.
    5. Compute metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned + feature-engineered dataset.
    feature_cols : Iterable[str]
    target_col : str
    test_size : float, default 0.2
    random_state : int, default RANDOM_STATE

    Returns
    -------
    dict[str, Any]
        A dictionary that may include:
        - 'model': fitted model
        - 'metrics': regression metrics dict
        - 'y_test': true test values (pd.Series)
        - 'y_pred': predicted values (np.ndarray)

    Notes
    -----
    - This is optional but can reduce boilerplate in your notebook.
    - You may adapt this pattern for classification if useful.
    """
    raise NotImplementedError("Implement full_regression_workflow() or remove if unused.")
