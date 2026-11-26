"""
test_ds.py

Pytest-based tests for the Final Project ds.py scaffold.

These tests assume the following functions are defined in ds.py:

- load_data
- initial_summary
- clean_data
- feature_engineering
- prepare_features_and_target
- split_train_test
- train_regression_model
- predict_regression
- regression_metrics
- train_classification_model
- predict_classification
- classification_metrics
- run_clustering_model
- clustering_metrics
- full_regression_workflow
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score, calinski_harabasz_score

import ds


# ============================================================
# Helpers
# ============================================================

def _make_small_df() -> pd.DataFrame:
    """Create a small mixed-type DataFrame for basic tests."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 3],
            "group": ["A", "B", "A", "A"],
            "x1": [1.0, 2.0, 3.0, 3.0],
            "x2": [10, 20, 30, 30],
            "y_reg": [2.0, 4.0, 6.0, 6.0],
            "y_clf": [0, 1, 0, 0],
        }
    )


# ============================================================
# Part 1: Data Loading & Overview
# ============================================================

def test_load_data_reads_csv(tmp_path):
    """load_data should load a CSV file into a DataFrame."""
    df_expected = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
        }
    )
    path = tmp_path / "test.csv"
    df_expected.to_csv(path, index=False)

    df_loaded = ds.load_data(str(path))

    assert isinstance(df_loaded, pd.DataFrame)
    pd.testing.assert_frame_equal(df_loaded, df_expected)


def test_initial_summary_basic():
    """initial_summary should correctly report basic info and column types."""
    df = _make_small_df()
    summary = ds.initial_summary(df)

    assert summary["n_rows"] == len(df)
    assert summary["n_cols"] == df.shape[1]

    cols = summary["columns"]
    assert isinstance(cols, list)
    assert set(cols) == set(df.columns)

    numeric_cols = summary["numeric_columns"]
    cat_cols = summary["categorical_columns"]

    assert "x1" in numeric_cols
    assert "x2" in numeric_cols
    assert "y_reg" in numeric_cols
    assert "id" in numeric_cols
    assert "group" in cat_cols

    # No overlap
    assert set(numeric_cols).isdisjoint(set(cat_cols))


# ============================================================
# Part 2: Cleaning & Feature Engineering
# ============================================================

def test_clean_data_does_not_modify_input_and_drops_duplicates():
    """clean_data should return a new DataFrame and drop duplicate rows."""
    df = _make_small_df()
    df_copy = df.copy(deep=True)

    cleaned = ds.clean_data(df)

    # Original unchanged
    pd.testing.assert_frame_equal(df, df_copy)

    # At least one duplicate row removed (id=3 row appears twice)
    assert len(cleaned) <= len(df)
    assert cleaned.duplicated().sum() == 0


def test_feature_engineering_returns_same_length_dataframe():
    """feature_engineering should return a DataFrame with same number of rows."""
    df = _make_small_df()
    fe = ds.feature_engineering(df)

    assert isinstance(fe, pd.DataFrame)
    assert len(fe) == len(df)


# ============================================================
# Part 3: Features, Target & Splitting
# ============================================================

def test_prepare_features_and_target():
    """prepare_features_and_target should correctly slice X and y."""
    df = _make_small_df()
    feature_cols = ["x1", "x2"]
    target_col = "y_reg"

    X, y = ds.prepare_features_and_target(df, feature_cols, target_col)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert list(X.columns) == feature_cols
    assert y.name == target_col
    assert X.shape[0] == len(df)
    assert len(y) == len(df)


def test_split_train_test_stratify_false():
    """split_train_test should produce correct shapes without stratification."""
    df = _make_small_df()
    X = df[["x1", "x2"]]
    y = df["y_reg"]

    X_train, X_test, y_train, y_test = ds.split_train_test(
        X, y, test_size=0.25, stratify=False
    )

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    # Rough size check (3 train, 1 test)
    assert len(X_test) == 1
    assert len(y_test) == 1


def test_split_train_test_stratify_true():
    """split_train_test should support stratification for classification."""
    df = _make_small_df()
    X = df[["x1", "x2"]]
    y = df["y_clf"]

    X_train, X_test, y_train, y_test = ds.split_train_test(
        X, y, test_size=0.25, stratify=True
    )

    # All class labels should still be present in either train or test
    assert set(y.unique()).issubset(set(y_train.unique()) | set(y_test.unique()))


# ============================================================
# Part 4: Regression
# ============================================================

def test_train_regression_model_and_predict():
    """train_regression_model should return a model that can predict."""
    # Simple linear relation: y = 2 * x1
    X = pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([0.0, 2.0, 4.0, 6.0], name="y")

    model = ds.train_regression_model(X, y)
    y_pred = ds.predict_regression(model, X)

    assert hasattr(model, "predict")
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (len(X),)


def test_regression_metrics_correct_values():
    """regression_metrics should compute reasonable r2 and rmse."""
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 5.0])

    metrics = ds.regression_metrics(y_true, y_pred)

    assert "r2" in metrics
    assert "rmse" in metrics

    # Manual calculations
    # RSS = (0^2 + 0^2 + 0^2 + 1^2) = 1
    # TSS = sum((y_i - mean)^2) = 5
    # R2 = 1 - RSS/TSS = 1 - 1/5 = 0.8
    # RMSE = sqrt(1 / 4) = 0.5
    assert metrics["r2"] == pytest.approx(0.8, rel=1e-6)
    assert metrics["rmse"] == pytest.approx(0.5, rel=1e-6)


# ============================================================
# Part 5: Classification
# ============================================================

def test_train_classification_model_and_predict():
    """train_classification_model should return a classifier that can predict."""
    # Simple separable data
    X = pd.DataFrame(
        {
            "x1": [0.0, 0.1, 1.0, 1.1],
            "x2": [0.0, 0.1, 1.0, 1.1],
        }
    )
    y = pd.Series([0, 0, 1, 1], name="label")

    model = ds.train_classification_model(X, y)
    y_pred = ds.predict_classification(model, X)

    assert hasattr(model, "predict")
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (len(X),)
    # Should predict only valid labels
    assert set(np.unique(y_pred)).issubset({0, 1})


def test_classification_metrics_basic():
    """classification_metrics should compute accuracy and f1."""
    y_true = pd.Series([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    metrics = ds.classification_metrics(y_true, y_pred)

    assert "accuracy" in metrics
    assert "f1" in metrics

    # Manual:
    # TP=2, TN=1, FP=1, FN=0
    # accuracy = 3/4 = 0.75
    # precision = 2/3, recall = 1.0, f1 = 2*(2/3*1)/(2/3+1) = 0.8
    assert metrics["accuracy"] == pytest.approx(0.75, rel=1e-6)
    assert metrics["f1"] == pytest.approx(0.8, rel=1e-6)


# ============================================================
# Part 6: Clustering
# ============================================================

def test_run_clustering_model_returns_labels():
    """run_clustering_model should return labels with correct length."""
    # Two clear clusters in 2D
    rng = np.random.default_rng(0)
    X1 = rng.normal(loc=[0.0, 0.0], scale=0.1, size=(10, 2))
    X2 = rng.normal(loc=[3.0, 3.0], scale=0.1, size=(10, 2))
    X = np.vstack([X1, X2])
    X_df = pd.DataFrame(X, columns=["x1", "x2"])

    labels = ds.run_clustering_model(X_df, n_clusters=2)

    assert isinstance(labels, np.ndarray)
    assert labels.shape == (X_df.shape[0],)
    assert len(np.unique(labels)) == 2


def test_clustering_metrics_match_sklearn():
    """clustering_metrics should agree with sklearn's silhouette and CH scores."""
    rng = np.random.default_rng(1)
    X1 = rng.normal(loc=[0.0, 0.0], scale=0.1, size=(10, 2))
    X2 = rng.normal(loc=[2.0, 2.0], scale=0.1, size=(10, 2))
    X = np.vstack([X1, X2])
    labels = np.array([0] * 10 + [1] * 10)
    X_df = pd.DataFrame(X, columns=["x1", "x2"])

    metrics = ds.clustering_metrics(X_df, labels)

    assert "silhouette" in metrics
    assert "calinski_harabasz" in metrics

    sil_sk = silhouette_score(X_df, labels)
    ch_sk = calinski_harabasz_score(X_df, labels)

    assert metrics["silhouette"] == pytest.approx(sil_sk, rel=1e-6)
    assert metrics["calinski_harabasz"] == pytest.approx(ch_sk, rel=1e-6)


def test_clustering_metrics_single_cluster_returns_nan():
    """clustering_metrics should handle single-cluster case gracefully."""
    X_df = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "x2": [0.0, 1.0, 2.0]})
    labels = np.array([0, 0, 0])

    metrics = ds.clustering_metrics(X_df, labels)

    assert np.isnan(metrics["silhouette"])
    assert np.isnan(metrics["calinski_harabasz"])


# ============================================================
# Part 7: Full Regression Workflow
# ============================================================

def test_full_regression_workflow_end_to_end():
    """
    full_regression_workflow should run an end-to-end regression pipeline.

    This test assumes:
    - It returns a dict with keys: 'model', 'metrics', 'y_test', 'y_pred'
    - 'metrics' includes at least 'r2' and 'rmse'.
    """
    df = _make_small_df()
    feature_cols = ["x1", "x2"]
    target_col = "y_reg"

    result = ds.full_regression_workflow(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        test_size=0.25,
        random_state=ds.RANDOM_STATE,
    )

    assert isinstance(result, dict)
    assert "model" in result
    assert "metrics" in result
    assert "y_test" in result
    assert "y_pred" in result

    model = result["model"]
    metrics = result["metrics"]
    y_test = result["y_test"]
    y_pred = result["y_pred"]

    assert hasattr(model, "predict")
    assert isinstance(metrics, dict)
    assert "r2" in metrics
    assert "rmse" in metrics

    assert isinstance(y_test, pd.Series)
    assert isinstance(y_pred, np.ndarray)
    assert len(y_test) == len(y_pred)
