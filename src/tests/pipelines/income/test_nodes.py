import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from unittest.mock import patch, MagicMock
import sys
import os

# Adjust path to import the nodes module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../income_kedro/pipelines/income')))

from nodes import (
    clean_data,
    split_data,
    fit_preprocessor,
    apply_preprocessor,
    train_model,
    _encode_label,
    _predict,
    _predict_proba,
    _gen_metrics,
    evaluate_model,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'age': [25, 35, 45, 55, 65],
        'education': ['HS-grad', 'Bachelors', 'Masters', 'Doctorate', 'HS-grad'],
        'hours.per.week': [40, 50, 60, 35, 40],
        'income': ['>50K', '<=50K', '>50K', '>50K', '<=50K']
    })


@pytest.fixture
def sample_df_with_nan():
    """Create a sample DataFrame with NaN values."""
    return pd.DataFrame({
        'age': [25, 35, np.nan, 55, 65],
        'education': ['HS-grad', 'Bachelors', 'Masters', None, 'HS-grad'],
        'hours.per.week': [40, 50, 60, 35, 40],
        'income': ['>50K', '<=50K', '>50K', '>50K', '<=50K']
    })


@pytest.fixture
def feature_params():
    """Create feature parameters."""
    return {
        'cat_cols': ['education'],
        'num_cols': ['age', 'hours.per.week'],
        'target_col': 'income'
    }


@pytest.fixture
def split_params():
    """Create split parameters."""
    return {
        'test_size': 0.2,
        'random_state': 42
    }


@pytest.fixture
def model_params():
    """Create model parameters."""
    return {
        'n_estimators': 10,
        'random_state': 42,
        'max_depth': 5
    }


@pytest.fixture
def model_configs():
    """Create model configurations."""
    return {
        'threshold': 0.5
    }


class TestCleanData:
    def test_clean_data_removes_nan(self, sample_df_with_nan):
        """Test that clean_data removes rows with NaN values."""
        result = clean_data(sample_df_with_nan)
        assert result.isnull().sum().sum() == 0
        assert len(result) < len(sample_df_with_nan)

    def test_clean_data_no_nan(self, sample_df):
        """Test that clean_data preserves data without NaN."""
        result = clean_data(sample_df)
        assert len(result) == len(sample_df)


class TestSplitData:
    def test_split_data_proportions(self, sample_df, split_params):
        """Test that split_data correctly splits the data."""
        X_train, X_test, y_train, y_test = split_data(sample_df, split_params)
        
        test_ratio = len(X_test) / len(sample_df)
        assert 0.1 <= test_ratio <= 0.3  # Allow some variation
        assert len(X_train) + len(X_test) == len(sample_df)
        assert len(y_train) + len(y_test) == len(sample_df)

    def test_split_data_removes_target(self, sample_df, split_params):
        """Test that X doesn't contain the target column."""
        X_train, X_test, y_train, y_test = split_data(sample_df, split_params)
        assert 'income' not in X_train.columns
        assert 'income' not in X_test.columns


class TestFitPreprocessor:
    def test_fit_preprocessor_returns_columntransformer(self, sample_df, feature_params, split_params):
        """Test that fit_preprocessor returns a ColumnTransformer."""
        X_train, _, _, _ = split_data(sample_df, split_params)
        preprocessor = fit_preprocessor(X_train, feature_params)
        assert isinstance(preprocessor, ColumnTransformer)

    def test_fit_preprocessor_has_transformers(self, sample_df, feature_params, split_params):
        """Test that preprocessor has the expected transformers."""
        X_train, _, _, _ = split_data(sample_df, split_params)
        preprocessor = fit_preprocessor(X_train, feature_params)
        assert len(preprocessor.transformers_) == 2


class TestEncodeLabel:
    def test_encode_label_converts_gt_50k_to_1(self, sample_df, feature_params):
        """Test that '>50K' is encoded as 1."""
        y = sample_df[['income']]
        encoded = _encode_label(y, feature_params)
        assert (encoded[encoded['income'] == 1].index == y[y['income'] == '>50K'].index).all()

    def test_encode_label_converts_others_to_0(self, sample_df, feature_params):
        """Test that '<=50K' is encoded as 0."""
        y = sample_df[['income']]
        encoded = _encode_label(y, feature_params)
        assert (encoded[encoded['income'] == 0].index == y[y['income'] == '<=50K'].index).all()


class TestApplyPreprocessor:
    def test_apply_preprocessor_output_shape(self, sample_df, feature_params, split_params):
        """Test that apply_preprocessor returns correct output."""
        X_train, X_test, y_train, y_test = split_data(sample_df, split_params)
        preprocessor = fit_preprocessor(X_train, feature_params)
        
        X_transformed, y_transformed = apply_preprocessor(X_train, y_train, preprocessor, feature_params)
        assert X_transformed.shape[0] == X_train.shape[0]
        assert y_transformed.shape[0] == y_train.shape[0]

    def test_apply_preprocessor_encodes_target(self, sample_df, feature_params, split_params):
        """Test that target is properly encoded."""
        X_train, X_test, y_train, y_test = split_data(sample_df, split_params)
        preprocessor = fit_preprocessor(X_train, feature_params)
        
        X_transformed, y_transformed = apply_preprocessor(X_train, y_train, preprocessor, feature_params)
        assert set(y_transformed['income'].unique()).issubset({0, 1})


class TestTrainModel:
    def test_train_model_returns_classifier(self, sample_df, feature_params, split_params, model_params):
        """Test that train_model returns a fitted classifier."""
        X_train, _, y_train, _ = split_data(sample_df, split_params)
        preprocessor = fit_preprocessor(X_train, feature_params)
        X_train, y_train = apply_preprocessor(X_train, y_train, preprocessor, feature_params)
        
        model = train_model(X_train, y_train.values.ravel(), model_params)
        assert isinstance(model, RandomForestClassifier)
        assert hasattr(model, 'predict')

    def test_train_model_is_fitted(self, sample_df, feature_params, split_params, model_params):
        """Test that the returned model is fitted."""
        X_train, _, y_train, _ = split_data(sample_df, split_params)
        preprocessor = fit_preprocessor(X_train, feature_params)
        X_train, y_train = apply_preprocessor(X_train, y_train, preprocessor, feature_params)
        
        model = train_model(X_train, y_train.values.ravel(), model_params)
        assert model.n_features_in_ == X_train.shape[1]


class TestPredict:
    def test_predict_returns_binary_labels(self, sample_df, feature_params, split_params, model_params, model_configs):
        """Test that _predict returns binary labels."""
        X_train, X_test, y_train, y_test = split_data(sample_df, split_params)
        preprocessor = fit_preprocessor(X_train, feature_params)
        X_train, y_train = apply_preprocessor(X_train, y_train, preprocessor, feature_params)
        X_test, y_test = apply_preprocessor(X_test, y_test, preprocessor, feature_params)
        
        model = train_model(X_train, y_train.values.ravel(), model_params)
        predictions = _predict(model, model_configs, X_test)
        
        assert set(predictions).issubset({0, 1})
        assert len(predictions) == len(X_test)

    def test_predict_proba(self, sample_df, feature_params, split_params, model_params, model_configs):
        """Test that _predict can return probabilities."""
        X_train, X_test, y_train, y_test = split_data(sample_df, split_params)
        preprocessor = fit_preprocessor(X_train, feature_params)
        X_train, y_train = apply_preprocessor(X_train, y_train, preprocessor, feature_params)
        X_test, y_test = apply_preprocessor(X_test, y_test, preprocessor, feature_params)
        
        model = train_model(X_train, y_train.values.ravel(), model_params)
        probabilities = _predict_proba(model, X_test)[:, 1]

        assert len(probabilities) == len(X_test)
        assert np.all((probabilities >= 0) & (probabilities <= 1))


class TestGenMetrics:
    def test_gen_metrics_returns_dict(self, sample_df, feature_params, split_params, model_params, model_configs):
        """Test that _gen_metrics returns a dictionary with expected keys."""
        X_train, X_test, y_train, y_test = split_data(sample_df, split_params)
        preprocessor = fit_preprocessor(X_train, feature_params)
        X_train, y_train = apply_preprocessor(X_train, y_train, preprocessor, feature_params)
        X_test, y_test = apply_preprocessor(X_test, y_test, preprocessor, feature_params)
        
        model = train_model(X_train, y_train.values.ravel(), model_params)
        metrics = _gen_metrics(model, model_configs, X_test, y_test.values.ravel())
        
        assert isinstance(metrics, dict)
        assert 'f1_score' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics

    def test_gen_metrics_values_in_range(self, sample_df, feature_params, split_params, model_params, model_configs):
        """Test that metric values are in valid ranges."""
        X_train, X_test, y_train, y_test = split_data(sample_df, split_params)
        preprocessor = fit_preprocessor(X_train, feature_params)
        X_train, y_train = apply_preprocessor(X_train, y_train, preprocessor, feature_params)
        X_test, y_test = apply_preprocessor(X_test, y_test, preprocessor, feature_params)
        
        model = train_model(X_train, y_train.values.ravel(), model_params)
        metrics = _gen_metrics(model, model_configs, X_test, y_test.values.ravel())
        
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1


class TestEvaluateModel:
    @patch('nodes._store_pr_curve')
    @patch('nodes._store_confusion_matrix')
    def test_evaluate_model_returns_dict(self, mock_cm, mock_pr, sample_df, feature_params, split_params, model_params, model_configs):
        """Test that evaluate_model returns a dictionary."""
        X_train, X_test, y_train, y_test = split_data(sample_df, split_params)
        preprocessor = fit_preprocessor(X_train, feature_params)
        X_train, y_train = apply_preprocessor(X_train, y_train, preprocessor, feature_params)
        X_test, y_test = apply_preprocessor(X_test, y_test, preprocessor, feature_params)
        
        model = train_model(X_train, y_train.values.ravel(), model_params)
        result = evaluate_model(model, model_configs, X_test, y_test.values.ravel())
        
        assert isinstance(result, dict)
        assert 'f1_score' in result