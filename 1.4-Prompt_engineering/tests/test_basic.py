"""
Smoke tests for MMM app core functionality.
Tests data generation, model fitting, and scenario predictions.
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from data_loader import generate_synthetic_data
from app import fit_linear_mmm


class TestDataGeneration:
    """Test synthetic data generation."""
    
    def test_demo_data_shape(self):
        """Verify generated data has correct shape."""
        df = generate_synthetic_data(num_weeks=52, seed=42)
        
        assert df.shape[0] == 52, f"Expected 52 rows, got {df.shape[0]}"
        assert df.shape[1] == 5, f"Expected 5 columns, got {df.shape[1]}"
    
    def test_demo_data_columns(self):
        """Verify generated data has required columns."""
        df = generate_synthetic_data(num_weeks=52, seed=42)
        
        required_cols = {'date', 'Sales', 'TV_Spend', 'Radio_Spend', 'Digital_Spend'}
        actual_cols = set(df.columns)
        
        assert required_cols == actual_cols, \
            f"Column mismatch. Expected {required_cols}, got {actual_cols}"
    
    def test_demo_data_types(self):
        """Verify data types are correct."""
        df = generate_synthetic_data(num_weeks=52, seed=42)
        
        assert pd.api.types.is_datetime64_any_dtype(df['date']), \
            f"'date' should be datetime, got {df['date'].dtype}"
        
        numeric_cols = ['Sales', 'TV_Spend', 'Radio_Spend', 'Digital_Spend']
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df[col]), \
                f"'{col}' should be numeric, got {df[col].dtype}"
    
    def test_demo_data_no_nulls(self):
        """Verify no missing values in generated data."""
        df = generate_synthetic_data(num_weeks=52, seed=42)
        
        null_counts = df.isnull().sum()
        assert null_counts.sum() == 0, \
            f"Found null values: {null_counts[null_counts > 0].to_dict()}"
    
    def test_demo_data_positive_values(self):
        """Verify all numeric values are positive."""
        df = generate_synthetic_data(num_weeks=52, seed=42)
        
        numeric_cols = ['Sales', 'TV_Spend', 'Radio_Spend', 'Digital_Spend']
        for col in numeric_cols:
            min_val = df[col].min()
            assert min_val > 0, f"'{col}' has non-positive value: {min_val}"


class TestModelFitting:
    """Test MMM model fitting function."""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample data for testing."""
        return generate_synthetic_data(num_weeks=52, seed=42)
    
    def test_fitting_returns_dict(self, sample_data):
        """Verify fit_linear_mmm returns a dictionary."""
        features = ['TV_Spend', 'Radio_Spend', 'Digital_Spend']
        result = fit_linear_mmm(sample_data, target='Sales', features=features)
        
        assert isinstance(result, dict), \
            f"Expected dict, got {type(result)}"
    
    def test_fitting_returns_metrics(self, sample_data):
        """Verify fit_linear_mmm returns required metrics."""
        features = ['TV_Spend', 'Radio_Spend', 'Digital_Spend']
        result = fit_linear_mmm(sample_data, target='Sales', features=features)
        
        required_keys = {'r2', 'mae', 'mape'}
        actual_keys = set(result.keys())
        
        assert required_keys.issubset(actual_keys), \
            f"Missing metrics. Expected {required_keys}, got {actual_keys}"
        
        # Verify metrics are numeric
        assert isinstance(result['r2'], (int, float, np.number)), \
            f"r2 should be numeric, got {type(result['r2'])}"
        assert isinstance(result['mae'], (int, float, np.number)), \
            f"mae should be numeric, got {type(result['mae'])}"
        assert isinstance(result['mape'], (int, float, np.number)), \
            f"mape should be numeric, got {type(result['mape'])}"
    
    def test_fitting_returns_coefficients(self, sample_data):
        """Verify fit_linear_mmm returns coefficients."""
        features = ['TV_Spend', 'Radio_Spend', 'Digital_Spend']
        result = fit_linear_mmm(sample_data, target='Sales', features=features)
        
        assert 'coefficients' in result, "Missing 'coefficients' key"
        assert 'intercept' in result, "Missing 'intercept' key"
        
        coeffs = result['coefficients']
        assert isinstance(coeffs, dict), \
            f"Coefficients should be dict, got {type(coeffs)}"
        assert len(coeffs) == 3, \
            f"Expected 3 coefficients, got {len(coeffs)}"
        
        # Verify intercept is numeric
        assert isinstance(result['intercept'], (int, float, np.number)), \
            f"Intercept should be numeric, got {type(result['intercept'])}"
    
    def test_fitting_metric_ranges(self, sample_data):
        """Verify metrics are in reasonable ranges."""
        features = ['TV_Spend', 'Radio_Spend', 'Digital_Spend']
        result = fit_linear_mmm(sample_data, target='Sales', features=features)
        
        # R² should be between 0 and 1
        assert 0 <= result['r2'] <= 1, \
            f"R² should be in [0, 1], got {result['r2']}"
        
        # MAE should be positive
        assert result['mae'] > 0, \
            f"MAE should be positive, got {result['mae']}"
        
        # MAPE should be positive
        assert result['mape'] > 0, \
            f"MAPE should be positive, got {result['mape']}"
    
    def test_fitting_returns_predictions(self, sample_data):
        """Verify fit_linear_mmm returns predictions."""
        features = ['TV_Spend', 'Radio_Spend', 'Digital_Spend']
        result = fit_linear_mmm(sample_data, target='Sales', features=features)
        
        assert 'y_pred' in result, "Missing 'y_pred' key"
        assert 'y_actual' in result, "Missing 'y_actual' key"
        
        assert len(result['y_pred']) == len(sample_data), \
            f"Expected {len(sample_data)} predictions, got {len(result['y_pred'])}"
        assert len(result['y_actual']) == len(sample_data), \
            f"Expected {len(sample_data)} actuals, got {len(result['y_actual'])}"


class TestScenarioPlanner:
    """Test scenario planning predictions."""
    
    @pytest.fixture
    def fitted_model(self):
        """Fixture providing a fitted model."""
        df = generate_synthetic_data(num_weeks=52, seed=42)
        features = ['TV_Spend', 'Radio_Spend', 'Digital_Spend']
        return fit_linear_mmm(df, target='Sales', features=features)
    
    def test_scenario_prediction_returns_float(self, fitted_model):
        """Verify scenario prediction returns a float."""
        model = fitted_model['model']
        
        # Test scenario with typical spend values
        scenario_spend = np.array([[15000, 8000, 12000]])  # TV, Radio, Digital
        prediction = model.predict(scenario_spend)[0]
        
        assert isinstance(prediction, (float, np.floating)), \
            f"Prediction should be float, got {type(prediction)}"
    
    def test_scenario_prediction_positive(self, fitted_model):
        """Verify scenario prediction is positive."""
        model = fitted_model['model']
        
        scenario_spend = np.array([[15000, 8000, 12000]])
        prediction = model.predict(scenario_spend)[0]
        
        assert prediction > 0, \
            f"Prediction should be positive, got {prediction}"
    
    def test_scenario_zero_spend(self, fitted_model):
        """Verify prediction with zero spend returns intercept."""
        model = fitted_model['model']
        intercept = fitted_model['intercept']
        
        zero_spend = np.array([[0, 0, 0]])
        prediction = model.predict(zero_spend)[0]
        
        # Should be close to intercept (allowing for floating point precision)
        assert abs(prediction - intercept) < 0.01, \
            f"Zero spend should return intercept {intercept}, got {prediction}"
    
    def test_scenario_higher_spend_higher_sales(self, fitted_model):
        """Verify higher spend predicts higher sales."""
        model = fitted_model['model']
        
        low_spend = np.array([[5000, 3000, 4000]])
        high_spend = np.array([[20000, 12000, 16000]])
        
        pred_low = model.predict(low_spend)[0]
        pred_high = model.predict(high_spend)[0]
        
        assert pred_high > pred_low, \
            f"Higher spend should predict higher sales. Low: {pred_low}, High: {pred_high}"
    
    def test_scenario_prediction_shape(self, fitted_model):
        """Verify prediction array shape."""
        model = fitted_model['model']
        
        # Single scenario
        single = np.array([[15000, 8000, 12000]])
        pred_single = model.predict(single)
        assert pred_single.shape == (1,), \
            f"Single prediction shape should be (1,), got {pred_single.shape}"
        
        # Multiple scenarios
        multiple = np.array([
            [15000, 8000, 12000],
            [10000, 5000, 8000],
            [20000, 10000, 15000]
        ])
        pred_multiple = model.predict(multiple)
        assert pred_multiple.shape == (3,), \
            f"Multiple prediction shape should be (3,), got {pred_multiple.shape}"


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])
