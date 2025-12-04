"""
Data loading and generation module for Streamlit MMM app.

Provides functions to load uploaded CSV data or generate synthetic
weekly marketing data with seasonality and Gaussian noise.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import streamlit as st


def validate_csv_schema(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate that DataFrame has required columns for MMM analysis.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_columns = ['date', 'Sales', 'TV_Spend', 'Radio_Spend', 'Digital_Spend']
    
    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check numeric columns
    numeric_columns = ['Sales', 'TV_Spend', 'Radio_Spend', 'Digital_Spend']
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Column '{col}' must be numeric"
    
    # Check for null values
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        null_cols = null_counts[null_counts > 0].index.tolist()
        return False, f"Null values found in columns: {', '.join(null_cols)}"
    
    return True, ""


def load_uploaded_csv(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load and parse uploaded CSV file with date parsing.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Parsed DataFrame or None if validation fails
    """
    try:
        # Read CSV with date parsing
        df = pd.read_csv(uploaded_file, parse_dates=['date'])
        
        # Validate schema
        is_valid, error_msg = validate_csv_schema(df)
        if not is_valid:
            st.error(f"❌ CSV validation failed: {error_msg}")
            return None
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        st.success(f"✅ Successfully loaded {len(df)} rows of data")
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading CSV: {str(e)}")
        return None


def generate_synthetic_data(
    num_weeks: int = 52,
    start_date: str = "2024-01-01",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic weekly marketing data with seasonality and noise.
    
    Creates realistic marketing spend and sales data with:
    - Weekly seasonality (52 weeks)
    - Annual trend component
    - Gaussian noise for variability
    - Realistic spend-to-sales relationships
    
    Args:
        num_weeks: Number of weekly observations (default: 52)
        start_date: Start date in YYYY-MM-DD format
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: date, Sales, TV_Spend, Radio_Spend, Digital_Spend
    """
    np.random.seed(seed)
    
    # Generate weekly dates
    start = pd.to_datetime(start_date)
    dates = [start + timedelta(weeks=i) for i in range(num_weeks)]
    
    # Time index for seasonality (0 to 2π over the year)
    t = np.linspace(0, 2 * np.pi, num_weeks)
    
    # Seasonal component (peaks in Q4 for holidays)
    seasonality = 1 + 0.3 * np.sin(t - np.pi/2)  # Peak around week 39-52
    
    # Trend component (slight growth over time)
    trend = 1 + 0.2 * (np.arange(num_weeks) / num_weeks)
    
    # Base marketing spend levels
    base_tv = 12000
    base_radio = 6000
    base_digital = 10000
    
    # Generate marketing spend with seasonality and noise
    tv_spend = base_tv * seasonality * trend + np.random.normal(0, 1500, num_weeks)
    radio_spend = base_radio * seasonality * trend + np.random.normal(0, 800, num_weeks)
    digital_spend = base_digital * seasonality * trend + np.random.normal(0, 1200, num_weeks)
    
    # Ensure no negative spend
    tv_spend = np.maximum(tv_spend, 500)
    radio_spend = np.maximum(radio_spend, 300)
    digital_spend = np.maximum(digital_spend, 500)
    
    # Generate sales with contribution from each channel
    # ROI: TV=3.5x, Radio=2.5x, Digital=4.0x
    tv_contribution = tv_spend * 3.5
    radio_contribution = radio_spend * 2.5
    digital_contribution = digital_spend * 4.0
    
    # Base sales (organic) with seasonality
    base_sales = 50000 * seasonality * trend
    
    # Total sales with noise
    sales = (
        base_sales + 
        tv_contribution + 
        radio_contribution + 
        digital_contribution +
        np.random.normal(0, 8000, num_weeks)
    )
    
    # Ensure positive sales
    sales = np.maximum(sales, 10000)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'Sales': np.round(sales, 2),
        'TV_Spend': np.round(tv_spend, 2),
        'Radio_Spend': np.round(radio_spend, 2),
        'Digital_Spend': np.round(digital_spend, 2)
    })
    
    return df


def get_data(uploaded_file=None) -> pd.DataFrame:
    """
    Main data loading function - returns uploaded data or synthetic fallback.
    
    Args:
        uploaded_file: Optional Streamlit UploadedFile object
        
    Returns:
        DataFrame with marketing data
    """
    if uploaded_file is not None:
        df = load_uploaded_csv(uploaded_file)
        if df is not None:
            return df
    
    # Fallback to synthetic data
    st.info("📊 No file uploaded. Using synthetic demo data (52 weeks).")
    return generate_synthetic_data()


# Example usage for testing
if __name__ == "__main__":
    # Test synthetic data generation
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nBasic statistics:")
    print(df.describe())
    
    # Validate schema
    is_valid, msg = validate_csv_schema(df)
    print(f"\nSchema validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
    if not is_valid:
        print(f"Error: {msg}")
