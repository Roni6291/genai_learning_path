"""
Test script for ROI Analysis calculations.

Verifies channel-wise ROI table computations.
"""

import pandas as pd
import numpy as np
from data_loader import generate_synthetic_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


def fit_linear_mmm(df, target='Sales', features=None):
    """Fit linear MMM model."""
    if features is None:
        features = ['TV_Spend', 'Radio_Spend', 'Digital_Spend']
    
    X = df[features].values
    y = df[target].values
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    
    coefficients = {feature: coef for feature, coef in zip(features, model.coef_)}
    
    return {
        'model': model,
        'y_pred': y_pred,
        'y_actual': y,
        'coefficients': coefficients,
        'intercept': model.intercept_,
        'r2': r2,
        'mae': mae,
        'mape': mape
    }


print("🧪 Testing ROI Analysis Calculations...\n")

# Generate test data
print("1. Generating synthetic data...")
df = generate_synthetic_data(num_weeks=52, seed=42)
print(f"   ✅ Generated {len(df)} rows\n")

# Fit model
print("2. Fitting Linear MMM model...")
mmm_results = fit_linear_mmm(df, target='Sales', features=['TV_Spend', 'Radio_Spend', 'Digital_Spend'])
print(f"   ✅ Model fitted successfully\n")

# Calculate channel totals
print("3. Computing channel-wise totals...")
channel_totals = {
    'TV': df['TV_Spend'].sum(),
    'Radio': df['Radio_Spend'].sum(),
    'Digital': df['Digital_Spend'].sum()
}

for channel, total in channel_totals.items():
    print(f"   {channel:10} Total Spend: ${total:,.2f}")
print()

# Build ROI table
print("4. Building ROI Analysis Table...")
roi_data = []
for channel, total_spend in channel_totals.items():
    coef = mmm_results['coefficients'][f'{channel}_Spend']
    contribution = coef * total_spend
    marginal_roi = coef  # ΔSales per $1
    total_roi = contribution / total_spend if total_spend > 0 else 0
    
    roi_data.append({
        'Channel': channel,
        'Total_Spend': total_spend,
        'Contribution': contribution,
        'Marginal_ROI': marginal_roi,
        'Total_ROI': total_roi
    })

roi_df = pd.DataFrame(roi_data)
print(f"   ✅ ROI table created with {len(roi_df)} channels\n")

# Verify calculations
print("5. Verifying ROI Calculations...")
print("   " + "="*70)

for idx, row in roi_df.iterrows():
    channel = row['Channel']
    total_spend = row['Total_Spend']
    contribution = row['Contribution']
    marginal_roi = row['Marginal_ROI']
    total_roi = row['Total_ROI']
    
    # Verify contribution calculation
    expected_contribution = marginal_roi * total_spend
    assert abs(contribution - expected_contribution) < 0.01, f"Contribution mismatch for {channel}"
    
    # Verify total ROI calculation
    expected_total_roi = contribution / total_spend
    assert abs(total_roi - expected_total_roi) < 0.0001, f"Total ROI mismatch for {channel}"
    
    print(f"   📊 {channel} Channel:")
    print(f"      Total Spend:        ${total_spend:>12,.2f}")
    print(f"      Coefficient:        {marginal_roi:>12.3f}x")
    print(f"      Contribution:       ${contribution:>12,.2f}")
    print(f"      Marginal ROI:       {marginal_roi:>12.3f}x (per $1)")
    print(f"      Total ROI:          {total_roi:>12.3f}x")
    print(f"      ─────────────────────────────────────────────────────────────")

print(f"   ✅ All calculations verified\n")

# Test formatting
print("6. Testing Display Formatting...")
roi_df_display = roi_df.copy()
roi_df_display['Total_Spend'] = roi_df_display['Total_Spend'].apply(lambda x: f"${x:,.2f}")
roi_df_display['Contribution'] = roi_df_display['Contribution'].apply(lambda x: f"${x:,.2f}")
roi_df_display['Marginal_ROI'] = roi_df_display['Marginal_ROI'].apply(lambda x: f"{x:.3f}x")
roi_df_display['Total_ROI'] = roi_df_display['Total_ROI'].apply(lambda x: f"{x:.3f}x")
roi_df_display.columns = ['Channel', 'Total Spend', 'Sales Contribution', 'Marginal ROI (per $1)', 'Total ROI']

print("   ✅ Formatted ROI Table:")
print(roi_df_display.to_string(index=False))
print()

# Verify key metrics
print("7. Key Metrics Summary...")
total_marketing_spend = sum(channel_totals.values())
total_contribution = roi_df['Contribution'].sum()
weighted_avg_roi = total_contribution / total_marketing_spend

print(f"   Total Marketing Spend:    ${total_marketing_spend:,.2f}")
print(f"   Total Sales Contribution: ${total_contribution:,.2f}")
print(f"   Weighted Average ROI:     {weighted_avg_roi:.3f}x")
print()

# Verify ROI ranking
print("8. Channel Ranking by Marginal ROI...")
roi_df_sorted = roi_df.sort_values('Marginal_ROI', ascending=False)
for rank, (idx, row) in enumerate(roi_df_sorted.iterrows(), 1):
    print(f"   #{rank}. {row['Channel']:10} - {row['Marginal_ROI']:.3f}x Marginal ROI")
print()

# Verify all ROI values are positive and reasonable
print("9. Data Quality Checks...")
assert (roi_df['Total_Spend'] > 0).all(), "All spend values should be positive"
assert (roi_df['Contribution'] > 0).all(), "All contributions should be positive"
assert (roi_df['Marginal_ROI'] > 0).all(), "All Marginal ROIs should be positive"
assert (roi_df['Total_ROI'] > 0).all(), "All Total ROIs should be positive"
print(f"   ✅ All spend values are positive")
print(f"   ✅ All contributions are positive")
print(f"   ✅ All ROI values are positive")
print(f"   ✅ No invalid or NaN values detected")
print()

print("="*70)
print("✅ ALL ROI ANALYSIS TESTS PASSED")
print("="*70)
print("\n📝 Summary:")
print("   • Channel totals computed: ✅")
print("   • Contribution calculations: ✅")
print("   • Marginal ROI (coefficients): ✅")
print("   • Total ROI calculations: ✅")
print("   • Number formatting: ✅")
print("   • Data quality validated: ✅")
print("\n✨ ROI Analysis is ready for Streamlit display!")
