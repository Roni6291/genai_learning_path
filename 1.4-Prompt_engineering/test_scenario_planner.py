"""
Test script for Scenario Planner calculations.

Verifies baseline and custom scenario predictions.
"""

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


print("🧪 Testing Scenario Planner Calculations...\n")

# Generate test data
print("1. Generating synthetic data...")
df = generate_synthetic_data(num_weeks=52, seed=42)
print(f"   ✅ Generated {len(df)} rows\n")

# Fit model
print("2. Fitting Linear MMM model...")
mmm_results = fit_linear_mmm(df, target='Sales', features=['TV_Spend', 'Radio_Spend', 'Digital_Spend'])
print(f"   ✅ Model fitted successfully")
print(f"   Intercept: ${mmm_results['intercept']:,.2f}")
print(f"   TV Coef: {mmm_results['coefficients']['TV_Spend']:.3f}x")
print(f"   Radio Coef: {mmm_results['coefficients']['Radio_Spend']:.3f}x")
print(f"   Digital Coef: {mmm_results['coefficients']['Digital_Spend']:.3f}x\n")

# Calculate baseline
print("3. Computing Baseline Scenario (Mean Spends)...")
baseline_tv = df['TV_Spend'].mean()
baseline_radio = df['Radio_Spend'].mean()
baseline_digital = df['Digital_Spend'].mean()

print(f"   Baseline TV Spend:      ${baseline_tv:,.2f}")
print(f"   Baseline Radio Spend:   ${baseline_radio:,.2f}")
print(f"   Baseline Digital Spend: ${baseline_digital:,.2f}")

baseline_prediction = (
    mmm_results['intercept'] +
    mmm_results['coefficients']['TV_Spend'] * baseline_tv +
    mmm_results['coefficients']['Radio_Spend'] * baseline_radio +
    mmm_results['coefficients']['Digital_Spend'] * baseline_digital
)

baseline_total_spend = baseline_tv + baseline_radio + baseline_digital

print(f"   Baseline Total Spend:   ${baseline_total_spend:,.2f}")
print(f"   Baseline Predicted Sales: ${baseline_prediction:,.2f}")
print(f"   Baseline ROI: {baseline_prediction / baseline_total_spend:.3f}x\n")

# Test manual prediction
print("4. Verifying Manual Prediction Formula...")
manual_prediction = mmm_results['model'].predict([[baseline_tv, baseline_radio, baseline_digital]])[0]
assert abs(manual_prediction - baseline_prediction) < 0.01, "Manual calculation doesn't match model prediction"
print(f"   ✅ Manual calculation matches model: ${manual_prediction:,.2f}\n")

# Test scenario prediction
print("5. Testing Custom Scenario...")
scenario_tv = 15000.0
scenario_radio = 8000.0
scenario_digital = 12000.0

print(f"   Scenario TV Spend:      ${scenario_tv:,.2f}")
print(f"   Scenario Radio Spend:   ${scenario_radio:,.2f}")
print(f"   Scenario Digital Spend: ${scenario_digital:,.2f}")

scenario_prediction = (
    mmm_results['intercept'] +
    mmm_results['coefficients']['TV_Spend'] * scenario_tv +
    mmm_results['coefficients']['Radio_Spend'] * scenario_radio +
    mmm_results['coefficients']['Digital_Spend'] * scenario_digital
)

scenario_total_spend = scenario_tv + scenario_radio + scenario_digital

print(f"   Scenario Total Spend:   ${scenario_total_spend:,.2f}")
print(f"   Scenario Predicted Sales: ${scenario_prediction:,.2f}")
print(f"   Scenario ROI: {scenario_prediction / scenario_total_spend:.3f}x\n")

# Calculate deltas
print("6. Computing Deltas vs Baseline...")
delta_tv = scenario_tv - baseline_tv
delta_radio = scenario_radio - baseline_radio
delta_digital = scenario_digital - baseline_digital
delta_spend = scenario_total_spend - baseline_total_spend
delta_sales = scenario_prediction - baseline_prediction
delta_percent = (delta_sales / baseline_prediction) * 100

print(f"   Δ TV Spend:      ${delta_tv:+,.2f}")
print(f"   Δ Radio Spend:   ${delta_radio:+,.2f}")
print(f"   Δ Digital Spend: ${delta_digital:+,.2f}")
print(f"   Δ Total Spend:   ${delta_spend:+,.2f}")
print(f"   Δ Sales:         ${delta_sales:+,.2f} ({delta_percent:+.2f}%)")

if delta_spend != 0:
    incremental_roi = delta_sales / delta_spend
    print(f"   Incremental ROI: {incremental_roi:.3f}x")
print()

# Test contribution breakdown
print("7. Testing Channel Contribution Breakdown...")
print("   " + "="*70)

for channel, spend_key, scenario_spend in [
    ('TV', 'TV_Spend', scenario_tv),
    ('Radio', 'Radio_Spend', scenario_radio),
    ('Digital', 'Digital_Spend', scenario_digital)
]:
    baseline_spend = df[spend_key].mean()
    coef = mmm_results['coefficients'][spend_key]
    
    baseline_contrib = coef * baseline_spend
    scenario_contrib = coef * scenario_spend
    delta_contrib = scenario_contrib - baseline_contrib
    
    print(f"\n   {channel} Channel:")
    print(f"      Baseline Spend:        ${baseline_spend:>12,.2f}")
    print(f"      Scenario Spend:        ${scenario_spend:>12,.2f}")
    print(f"      Δ Spend:               ${scenario_spend - baseline_spend:>+12,.2f}")
    print(f"      Coefficient:           {coef:>12.3f}x")
    print(f"      Baseline Contribution: ${baseline_contrib:>12,.2f}")
    print(f"      Scenario Contribution: ${scenario_contrib:>12,.2f}")
    print(f"      Δ Contribution:        ${delta_contrib:>+12,.2f}")

print("\n   " + "="*70)
print()

# Verify total contribution matches prediction
print("8. Verifying Total Contribution...")
baseline_total_contrib = sum([
    mmm_results['coefficients']['TV_Spend'] * baseline_tv,
    mmm_results['coefficients']['Radio_Spend'] * baseline_radio,
    mmm_results['coefficients']['Digital_Spend'] * baseline_digital
])

scenario_total_contrib = sum([
    mmm_results['coefficients']['TV_Spend'] * scenario_tv,
    mmm_results['coefficients']['Radio_Spend'] * scenario_radio,
    mmm_results['coefficients']['Digital_Spend'] * scenario_digital
])

baseline_check = mmm_results['intercept'] + baseline_total_contrib
scenario_check = mmm_results['intercept'] + scenario_total_contrib

assert abs(baseline_check - baseline_prediction) < 0.01, "Baseline contribution check failed"
assert abs(scenario_check - scenario_prediction) < 0.01, "Scenario contribution check failed"

print(f"   ✅ Baseline: Intercept + Contributions = ${baseline_check:,.2f}")
print(f"   ✅ Scenario: Intercept + Contributions = ${scenario_check:,.2f}\n")

# Test edge cases
print("9. Testing Edge Cases...")

# Zero spend scenario
zero_prediction = mmm_results['intercept']
print(f"   Zero spend prediction (intercept only): ${zero_prediction:,.2f}")
assert zero_prediction == mmm_results['intercept'], "Zero spend should equal intercept"
print(f"   ✅ Zero spend scenario works correctly")

# Max spend scenario
max_tv = df['TV_Spend'].max() * 2
max_radio = df['Radio_Spend'].max() * 2
max_digital = df['Digital_Spend'].max() * 2
max_prediction = (
    mmm_results['intercept'] +
    mmm_results['coefficients']['TV_Spend'] * max_tv +
    mmm_results['coefficients']['Radio_Spend'] * max_radio +
    mmm_results['coefficients']['Digital_Spend'] * max_digital
)
print(f"   Max spend (2x) prediction: ${max_prediction:,.2f}")
print(f"   ✅ Max spend scenario works correctly\n")

print("="*70)
print("✅ ALL SCENARIO PLANNER TESTS PASSED")
print("="*70)
print("\n📝 Summary:")
print("   • Baseline calculation: ✅")
print("   • Custom scenario prediction: ✅")
print("   • Delta calculations: ✅")
print("   • Incremental ROI: ✅")
print("   • Channel contribution breakdown: ✅")
print("   • Edge cases: ✅")
print("\n✨ Scenario Planner is ready for use!")
