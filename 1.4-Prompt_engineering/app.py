"""
Streamlit Marketing Mix Modeling (MMM) Application

Main application file for interactive MMM analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from data_loader import get_data


def fit_linear_mmm(df, target='Sales', features=None):
    """
    Fit a linear Marketing Mix Model using scikit-learn LinearRegression.
    
    Args:
        df: DataFrame with marketing data
        target: Target variable column name (default: 'Sales')
        features: List of feature column names (default: ['TV_Spend', 'Radio_Spend', 'Digital_Spend'])
    
    Returns:
        dict containing:
            - model: Fitted LinearRegression model
            - y_pred: Predicted values
            - coefficients: Dictionary of feature coefficients
            - intercept: Model intercept
            - r2: R-squared score
            - mae: Mean Absolute Error
            - mape: Mean Absolute Percentage Error
    """
    if features is None:
        features = ['TV_Spend', 'Radio_Spend', 'Digital_Spend']
    
    # Prepare data
    X = df[features].values
    y = df[target].values
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predictions
    y_pred = model.predict(X)
    
    # Compute metrics
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    # MAPE calculation (avoid division by zero)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    
    # Coefficients
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


# Page configuration
st.set_page_config(
    page_title="Marketing Mix Modeling",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📊 Marketing Mix Modeling (MMM) App")
st.markdown("""
Analyze the effectiveness of your marketing channels on sales performance.
Upload your data or use synthetic demo data to get started.
""")

# Sidebar for data upload
with st.sidebar:
    st.header("📁 Data Upload")
    st.markdown("Upload a CSV file with your marketing data.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="CSV must contain: date, Sales, TV_Spend, Radio_Spend, Digital_Spend"
    )
    
    st.markdown("---")
    st.markdown("### Expected Schema")
    st.code("""
date           (datetime)
Sales          (numeric)
TV_Spend       (numeric)
Radio_Spend    (numeric)
Digital_Spend  (numeric)
    """, language="text")

# Load data
df = get_data(uploaded_file)

# Display data overview
st.header("📋 Data Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Observations", len(df))
with col2:
    st.metric("Date Range", f"{(df['date'].max() - df['date'].min()).days} days")
with col3:
    st.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
with col4:
    st.metric("Total Spend", f"${(df['TV_Spend'] + df['Radio_Spend'] + df['Digital_Spend']).sum():,.0f}")

# Data preview
with st.expander("🔍 View Raw Data", expanded=False):
    # Convert date to string to avoid Arrow serialization issues
    df_display = df.copy()
    df_display['date'] = df_display['date'].dt.strftime('%Y-%m-%d')
    st.dataframe(df_display, width='stretch', height=300)
    
    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name="mmm_data.csv",
        mime="text/csv"
    )

# Summary statistics
st.header("📈 Summary Statistics")
# Exclude date column to avoid Arrow serialization issues
numeric_cols = ['Sales', 'TV_Spend', 'Radio_Spend', 'Digital_Spend']
st.dataframe(df[numeric_cols].describe().round(2), width='stretch')

# Visualizations
st.header("📊 Visualizations")

# 1. Sales over time line chart (Streamlit built-in)
st.subheader("1️⃣ Sales Trend Over Time")
st.caption("Line chart showing sales performance over the time period")
sales_trend_df = df[['date', 'Sales']].set_index('date')
st.line_chart(sales_trend_df, height=400)

# Original time series plot with all channels
st.subheader("Sales and Marketing Spend Over Time")
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['Sales'],
    name='Sales',
    line=dict(color='#1f77b4', width=3),
    yaxis='y1'
))

fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['TV_Spend'],
    name='TV Spend',
    line=dict(color='#ff7f0e', dash='dash'),
    yaxis='y2'
))

fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['Radio_Spend'],
    name='Radio Spend',
    line=dict(color='#2ca02c', dash='dash'),
    yaxis='y2'
))

fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['Digital_Spend'],
    name='Digital Spend',
    line=dict(color='#d62728', dash='dash'),
    yaxis='y2'
))

fig.update_layout(
    xaxis=dict(title='Date'),
    yaxis=dict(
        title='Sales ($)',
        title_font=dict(color='#1f77b4'),
        tickfont=dict(color='#1f77b4')
    ),
    yaxis2=dict(
        title='Marketing Spend ($)',
        title_font=dict(color='#ff7f0e'),
        tickfont=dict(color='#ff7f0e'),
        overlaying='y',
        side='right'
    ),
    hovermode='x unified',
    height=500
)

st.plotly_chart(fig, width='stretch')

# Spending distribution
st.subheader("Marketing Spend Distribution by Channel")

col1, col2 = st.columns(2)

with col1:
    # Pie chart
    total_spend = {
        'TV': df['TV_Spend'].sum(),
        'Radio': df['Radio_Spend'].sum(),
        'Digital': df['Digital_Spend'].sum()
    }
    
    fig_pie = px.pie(
        values=list(total_spend.values()),
        names=list(total_spend.keys()),
        title='Total Spend by Channel',
        color_discrete_sequence=['#ff7f0e', '#2ca02c', '#d62728']
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, width='stretch')

with col2:
    # Bar chart
    fig_bar = go.Figure(data=[
        go.Bar(name='TV', x=['Total Spend'], y=[total_spend['TV']], marker_color='#ff7f0e'),
        go.Bar(name='Radio', x=['Total Spend'], y=[total_spend['Radio']], marker_color='#2ca02c'),
        go.Bar(name='Digital', x=['Total Spend'], y=[total_spend['Digital']], marker_color='#d62728')
    ])
    fig_bar.update_layout(
        title='Spend Comparison',
        yaxis_title='Amount ($)',
        barmode='group',
        height=400
    )
    st.plotly_chart(fig_bar, width='stretch')

# 2. Scatter plots: Spend vs Sales for each channel (Streamlit built-in)
st.header("🔗 Marketing Spend vs Sales Analysis")
st.subheader("2️⃣ Channel Performance: Spend vs Sales Relationships")
st.caption("Scatter plots showing the relationship between marketing spend and sales for each channel")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**TV Spend vs Sales**")
    st.scatter_chart(df, x='TV_Spend', y='Sales', color='#ff7f0e', height=300)
    corr_tv_pearson = df['TV_Spend'].corr(df['Sales'])
    corr_tv_spearman, _ = spearmanr(df['TV_Spend'], df['Sales'])
    st.caption(f"Pearson: {corr_tv_pearson:.3f} | Spearman: {corr_tv_spearman:.3f}")

with col2:
    st.markdown("**Radio Spend vs Sales**")
    st.scatter_chart(df, x='Radio_Spend', y='Sales', color='#2ca02c', height=300)
    corr_radio_pearson = df['Radio_Spend'].corr(df['Sales'])
    corr_radio_spearman, _ = spearmanr(df['Radio_Spend'], df['Sales'])
    st.caption(f"Pearson: {corr_radio_pearson:.3f} | Spearman: {corr_radio_spearman:.3f}")

with col3:
    st.markdown("**Digital Spend vs Sales**")
    st.scatter_chart(df, x='Digital_Spend', y='Sales', color='#d62728', height=300)
    corr_digital_pearson = df['Digital_Spend'].corr(df['Sales'])
    corr_digital_spearman, _ = spearmanr(df['Digital_Spend'], df['Sales'])
    st.caption(f"Pearson: {corr_digital_pearson:.3f} | Spearman: {corr_digital_spearman:.3f}")

# 3. Spearman correlation matrix
st.subheader("3️⃣ Spearman Correlation Matrix")
st.caption("Non-parametric correlation coefficients between sales and all marketing channels (Spearman's rank correlation)")

# Calculate Spearman correlation matrix
corr_columns = ['Sales', 'TV_Spend', 'Radio_Spend', 'Digital_Spend']
corr_df = df[corr_columns].copy()

# Compute Spearman correlation matrix
spearman_corr = corr_df.corr(method='spearman')

# Display as dataframe with styling
st.dataframe(
    spearman_corr.style.background_gradient(cmap='RdYlGn', axis=None, vmin=-1, vmax=1).format("{:.3f}"),
    width='stretch'
)

# Additional insights
st.caption("""
**Interpretation:** Values range from -1 to +1. Values closer to +1 indicate strong positive correlation, 
values closer to -1 indicate strong negative correlation, and values near 0 indicate weak or no correlation.
Spearman correlation measures monotonic relationships and is robust to outliers.
""")

# Original detailed correlation analysis (Plotly with trendlines)
st.subheader("Detailed Correlation Analysis with Trendlines")
st.caption("Interactive scatter plots with linear regression trendlines for deeper insights")

col1, col2, col3 = st.columns(3)

with col1:
    fig_tv = px.scatter(
        df,
        x='TV_Spend',
        y='Sales',
        title='TV Spend vs Sales',
        trendline='ols',
        color_discrete_sequence=['#ff7f0e']
    )
    fig_tv.update_layout(height=350)
    st.plotly_chart(fig_tv, width='stretch')
    
    corr_tv = df['TV_Spend'].corr(df['Sales'])
    st.metric("Pearson Correlation", f"{corr_tv:.3f}")

with col2:
    fig_radio = px.scatter(
        df,
        x='Radio_Spend',
        y='Sales',
        title='Radio Spend vs Sales',
        trendline='ols',
        color_discrete_sequence=['#2ca02c']
    )
    fig_radio.update_layout(height=350)
    st.plotly_chart(fig_radio, width='stretch')
    
    corr_radio = df['Radio_Spend'].corr(df['Sales'])
    st.metric("Pearson Correlation", f"{corr_radio:.3f}")

with col3:
    fig_digital = px.scatter(
        df,
        x='Digital_Spend',
        y='Sales',
        title='Digital Spend vs Sales',
        trendline='ols',
        color_discrete_sequence=['#d62728']
    )
    fig_digital.update_layout(height=350)
    st.plotly_chart(fig_digital, width='stretch')
    
    corr_digital = df['Digital_Spend'].corr(df['Sales'])
    st.metric("Pearson Correlation", f"{corr_digital:.3f}")

# Linear Marketing Mix Model
st.header("📐 Linear Marketing Mix Model")
st.subheader("4️⃣ Model Performance & Predictions")
st.caption("Linear regression model to predict sales based on marketing spend across all channels")

# Fit the model
mmm_results = fit_linear_mmm(df, target='Sales', features=['TV_Spend', 'Radio_Spend', 'Digital_Spend'])

# Display metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="R² Score",
        value=f"{mmm_results['r2']:.4f}",
        help="Coefficient of determination - proportion of variance explained by the model (0-1, higher is better)"
    )

with col2:
    st.metric(
        label="MAE (Mean Absolute Error)",
        value=f"${mmm_results['mae']:,.2f}",
        help="Average absolute difference between actual and predicted sales"
    )

with col3:
    st.metric(
        label="MAPE (Mean Absolute % Error)",
        value=f"{mmm_results['mape']:.2f}%",
        help="Average percentage error - measures prediction accuracy"
    )

# Model coefficients
st.subheader("Model Coefficients (ROI Multipliers)")
st.caption("How much sales increase for each dollar spent on each channel")

coef_col1, coef_col2, coef_col3, coef_col4 = st.columns(4)

with coef_col1:
    st.metric("Intercept (Base Sales)", f"${mmm_results['intercept']:,.2f}")

with coef_col2:
    st.metric("TV Coefficient", f"{mmm_results['coefficients']['TV_Spend']:.3f}x")

with coef_col3:
    st.metric("Radio Coefficient", f"{mmm_results['coefficients']['Radio_Spend']:.3f}x")

with coef_col4:
    st.metric("Digital Coefficient", f"{mmm_results['coefficients']['Digital_Spend']:.3f}x")

# Actual vs Predicted scatter plot
st.subheader("Actual vs Predicted Sales")
st.caption("Scatter plot comparing actual sales with model predictions - points closer to the diagonal line indicate better predictions")

# Create scatter plot with perfect prediction line
fig_pred = go.Figure()

# Add actual vs predicted scatter
fig_pred.add_trace(go.Scatter(
    x=mmm_results['y_actual'],
    y=mmm_results['y_pred'],
    mode='markers',
    name='Predictions',
    marker=dict(
        color='#1f77b4',
        size=8,
        opacity=0.6
    ),
    text=[f"Actual: ${actual:,.0f}<br>Predicted: ${pred:,.0f}" 
          for actual, pred in zip(mmm_results['y_actual'], mmm_results['y_pred'])],
    hovertemplate='%{text}<extra></extra>'
))

# Add perfect prediction line (y=x)
min_val = min(mmm_results['y_actual'].min(), mmm_results['y_pred'].min())
max_val = max(mmm_results['y_actual'].max(), mmm_results['y_pred'].max())

fig_pred.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode='lines',
    name='Perfect Prediction',
    line=dict(color='red', dash='dash', width=2)
))

fig_pred.update_layout(
    xaxis_title='Actual Sales ($)',
    yaxis_title='Predicted Sales ($)',
    height=500,
    hovermode='closest',
    showlegend=True
)

st.plotly_chart(fig_pred, width='stretch')

# Model equation
st.subheader("Model Equation")
equation = f"""
**Sales = ${mmm_results['intercept']:,.2f}** + 
**{mmm_results['coefficients']['TV_Spend']:.3f}** × TV_Spend + 
**{mmm_results['coefficients']['Radio_Spend']:.3f}** × Radio_Spend + 
**{mmm_results['coefficients']['Digital_Spend']:.3f}** × Digital_Spend
"""
st.markdown(equation)
st.caption("This equation shows how each marketing channel contributes to total sales")

# Channel-wise ROI Analysis
st.subheader("5️⃣ Channel-wise ROI Analysis")
st.caption("Detailed return on investment analysis for each marketing channel")

# Calculate channel totals
channel_totals = {
    'TV': df['TV_Spend'].sum(),
    'Radio': df['Radio_Spend'].sum(),
    'Digital': df['Digital_Spend'].sum()
}

# Build ROI table
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

# Coefficient bar chart
st.markdown("**Marketing Channel Coefficients (Marginal ROI)**")
st.caption("Each coefficient represents the increase in sales for every $1 spent on that channel")

fig_coef = go.Figure(data=[
    go.Bar(
        x=roi_df['Channel'],
        y=roi_df['Marginal_ROI'],
        marker_color=['#ff7f0e', '#2ca02c', '#d62728'],
        text=roi_df['Marginal_ROI'].round(3),
        textposition='outside',
        texttemplate='%{text}x'
    )
])

fig_coef.update_layout(
    xaxis_title='Marketing Channel',
    yaxis_title='Marginal ROI (Sales per $1 Spent)',
    height=400,
    showlegend=False
)

st.plotly_chart(fig_coef, width='stretch')

# ROI Table
st.markdown("**Comprehensive ROI Table**")
st.caption("Detailed breakdown of spending, contribution, and return on investment by channel")

# Format the dataframe for display
roi_df_display = roi_df.copy()
roi_df_display['Total_Spend'] = roi_df_display['Total_Spend'].apply(lambda x: f"${x:,.2f}")
roi_df_display['Contribution'] = roi_df_display['Contribution'].apply(lambda x: f"${x:,.2f}")
roi_df_display['Marginal_ROI'] = roi_df_display['Marginal_ROI'].apply(lambda x: f"{x:.3f}x")
roi_df_display['Total_ROI'] = roi_df_display['Total_ROI'].apply(lambda x: f"{x:.3f}x")

# Rename columns for better readability
roi_df_display.columns = ['Channel', 'Total Spend', 'Sales Contribution', 'Marginal ROI (per $1)', 'Total ROI']

st.dataframe(roi_df_display, width='stretch', hide_index=True)

# Interpretation note
st.info("""
**📊 How to Interpret This Table:**

- **Total Spend**: Total amount invested in each marketing channel during the period
- **Sales Contribution**: Total sales attributed to this channel (Coefficient × Total Spend)
- **Marginal ROI (per $1)**: Additional sales generated for every $1 spent (the coefficient from the model)
- **Total ROI**: Overall return ratio (Sales Contribution / Total Spend)

**💡 Key Insights:**
- Higher Marginal ROI means better efficiency per dollar spent
- Total ROI shows the overall effectiveness of the channel
- Channels with Marginal ROI > 1.0 generate more in sales than they cost
- Compare these values to optimize your marketing budget allocation
""")

# Scenario Planner
st.header("🎯 Scenario Planner")
st.subheader("6️⃣ What-If Analysis: Predict Sales from Custom Marketing Spend")
st.caption("Adjust marketing spend for each channel and see predicted sales impact compared to baseline")

# Calculate baseline (mean spends)
baseline_tv = df['TV_Spend'].mean()
baseline_radio = df['Radio_Spend'].mean()
baseline_digital = df['Digital_Spend'].mean()

# Predict baseline sales
baseline_prediction = (
    mmm_results['intercept'] +
    mmm_results['coefficients']['TV_Spend'] * baseline_tv +
    mmm_results['coefficients']['Radio_Spend'] * baseline_radio +
    mmm_results['coefficients']['Digital_Spend'] * baseline_digital
)

# Display baseline
st.markdown("**📊 Baseline Scenario (Current Average Spend)**")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Avg TV Spend", f"${baseline_tv:,.2f}")
with col2:
    st.metric("Avg Radio Spend", f"${baseline_radio:,.2f}")
with col3:
    st.metric("Avg Digital Spend", f"${baseline_digital:,.2f}")
with col4:
    st.metric("Predicted Sales", f"${baseline_prediction:,.2f}")

st.markdown("---")

# Scenario inputs
st.markdown("**🎨 Custom Scenario: Adjust Marketing Spend**")
st.caption("Move the sliders or enter custom values to see how changes in marketing spend affect predicted sales")

col1, col2, col3 = st.columns(3)

with col1:
    scenario_tv = st.number_input(
        "TV Spend ($)",
        min_value=0.0,
        max_value=float(df['TV_Spend'].max() * 2),
        value=float(baseline_tv),
        step=1000.0,
        format="%.2f",
        help="Enter TV advertising budget"
    )

with col2:
    scenario_radio = st.number_input(
        "Radio Spend ($)",
        min_value=0.0,
        max_value=float(df['Radio_Spend'].max() * 2),
        value=float(baseline_radio),
        step=500.0,
        format="%.2f",
        help="Enter Radio advertising budget"
    )

with col3:
    scenario_digital = st.number_input(
        "Digital Spend ($)",
        min_value=0.0,
        max_value=float(df['Digital_Spend'].max() * 2),
        value=float(baseline_digital),
        step=1000.0,
        format="%.2f",
        help="Enter Digital marketing budget"
    )

# Predict scenario sales
scenario_prediction = (
    mmm_results['intercept'] +
    mmm_results['coefficients']['TV_Spend'] * scenario_tv +
    mmm_results['coefficients']['Radio_Spend'] * scenario_radio +
    mmm_results['coefficients']['Digital_Spend'] * scenario_digital
)

# Calculate deltas
delta_sales = scenario_prediction - baseline_prediction
delta_percent = (delta_sales / baseline_prediction) * 100 if baseline_prediction > 0 else 0

scenario_total_spend = scenario_tv + scenario_radio + scenario_digital
baseline_total_spend = baseline_tv + baseline_radio + baseline_digital
delta_spend = scenario_total_spend - baseline_total_spend
delta_spend_percent = (delta_spend / baseline_total_spend) * 100 if baseline_total_spend > 0 else 0

# Display scenario results
st.markdown("---")
st.markdown("**📈 Scenario Results**")

result_col1, result_col2, result_col3, result_col4 = st.columns(4)

with result_col1:
    st.metric(
        "Predicted Sales",
        f"${scenario_prediction:,.2f}",
        delta=f"${delta_sales:,.2f} ({delta_percent:+.1f}%)",
        help="Predicted sales for your custom scenario"
    )

with result_col2:
    st.metric(
        "Total Marketing Spend",
        f"${scenario_total_spend:,.2f}",
        delta=f"${delta_spend:,.2f} ({delta_spend_percent:+.1f}%)",
        help="Total spend across all channels"
    )

with result_col3:
    scenario_roi = scenario_prediction / scenario_total_spend if scenario_total_spend > 0 else 0
    baseline_roi = baseline_prediction / baseline_total_spend if baseline_total_spend > 0 else 0
    delta_roi = scenario_roi - baseline_roi
    st.metric(
        "Overall ROI",
        f"{scenario_roi:.3f}x",
        delta=f"{delta_roi:+.3f}x",
        help="Sales generated per dollar spent"
    )

with result_col4:
    incremental_roi = delta_sales / delta_spend if delta_spend != 0 else 0
    st.metric(
        "Incremental ROI",
        f"{incremental_roi:.3f}x" if delta_spend != 0 else "N/A",
        help="Return on incremental spend vs baseline"
    )

# Detailed breakdown
st.markdown("**🔍 Detailed Breakdown**")

breakdown_data = []
for channel, spend_key in [('TV', 'TV_Spend'), ('Radio', 'Radio_Spend'), ('Digital', 'Digital_Spend')]:
    baseline_spend = df[spend_key].mean()
    scenario_spend = {'TV': scenario_tv, 'Radio': scenario_radio, 'Digital': scenario_digital}[channel]
    coef = mmm_results['coefficients'][spend_key]
    
    baseline_contrib = coef * baseline_spend
    scenario_contrib = coef * scenario_spend
    delta_contrib = scenario_contrib - baseline_contrib
    
    breakdown_data.append({
        'Channel': channel,
        'Baseline Spend': f"${baseline_spend:,.2f}",
        'Scenario Spend': f"${scenario_spend:,.2f}",
        'Δ Spend': f"${scenario_spend - baseline_spend:+,.2f}",
        'Baseline Contribution': f"${baseline_contrib:,.2f}",
        'Scenario Contribution': f"${scenario_contrib:,.2f}",
        'Δ Contribution': f"${delta_contrib:+,.2f}"
    })

breakdown_df = pd.DataFrame(breakdown_data)
st.dataframe(breakdown_df, width='stretch', hide_index=True)

# Comparison chart
st.markdown("**📊 Baseline vs Scenario Comparison**")

fig_comparison = go.Figure()

channels = ['TV', 'Radio', 'Digital']
baseline_spends = [baseline_tv, baseline_radio, baseline_digital]
scenario_spends = [scenario_tv, scenario_radio, scenario_digital]

fig_comparison.add_trace(go.Bar(
    name='Baseline',
    x=channels,
    y=baseline_spends,
    marker_color='lightblue',
    text=[f"${val:,.0f}" for val in baseline_spends],
    textposition='outside'
))

fig_comparison.add_trace(go.Bar(
    name='Scenario',
    x=channels,
    y=scenario_spends,
    marker_color=['#ff7f0e', '#2ca02c', '#d62728'],
    text=[f"${val:,.0f}" for val in scenario_spends],
    textposition='outside'
))

fig_comparison.update_layout(
    xaxis_title='Marketing Channel',
    yaxis_title='Spend ($)',
    barmode='group',
    height=400
)

st.plotly_chart(fig_comparison, width='stretch')

# Interpretation
st.success("""
**💡 How to Use the Scenario Planner:**

1. **Baseline:** Shows your current average spend and predicted sales
2. **Adjust Inputs:** Modify spend for each channel to test different budget allocations
3. **View Impact:** See predicted sales, ROI, and comparison to baseline
4. **Optimize Budget:** Experiment to find the best channel mix for your goals

**Key Metrics:**
- **Δ vs Baseline:** Shows the change from your current average performance
- **Incremental ROI:** Return on additional spend compared to baseline
- **Overall ROI:** Total sales efficiency for the scenario
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Marketing Mix Modeling App | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
