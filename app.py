# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="COVID Prediction Dashboard", layout="wide")
st.title("🌍 COVID Prediction & Forecast Dashboard")

# ---------------------------
# Load Data
# ---------------------------
try:
    df = pd.read_csv("worldometer_data (1).csv")
except FileNotFoundError:
    st.error("Error: 'worldometer_data (1).csv' not found!")
    st.stop()

# Fill numeric NaNs with median
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Feature Engineering
df['Test Positive Rate'] = np.where(df['TotalTests'] > 0, df['TotalCases'] / df['TotalTests'], 0)
df['Test Positive Rate'] = df['Test Positive Rate'].replace([np.inf, -np.inf], 0).fillna(0)
df['Total_case_log'] = np.log1p(df['TotalCases'])
df['Population_Log'] = np.log1p(df['Population'])
df['Testing_Log'] = np.log1p(df['TotalTests'])

# ---------------------------
# Load Trained Models
# ---------------------------
try:
    model = joblib.load('xgb_model.pkl')  # your existing model
    cases_model = deaths_model = vacc_model = model
except FileNotFoundError:
    st.error("Error: Model file not found! Ensure xgb_model.pkl exists.")
    st.stop()

# ---------------------------
# Sidebar: Country, Month & Year
# ---------------------------
st.sidebar.header("Prediction Settings")
countries = df['Country/Region'].unique()
selected_countries = st.sidebar.multiselect("Select Countries", countries, default=countries[:5])

years = list(range(2019, 2026))
selected_year = st.sidebar.selectbox("Select Year", years, index=1)
months = list(range(1, 13))
selected_month = st.sidebar.selectbox("Select Month", months, index=0)

st.sidebar.markdown(f"**Simulating predictions for: {selected_year}-{selected_month:02d}**")

# ---------------------------
# Prepare Data for Prediction
# ---------------------------
filtered_data = df[df['Country/Region'].isin(selected_countries)].copy()
feature_cols = ['TotalCases', 'TotalDeaths', 'TotalRecovered', 'ActiveCases', 'TotalTests']
X = filtered_data[feature_cols]

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Simulate effect of month/year
time_factor = 1 + ((selected_year - 2020) * 0.1) + ((selected_month - 1) * 0.01)
X_scaled_time = X_scaled * time_factor

# ---------------------------
# Predict Cases, Deaths, Vaccinations
# ---------------------------
filtered_data['Predicted_Cases'] = np.expm1(cases_model.predict(X_scaled_time))
filtered_data['Predicted_Deaths'] = np.expm1(deaths_model.predict(X_scaled_time))
filtered_data['Predicted_Vaccinated'] = np.expm1(vacc_model.predict(X_scaled_time))

# ---------------------------
# Display Table
# ---------------------------
st.subheader(f"Predicted COVID Metrics for {selected_month:02d}/{selected_year}")
st.dataframe(filtered_data[['Country/Region', 'Predicted_Cases', 'Predicted_Deaths', 'Predicted_Vaccinated']].sort_values(by='Predicted_Cases', ascending=False))

# ---------------------------
# Download Predictions
# ---------------------------
csv = filtered_data[['Country/Region', 'Predicted_Cases', 'Predicted_Deaths', 'Predicted_Vaccinated']].to_csv(index=False)
st.download_button("📥 Download Predictions as CSV", csv, f"covid_predictions_{selected_year}_{selected_month:02d}.csv", "text/csv")

# ---------------------------
# Country-wise Comparison Charts
# ---------------------------
st.subheader("Country-wise Comparison Charts")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Predicted Cases**")
    plt.figure(figsize=(6,4))
    sns.barplot(x='Country/Region', y='Predicted_Cases', data=filtered_data, palette='Oranges')
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()

with col2:
    st.markdown("**Predicted Deaths**")
    plt.figure(figsize=(6,4))
    sns.barplot(x='Country/Region', y='Predicted_Deaths', data=filtered_data, palette='Reds')
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()

with col3:
    st.markdown("**Predicted Vaccinations**")
    plt.figure(figsize=(6,4))
    sns.barplot(x='Country/Region', y='Predicted_Vaccinated', data=filtered_data, palette='Greens')
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()

# ---------------------------
# ARIMA Explanation
# ---------------------------
st.subheader("📖 What is ARIMA Forecast?")
st.markdown("""
**ARIMA (AutoRegressive Integrated Moving Average)** is a popular model for **time series forecasting**.  
It predicts future values based on past observations and trends.

- **AR (AutoRegressive)**: Uses previous values to predict the current value.
- **I (Integrated)**: Differencing the series to make it stationary (removes trends).
- **MA (Moving Average)**: Models the error term as a combination of past errors.

ARIMA is suitable for predicting COVID metrics like cases, deaths, or vaccinations over time.
""")

# ---------------------------
# Multi-Country ARIMA Forecast
# ---------------------------
st.subheader("Multi-Country ARIMA Forecast")
forecast_countries = st.multiselect("Select Countries for Forecast", countries, default=countries[:3])
forecast_metric = st.selectbox("Select Metric to Forecast", ['Predicted_Cases', 'Predicted_Deaths', 'Predicted_Vaccinated'], key='multi_metric')

# Sidebar: forecast horizon
forecast_period_type = st.radio("Forecast Horizon Type", ['Days', 'Months'], index=1)
if forecast_period_type == 'Days':
    forecast_horizon = st.slider("Forecast Next Days", 1, 365, 30)
else:
    forecast_horizon = st.slider("Forecast Next Months", 1, 36, 12)

plt.figure(figsize=(12,6))

for country in forecast_countries:
    country_series = filtered_data[filtered_data['Country/Region'] == country][forecast_metric]
    
    # Starting date from sidebar
    start_date = pd.Timestamp(year=selected_year, month=selected_month, day=1)
    
    # Simulate daily/monthly values if single snapshot
    freq = 'D' if forecast_period_type == 'Days' else 'M'
    series_length = 30 if freq=='D' else 1
    series = pd.Series(np.repeat(country_series.values[0], series_length),
                       index=pd.date_range(start=start_date, periods=series_length, freq=freq))
    
    try:
        # Fit ARIMA
        model = ARIMA(series, order=(1,1,1))
        arima_result = model.fit()
        forecast = arima_result.forecast(steps=forecast_horizon)
        
        # Forecast index
        forecast_index = pd.date_range(series.index[-1] + pd.Timedelta(days=1) if freq=='D' else series.index[-1] + pd.DateOffset(months=1),
                                       periods=forecast_horizon, freq=freq)
        
        # Plot
        plt.plot(series, label=f'{country} Original')
        plt.plot(forecast_index, forecast, marker='o', linestyle='--', label=f'{country} Forecast')
        
    except Exception as e:
        st.warning(f"ARIMA model could not be fit for {country}: {e}")

plt.title(f"ARIMA Forecast for Multiple Countries ({forecast_metric})")
plt.xlabel("Date")
plt.ylabel(forecast_metric)
plt.legend()
plt.grid(True)
st.pyplot(plt.gcf())
plt.clf()

st.markdown("---")
st.write("Project by Your Name | COVID Prediction Dashboard")
