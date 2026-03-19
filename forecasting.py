import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_idx = y_true != 0
    if not np.any(non_zero_idx):
        return 0.0
    return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100

@st.cache_resource(show_spinner=False)
def fit_arima_forecast(df, country, metric, forecast_steps=30):
    """
    Fits an ARIMA model to the selected metric for a country.
    Uses the historical timeframe. To demonstrate forecasting inside the window,
    we test on the last `forecast_steps` of the series.
    """
    df_country = df[df["location"] == country].sort_values("date")
    
    if len(df_country) < forecast_steps + 30:
        return None, None, f"Insufficient data for {country} to forecast."
        
    # Resample to daily metric
    df_country = df_country.set_index("date")
    series = df_country[metric].fillna(0)
    
    # We simulate forecasting by making a train/test split at the end of the historical range.
    train_series = series.iloc[:-forecast_steps]
    test_series = series.iloc[-forecast_steps:]
    
    if train_series.sum() == 0:
        return None, None, f"Data is purely empty for {country} for {metric}."
    
    # Stationarity Check
    try:
        result = adfuller(train_series.dropna())
        d = 0
        if result[1] > 0.05:  # not stationary
            d = 1  # difference it once
    except Exception:
        d = 1
        
    try:
        # ARIMA parameters (p,d,q)
        # We use a relatively quick model to ensure Streamlit dashboard stays performant
        model = ARIMA(train_series, order=(2, d, 2))
        fitted_model = model.fit()
        
        # Forecast
        forecast = fitted_model.get_forecast(steps=forecast_steps)
        pred_mean = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05) # 95% confidence interval
        
        # Metrics
        y_true = test_series.values
        y_pred = pred_mean.values
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        metrics = {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        }
        
        forecast_df = pd.DataFrame({
            "date": test_series.index,
            "actual": test_series.values,
            "forecast": pred_mean.values,
            "lower_bound": conf_int.iloc[:, 0].values,
            "upper_bound": conf_int.iloc[:, 1].values
        })
        
        return forecast_df, metrics, None
        
    except Exception as e:
        return None, None, f"Forecasting failed for {country}: {str(e)}"
