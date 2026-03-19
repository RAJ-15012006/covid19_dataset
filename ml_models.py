import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

# Avoid division by zero in MAPE
def calc_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    if not np.any(non_zero): return 0.0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

@st.cache_data
def create_lag_features(series, lags=7):
    df = pd.DataFrame(series)
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df.iloc[:, 0].shift(i)
    df["rolling_mean_7"] = df.iloc[:, 0].rolling(window=7).mean().shift(1)
    return df.dropna()

# -----------------
# 1. FORECASTER: XGBOOST vs ARIMA
# -----------------
@st.cache_resource(show_spinner=False)
def run_multi_model_forecast(df, country, metric="new_cases", forecast_steps=30):
    """
    Runs ARIMA and XGBoost. Returns a leaderboard comparing their accuracy
    on the held-out validation set.
    """
    from forecasting import fit_arima_forecast
    
    # 1. Get ARIMA results using existing function
    arima_df, arima_metrics, err = fit_arima_forecast(df, country, metric, forecast_steps)
    
    # 2. XGBoost Setup
    df_country = df[df["location"] == country].sort_values("date").set_index("date")
    if len(df_country) < forecast_steps + 30:
        return None, None, None, "Insufficient Data"
        
    series = df_country[metric].fillna(0).resample('D').sum()
    lagged_df = create_lag_features(series, lags=14)
    
    if len(lagged_df) < forecast_steps + 10:
        return arima_df, arima_metrics, None, err
        
    # Train/Test Split
    train_df = lagged_df.iloc[:-forecast_steps]
    test_df = lagged_df.iloc[-forecast_steps:]
    
    X_train, y_train = train_df.drop(columns=[metric]), train_df[metric]
    X_test, y_test = test_df.drop(columns=[metric]), test_df[metric]
    
    # Train XGBoost
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, learning_rate=0.1)
    xgb_model.fit(X_train, y_train)
    
    # Predict
    xgb_preds = xgb_model.predict(X_test)
    xgb_preds = np.maximum(xgb_preds, 0) # No negative cases
    
    xgb_metrics = {
        "MAE": mean_absolute_error(y_test, xgb_preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, xgb_preds)),
        "MAPE": calc_mape(y_test, xgb_preds)
    }
    
    xgb_df = pd.DataFrame({
        "date": test_df.index,
        "actual": y_test.values,
        "forecast": xgb_preds
    })
    
    # Leaderboard
    leaderboard = []
    if arima_metrics:
        leaderboard.append({"Model": "ARIMA", "MAE": arima_metrics["MAE"], "RMSE": arima_metrics["RMSE"], "MAPE": arima_metrics["MAPE"]})
    leaderboard.append({"Model": "XGBoost", "MAE": xgb_metrics["MAE"], "RMSE": xgb_metrics["RMSE"], "MAPE": xgb_metrics["MAPE"]})
    
    leaderboard_df = pd.DataFrame(leaderboard).sort_values("MAE").reset_index(drop=True)
    
    return arima_df, xgb_df, leaderboard_df, None

# -----------------
# 2. WAVE PREDICTOR (Random Forest Classifier)
# -----------------
@st.cache_resource(show_spinner=False)
def train_wave_detector(df_global):
    """
    Creates a heuristic-based training dataset across countries to predict 'Wave Next 30 Days'
    and trains a Random Forest Classifier.
    """
    features = []
    labels = []
    
    # To save time in the app, we build a lightweight training set sample
    sample_countries = df_global["location"].unique()[:20] 
    
    for country in sample_countries:
        country_data = df_global[df_global["location"] == country].sort_values("date").reset_index(drop=True)
        if len(country_data) < 100: continue
            
        country_data["7d_cases"] = country_data["new_cases"].rolling(7).sum()
        country_data["7d_growth"] = country_data["7d_cases"].pct_change(7).clip(lower=-1, upper=5).fillna(0)
        country_data["vax_pct"] = (country_data["people_fully_vaccinated"] / (country_data["population"] + 1)).fillna(0)
        
        # Determine if a "Wave" happened in the next 30 days
        country_data["future_max_cases"] = country_data["7d_cases"].shift(-30).rolling(15).max()
        country_data["is_wave"] = (country_data["future_max_cases"] > (country_data["7d_cases"] * 2)) & (country_data["future_max_cases"] > 500)
        
        valid_rows = country_data.dropna(subset=["7d_growth", "vax_pct", "is_wave"])
        
        if len(valid_rows) > 0:
            features.append(valid_rows[["7d_growth", "vax_pct", "stringency_index", "positive_rate"]].fillna(0))
            labels.append(valid_rows["is_wave"].astype(int))
            
    if not features: return None
    
    X = pd.concat(features, ignore_index=True)
    y = pd.concat(labels, ignore_index=True)
    
    rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    rf.fit(X, y)
    return rf

def predict_current_wave_risk(df_country, rf_model):
    """
    Scores the latest date for a given country using the trained RF model.
    """
    country_data = df_country.sort_values("date").tail(15).copy()
    if len(country_data) < 15 or rf_model is None:
        return 0, "Unknown"
        
    country_data["7d_cases"] = country_data["new_cases"].rolling(7).sum()
    country_data["7d_growth"] = country_data["7d_cases"].pct_change(7).clip(lower=-1, upper=5).fillna(0)
    country_data["vax_pct"] = (country_data["people_fully_vaccinated"] / (country_data["population"] + 1)).fillna(0)
    
    latest = country_data.iloc[-1]
    
    X_latest = pd.DataFrame([{
        "7d_growth": latest["7d_growth"],
        "vax_pct": latest["vax_pct"],
        "stringency_index": 0 if pd.isna(latest.get("stringency_index")) else latest["stringency_index"],
        "positive_rate": 0 if pd.isna(latest.get("positive_rate")) else latest["positive_rate"]
    }]).fillna(0)
    
    prob = rf_model.predict_proba(X_latest)[0][1] * 100
    
    if prob > 60: risk_label = "HIGH RISK 🔴"
    elif prob > 30: risk_label = "ELEVATED 🟡"
    else: risk_label = "LOW RISK 🟢"
    
    return prob, risk_label

# -----------------
# 3. COUNTRY RISK SCORER
# -----------------
def calculate_risk_score(df_country):
    """
    Risk Score = weighted combination of:
    - Case growth velocity        (30%)
    - Death rate trend            (25%)
    - Healthcare capacity proxy   (20%)
    - Vaccination gap             (15%)
    - Population density          (10%) -> We'll proxy with general severity if density not in df
    """
    recent = df_country.sort_values("date").tail(30).reset_index(drop=True)
    if len(recent) < 30: return 0, {}
    
    # 1. Growth Velocity (30%)
    cases_first_half = recent["new_cases"].iloc[:15].sum()
    cases_second_half = recent["new_cases"].iloc[15:].sum()
    growth_ratio = min((cases_second_half / (cases_first_half + 1)), 3.0) 
    score_growth = (growth_ratio / 3.0) * 30
    
    # 2. Death Trend (25%)
    deaths_first = recent["new_deaths"].iloc[:15].sum()
    deaths_second = recent["new_deaths"].iloc[15:].sum()
    death_ratio = min((deaths_second / (deaths_first + 1)), 2.0)
    score_deaths = (death_ratio / 2.0) * 25
    
    # 3. Healthcare Capacity Proxy (Using Hosp Patients or Death Rate) (20%)
    if "hosp_patients" in recent.columns and recent["hosp_patients"].sum() > 0:
        hosp_trend = min(recent["hosp_patients"].iloc[-1] / (recent["hosp_patients"].mean() + 1), 2.0)
        score_health = (hosp_trend / 2.0) * 20
    else:
        # Proxy via CFR
        cfr = recent["new_deaths"].sum() / (recent["new_cases"].sum() + 1)
        score_health = min(cfr * 100, 20)
        
    # 4. Vaccination Gap (15%)
    pop = recent["population"].iloc[0]
    vax = recent["people_fully_vaccinated"].iloc[-1]
    vax_pct = min(vax / (pop + 1), 1.0)
    score_vax = (1.0 - vax_pct) * 15
    
    # 5. Missing metrics catch-all / Base risk (10%)
    score_base = 5.0 
    
    total_score = min(score_growth + score_deaths + score_health + score_vax + score_base, 100)
    
    breakdown = {
        "Velocity Risk (30%)": round(score_growth, 1),
        "Death Trend Risk (25%)": round(score_deaths, 1),
        "Healthcare Load (20%)": round(score_health, 1),
        "Vaccination Gap (15%)": round(score_vax, 1),
        "Base Structural (10%)": 5.0
    }
    
    return round(total_score, 1), breakdown

# -----------------
# 4. VACCINE IMPACT (Causal Inference Counterfactual)
# -----------------
@st.cache_data
def calculate_vaccine_impact(df_country):
    """
    Train a simple regression on 2020 Data (No Vaccines) predicting Deaths from Cases.
    Then project this model into 2021-2022 to see what deaths WOULD HAVE BEEN without vaccines.
    """
    df = df_country.sort_values("date").copy()
    
    # Pre-vax era (Strictly 2020)
    pre_vax = df[df["date"] < "2021-01-01"].copy()
    post_vax = df[df["date"] >= "2021-01-01"].copy()
    
    if len(pre_vax) < 100 or len(post_vax) < 100:
        return None, None
        
    # Smooth cases to align with delayed deaths
    pre_vax["smoothed_cases"] = pre_vax["new_cases"].rolling(14).mean().shift(14).fillna(0)
    post_vax["smoothed_cases"] = post_vax["new_cases"].rolling(14).mean().shift(14).fillna(0)
    
    # Very simple Linear relationship: Cases -> Deaths
    from sklearn.linear_model import LinearRegression
    
    X_train = pre_vax[["smoothed_cases"]]
    y_train = pre_vax["new_deaths"].fillna(0)
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Predict Counterfactual on Post-Vax era
    X_test = post_vax[["smoothed_cases"]]
    projected_deaths = lr.predict(X_test)
    projected_deaths = np.maximum(projected_deaths, 0) # No negative deaths
    
    post_vax["counterfactual_deaths"] = projected_deaths
    
    # Calculate totals
    actual_deaths_total = post_vax["new_deaths"].sum()
    counterfactual_total = projected_deaths.sum()
    
    lives_saved = max(counterfactual_total - actual_deaths_total, 0)
    
    return post_vax, lives_saved
