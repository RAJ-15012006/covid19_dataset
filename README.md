🔭 Overview
The COVID-19 Predictive Intelligence System is not just another case tracker — it is a multi-module machine learning platform designed to answer questions that matter:
QuestionModuleHow many cases/deaths will a country see in the next 30 days?📈 Case & Death ForecasterIs a new COVID wave approaching?🌊 Wave PredictorWhich countries are currently at highest pandemic risk?🌍 Country Risk ScorerHow many lives did vaccines actually save?💉 Vaccine Impact Predictor
All predictions are grounded in real historical data (Jan 2020 – May 2023) and evaluated with rigorous backtesting. Forecasts are never extended beyond the WHO-declared end of the global health emergency (May 2023).

🚀 Live Demo

🔗 Streamlit App: [https://your-streamlit-app.streamlit.app](https://covid19dataset-qpjozfwf2memgbu2qiut9t.streamlit.app/)


✨ Features
📈 Module 1 — Future Case & Death Forecaster

Predict cases or deaths for any country over 7 / 14 / 30 day horizons
Three models compared side-by-side: ARIMA, Facebook Prophet, XGBoost
Confidence intervals (80% and 95%) shown as shaded bands
Automatic stationarity check (ADF test) with differencing if needed
Model accuracy leaderboard: which model wins for which country

🌊 Module 2 — Wave Predictor

Binary classification: Will a new wave hit in the next 30 days?
Outputs probability score (0–100%) with confidence level
Engineered features: Rt estimation, 7-day growth rate, days since last peak, vaccination coverage, seasonality
Model: Random Forest Classifier with SHAP explainability

🌍 Module 3 — Country Risk Scorer

Real-time risk score (0–100) for every country
Weighted multi-factor scoring: case velocity, death trend, healthcare capacity proxy, vaccination gap, population density
Interactive world map — click any country for full risk breakdown
Historical risk score timeline per country

💉 Module 4 — Vaccine Impact Predictor

Counterfactual analysis: What would deaths have looked like WITHOUT vaccines?
Train model on pre-vaccine data → project forward → compare to actuals
Estimate lives saved per country with confidence range
Visualize the "vaccination effect" directly on the death curve


🏗️ System Architecture
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                        │
│         (Forecast UI · Risk Map · Model Report)         │
└───────────────────────┬─────────────────────────────────┘
                        │ REST API calls
┌───────────────────────▼─────────────────────────────────┐
│                   FastAPI Backend                        │
│    /forecast  /wave-risk  /country-risk  /vaccine-impact │
└───────┬───────────────┬──────────────────────────────────┘
        │               │
┌───────▼──────┐ ┌──────▼───────────────────────────────┐
│  ML Models   │ │         Data Pipeline                 │
│  ARIMA       │ │  OWID CSV → Cleaning → Features       │
│  Prophet     │ │  Lag features, rolling means, Rt      │
│  XGBoost     │ │  estimation, wave labels              │
│  Random Forest│ └──────────────────────────────────────┘
└──────────────┘
        │
┌───────▼──────────────────────────────────────────────┐
│              Streamlit Analytics View                  │
│     (EDA · Backtesting · Model Comparison Report)     │
└───────────────────────────────────────────────────────┘

📂 Project Structure
covid-prediction-system/
│
├── backend/                        # FastAPI backend
│   ├── main.py                     # API entry point
│   ├── routers/
│   │   ├── forecast.py             # Forecast endpoints
│   │   ├── wave.py                 # Wave prediction endpoints
│   │   ├── risk.py                 # Risk scoring endpoints
│   │   └── vaccine.py             # Vaccine impact endpoints
│   ├── models/
│   │   ├── forecaster.py           # ARIMA + Prophet + XGBoost
│   │   ├── wave_detector.py        # RF Classifier for wave prediction
│   │   ├── risk_scorer.py          # Multi-factor risk engine
│   │   └── vaccine_impact.py       # Counterfactual analysis
│   ├── data/
│   │   ├── loader.py               # OWID data download & caching
│   │   ├── features.py             # Feature engineering pipeline
│   │   └── preprocessor.py         # Cleaning, imputation
│   └── requirements.txt
│
├── frontend/                       # React frontend
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Predict.jsx         # Forecasting interface
│   │   │   ├── RiskMap.jsx         # World risk map
│   │   │   └── ModelReport.jsx     # Accuracy & methodology
│   │   ├── components/
│   │   │   ├── ForecastChart.jsx   # Plotly line + confidence bands
│   │   │   ├── RiskGlobe.jsx       # Interactive world map
│   │   │   ├── WaveAlert.jsx       # Wave probability card
│   │   │   └── MetricCard.jsx      # KPI display card
│   │   └── api/
│   │       └── client.js           # Axios API calls
│   └── package.json
│
├── streamlit_app/                  # Streamlit analytics view
│   ├── app.py                      # Main Streamlit app
│   ├── pages/
│   │   ├── 1_Forecasting.py
│   │   ├── 2_Wave_Predictor.py
│   │   ├── 3_Risk_Map.py
│   │   └── 4_Vaccine_Impact.py
│   └── requirements.txt
│
├── notebooks/                      # Jupyter EDA & training notebooks
│   ├── 01_EDA_and_Cleaning.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training_Forecasting.ipynb
│   ├── 04_Wave_Detection_Model.ipynb
│   └── 05_Backtesting_and_Evaluation.ipynb
│
├── tests/
│   ├── test_forecaster.py
│   ├── test_wave_detector.py
│   └── test_risk_scorer.py
│
├── .env.example
├── docker-compose.yml
└── README.md

🤖 ML Models & Methodology
Forecasting Models
ModelTypeStrengthsUsed ForARIMAStatisticalInterpretable baseline, handles trendsShort-term forecastsFacebook ProphetStatistical + MLSeasonality, holiday effects, robust to missing dataMedium-term forecastsXGBoostGradient BoostingNon-linear patterns, lag feature learningBest overall accuracyEnsembleWeighted AverageReduces individual model varianceFinal prediction
Feature Engineering
Lag features:         cases_lag_7, cases_lag_14, deaths_lag_7
Rolling statistics:   7-day rolling mean, 14-day rolling std
Growth metrics:       daily_growth_rate, weekly_change_%
Epidemiological:      Rt_estimate (estimated reproduction number)
External:             stringency_index, vaccination_rate, population_density
Temporal:             day_of_week, month, days_since_outbreak_start
Wave Detection (Classification)

Labels: Waves auto-detected using scipy.signal.find_peaks with prominence threshold
Model: Random Forest (n_estimators=200, max_depth=10)
Explainability: SHAP values show which features drive each prediction
Validation: 5-fold time-series cross-validation (no data leakage)

Vaccine Impact (Counterfactual)
1. Train XGBoost on pre-vaccination data (Jan 2020 – Jan 2021)
2. Project death trajectory forward assuming no vaccine rollout
3. Compare projected deaths vs actual deaths post-vaccination
4. Delta = estimated lives saved (with 95% confidence interval)
Evaluation Metrics
MetricDescriptionMAEMean Absolute Error — average prediction errorRMSERoot Mean Squared Error — penalizes large errorsMAPEMean Absolute Percentage Error — scale-independentAUC-ROCWave prediction classification accuracy

📡 API Reference
POST /api/forecast
jsonRequest:
{
  "country": "India",
  "metric": "new_cases",
  "days_ahead": 30,
  "model": "ensemble"
}

Response:
{
  "predictions": [12000, 13400, ...],
  "confidence_upper_95": [15000, 16800, ...],
  "confidence_lower_95": [9000, 10200, ...],
  "metrics": { "mae": 1240, "rmse": 1890, "mape": 8.3 }
}
GET /api/wave-risk/{country}
jsonResponse:
{
  "country": "India",
  "wave_probability": 0.73,
  "risk_level": "HIGH",
  "key_factors": [
    { "feature": "7_day_growth_rate", "impact": 0.42 },
    { "feature": "vaccination_coverage", "impact": -0.28 }
  ]
}
GET /api/country-risk
jsonResponse:
{
  "countries": [
    {
      "name": "Brazil",
      "score": 67,
      "breakdown": {
        "case_velocity": 72,
        "death_trend": 65,
        "vaccination_gap": 58,
        "healthcare_capacity": 71,
        "population_density": 55
      }
    }
  ]
}
POST /api/vaccine-impact
jsonRequest: { "country": "USA" }

Response:
{
  "actual_deaths": 890000,
  "projected_deaths_no_vaccine": 1340000,
  "lives_saved": 450000,
  "lives_saved_confidence_interval": [380000, 520000],
  "vaccination_effectiveness_signal": "strong"
}

⚙️ Installation
Prerequisites

Python 3.10+
Node.js 18+
Git

1. Clone the Repository
bashgit clone https://github.com/yourusername/covid-prediction-system.git
cd covid-prediction-system
2. Backend Setup (FastAPI)
bashcd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
3. Frontend Setup (React)
bashcd frontend
npm install
4. Streamlit Setup
bashcd streamlit_app
pip install -r requirements.txt
5. Environment Variables
bashcp .env.example .env
# Edit .env — no API keys needed, all data is open source

🖥️ Usage
Run Backend
bashcd backend
uvicorn main:app --reload --port 8000
# API docs available at: http://localhost:8000/docs
Run React Frontend
bashcd frontend
npm run dev
# App available at: http://localhost:5173
Run Streamlit App
bashcd streamlit_app
streamlit run app.py
# App available at: http://localhost:8501
Run with Docker (all services)
bashdocker-compose up --build

📊 Dataset
SourceDescriptionURLOur World in DataCases, deaths, vaccinations, 200+ countriesowid-covid-data.csvOxford Stringency IndexGovernment response measuresIncluded in OWID datasetWHO Variant TimelineAlpha, Delta, Omicron emergence datesUsed for wave annotations
Data scope: January 22, 2020 – May 5, 2023 (WHO global emergency end date)
Preprocessing steps:

Forward-fill missing values (≤7 days gap)
Linear interpolation for longer gaps
Negative value correction (reporting corrections in raw data)
7-day rolling average applied before model training


📈 Results & Accuracy
Forecasting Model Comparison (30-day horizon, India)
ModelMAERMSEMAPEARIMA18,42024,31011.2%Prophet14,89019,7808.7%XGBoost11,23015,6406.4%Ensemble9,87013,2105.8%
Wave Detection Performance
MetricScoreAccuracy84.3%Precision81.7%Recall87.2%AUC-ROC0.91

Full backtesting results available in notebooks/05_Backtesting_and_Evaluation.ipynb


🛠️ Tech Stack
LayerTechnologyML & ForecastingPython, ARIMA (statsmodels), Prophet (Meta), XGBoost, Scikit-learnExplainabilitySHAPBackend APIFastAPI, Uvicorn, PydanticFrontendReact 18, Plotly.js, Axios, TailwindCSSAnalytics UIStreamlit, PlotlyData ProcessingPandas, NumPy, SciPyTestingPytestDeploymentDocker, Railway (API), Vercel (Frontend), Streamlit Cloud

🔮 Future Work

 Real-time data pipeline (auto-refresh daily from OWID)
 LSTM deep learning model for long-horizon forecasting
 Mobile-responsive React PWA
 Email/SMS alerts when wave probability exceeds 70%
 Extend framework to other infectious diseases (Monkeypox, Influenza)
 Natural language report generation per country




📄 License
This project is licensed under the MIT License — see the LICENSE file for details.

🙏 Acknowledgements

Our World in Data — for the comprehensive open-source COVID-19 dataset
Facebook Prophet — time series forecasting library
WHO COVID-19 Dashboard — reference data and variant timelines
