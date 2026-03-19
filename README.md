# COVID-19 Analytics & Forecasting Dashboard

A professional Streamlit dashboard using real historical data (Jan 2020 - May 2023) from Our World in Data.

## Features
- **Strict Data Scope**: Stops at May 2023 when WHO declared the end of the global emergency.
- **Forecasting Simulation**: Uses ARIMA to forecast the final 30 days of the historical data, demonstrating predictive capabilities against known actuals (with MAE, RMSE, MAPE metrics).
- **Advanced Plotly Visualizations**:
  - 3D Surface Plot of cases across time and country.
  - Animated Global Choropleth Map showing spread over time.
  - Dual-Axis Charts (Raw Data vs 7-Day Rolling Averages).
  - Stacked Area Charts.
  - Vaccination Progress Bar Race.
  - Correlation Heatmaps (Cases, Deaths, Stringency, GDP, Vaccinations).
- **Custom UI**: Dark theme styling, glowing KPI cards, CSS grid dot pattern background.

## Setup Instructions

1. **Prerequisites**
   Ensure you have Python 3.10+ installed.

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Dashboard**
   ```bash
   streamlit run app.py
   ```

## Architecture
- `data_loader.py`: Handles data ingestion from Our World In Data, date filtering, and missing value interpolation. Cached via `@st.cache_data`.
- `forecasting.py`: Implements stationarity checking (ADF) and ARIMA modeling. Cached via `@st.cache_resource`.
- `visualizations.py`: Houses all custom Plotly charting functions with a shared dark neon theme.
- `app.py`: Streamlit entry point, coordinating the sidebar state, KPI cards, and visual layout.

## Data Source
Data provided by [Our World in Data COVID-19 Dataset](https://covid.ourworldindata.org/data/owid-covid-data.csv).
