# COVID-19_Project(view) :- https://covid19dataset-qpjozfwf2memgbu2qiut9t.streamlit.app/

## COVID-19 Prediction Project

This project applies machine learning models to **predict COVID-19 case growth** using data from Worldometer. The workflow includes data preprocessing, feature engineering, scaling, and applying regression models (Linear Regression, Random Forest, XGBoost, ARIMA, and LSTM).

---

## 📂 Dataset

* Source: [Worldometer COVID-19 Data](https://www.worldometers.info/coronavirus/)
* Contains country-wise COVID-19 statistics:

  * Total Cases, Deaths, Recovered
  * Daily New Cases/Deaths
  * Population, Tests, Region
* Preprocessing:

  * Removed irrelevant/non-predictive columns (`Country/Region`, `WHO Region`, `Continent`, etc.).
  * Applied log-transformation (`Total_case_log`) for stability.
  * Scaled features using **MinMaxScaler** for models sensitive to feature ranges.

---

## 🛠 Libraries Used

* **Data Handling & Analysis**

  * `pandas`, `numpy`
* **Visualization**

  * `matplotlib`, `seaborn`
* **Machine Learning**

  * `scikit-learn` (Linear Regression, Random Forest, metrics)
  * `xgboost` (XGBRegressor)
* **Time Series**

  * `statsmodels` (ARIMA)
  * `tensorflow.keras` (LSTM)
* **Utilities**

  * `warnings`

---

## 📊 Models Implemented

### 1. **Linear Regression**

* Baseline model for regression.
* Captures linear relationships but underfits complex COVID-19 data.

### 2. **Random Forest Regressor**

* Parameters: `n_estimators=200`, `max_depth=10`, `random_state=42`
* Strengths: Handles non-linearity, robust to noise.
* Limitation: Less effective for long-term forecasting.

### 3. **XGBoost Regressor**

* Parameters: `n_estimators=100`, `learning_rate=0.1`, `max_depth=5`
* High performance on structured/tabular data.
* Often outperforms Random Forest with better generalization.

### 4. **ARIMA (Auto-Regressive Integrated Moving Average)**

* Captures temporal dependencies in time-series data.
* Used on India-specific cases for trend forecasting.

### 5. **LSTM (Long Short-Term Memory)**

* Neural network designed for sequential/time-series data.
* Applied on COVID-19 daily cases for deep temporal learning.

---

## 📈 Evaluation Metrics

* **R² (Coefficient of Determination):** Measures goodness of fit.
* **MSE (Mean Squared Error):** Penalizes large errors.
* **RMSE (Root Mean Squared Error):** Scaled version of MSE.
* **MAPE (Mean Absolute Percentage Error):** Measures relative error in percentages.

---

## ✅ Results (Sample)

| Model                   | RMSE                                                   | R² Score |
| ----------------------- | ------------------------------------------------------ | -------- |
| Linear Regression       | ~0.52                                                  | ~0.60    |
| Random Forest Regressor | ~0.31                                                  | ~0.85    |
| XGBoost Regressor       | ~0.28                                                  | ~0.88    |
| ARIMA                   | Good short-term forecasting, weaker long-term          |          |
| LSTM                    | Strong sequence modeling, needs more data for accuracy |          |

*(Exact scores may vary depending on dataset updates and preprocessing choices.)*

---

## 📌 Key Insights

* **XGBoost** gave the best performance among classical ML models.
* **LSTM** showed potential for time-series forecasting but required more tuning and larger datasets.
* **ARIMA** worked well for short-term trends in single-country analysis.
* Feature scaling was crucial for gradient-based models.

---

## 🚀 Future Improvements

* Hyperparameter tuning (GridSearchCV / RandomizedSearchCV).
* Incorporating external data (mobility, vaccination rates, policies).
* Deploying the model via Flask/Django REST API.
* Automating daily predictions with scheduled data updates.

---

## 🖼 Visualizations

* Country-wise COVID-19 case distributions.
* Predicted vs Actual plots for each model.
* Time-series forecasting curves for ARIMA and LSTM.

---

## 📌 How to Run

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/covid19-prediction.git
   cd covid19-prediction
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run Jupyter Notebook:

   ```bash
   jupyter notebook covid1.ipynb
   ```

---

## 📜 License

This project is licensed under the MIT License.
