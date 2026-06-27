# Care-Transition-Efficiency-Placement-Outcome-Forecasting
# 📊 Predictive Forecasting of Care Load & Placement Demand
### HHS Unaccompanied Alien Children (UAC) Program

## Overview

This project develops an end-to-end predictive analytics framework for forecasting care demand within the U.S. Department of Health and Human Services (HHS) Unaccompanied Alien Children (UAC) Program.

The objective is to provide short-term forecasts of children in HHS care, estimate discharge demand, and support proactive resource planning through statistical and machine learning forecasting techniques.

Unlike descriptive dashboards that only summarize historical trends, this project focuses on predictive intelligence for operational decision-making.

---

## Project Objectives

### Primary Objectives

- Forecast future HHS care load
- Predict discharge demand
- Estimate future care capacity
- Compare statistical and machine learning forecasting methods

### Secondary Objectives

- Identify operational bottlenecks
- Provide early warning indicators
- Support healthcare resource allocation
- Improve child welfare planning

---

# Dataset

The dataset contains daily operational data from the HHS Unaccompanied Alien Children Program.

Variables include:

| Variable | Description |
|-----------|-------------|
| Date | Reporting date |
| Children apprehended and placed in CBP custody | Daily intake |
| Children in CBP custody | Active CBP population |
| Children transferred out of CBP custody | Transfers to HHS |
| Children in HHS Care | Active HHS population |
| Children discharged from HHS Care | Sponsor reunifications |

---

# Project Workflow

```
Data Cleaning
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Statistical Forecasting
        │
        ▼
Machine Learning Forecasting
        │
        ▼
Model Comparison
        │
        ▼
Interactive Streamlit Dashboard
```

---

# Feature Engineering

The following predictive features were created:

- Lag Features
  - Lag 1
  - Lag 7
  - Lag 14
  - Lag 30

- Rolling Statistics
  - 7-day Rolling Mean
  - 14-day Rolling Mean
  - 30-day Rolling Mean
  - Rolling Standard Deviations

- Flow-Based Features
  - Net Pressure
  - Transfer / Discharge Ratio

- Calendar Features
  - Day of Week
  - Month
  - Quarter
  - Week of Year

- Cyclical Features
  - Sine/Cosine transformations
  - Monthly seasonality encoding

---

# Forecasting Models

## Baseline Models

- Naïve Forecast
- Moving Average

---

## Statistical Models

- Exponential Smoothing
- ARIMA
- SARIMA

---

## Machine Learning Models

- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

Hyperparameter tuning was performed using RandomizedSearchCV.

---

# Model Performance

| Rank | Model |
|------|-----------------------------|
| 🥇 | Naïve Forecast |
| 🥈 | Moving Average |
| 🥉 | Gradient Boosting |
| 4 | XGBoost |
| 5 | Random Forest |
| 6 | ARIMA |
| 7 | SARIMA |
| 8 | Exponential Smoothing |

The results demonstrate that while persistence-based forecasting performs exceptionally well for this operational dataset, Gradient Boosting provides the strongest advanced predictive model by leveraging engineered temporal and operational features.

---

# Feature Importance

The most influential predictors were:

- Lag 1
- Rolling Mean (7-day)
- Rolling Mean (14-day)
- Lag 14
- Children Transferred out of CBP Custody
- Children Discharged from HHS Care

These findings indicate that recent care load history is the strongest determinant of future HHS care demand.

---

# Streamlit Dashboard

The project includes a fully interactive Streamlit dashboard featuring:

- Executive Dashboard
- Forecast Explorer
- Model Comparison
- Feature Importance
- Historical Trends
- Capacity Risk Assessment
- Downloadable Reports

Interactive visualizations were developed using Plotly.

---

# Technologies Used

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn
- Plotly

### Statistical Forecasting

- Statsmodels
- pmdarima

### Machine Learning

- Scikit-learn
- XGBoost

### Dashboard

- Streamlit

---

# Project Structure

```
UAC_Forecasting/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── forecasting.ipynb
│
├── models/
│
├── outputs/
│   ├── figures/
│   ├── reports/
│   └── forecasts/
│
├── streamlit_app.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/UAC_Forecasting.git
```

Navigate into the project

```bash
cd UAC_Forecasting
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the dashboard

```bash
streamlit run streamlit_app.py
```

---

# Key Findings

- Historical HHS care load exhibits strong persistence.
- Lag-based temporal features were the strongest predictors.
- Gradient Boosting outperformed all advanced forecasting models.
- Statistical forecasting models were less effective than tree-based machine learning methods for this dataset.
- Predictive analytics can support proactive resource allocation and operational planning within the UAC Program.

---

# Future Improvements

- Walk-forward cross-validation
- SHAP explainability
- LightGBM implementation
- LSTM and Transformer forecasting
- Probabilistic forecasting
- Automated retraining pipeline
- Real-time API integration
- Cloud deployment

---

# Author

**Mohit**

Pharm D Student | Data Science Intern

Python • Machine Learning • Forecasting • Healthcare Analytics • Business Intelligence

---

# License

This project is licensed under the MIT License.
