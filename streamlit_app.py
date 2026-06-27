import warnings
warnings.filterwarnings("ignore")

import datetime 
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# --- CONFIGURATION & THEME ---
st.set_page_config(page_title="UAC Analytics Platform", layout="wide", page_icon="🏛️")

# Custom Colors
COLOR_PRIMARY = "#1F4E79"
COLOR_SUCCESS = "#28a745"
COLOR_WARNING = "#ffc107"
COLOR_DANGER = "#dc3545"
COLOR_BG = "#f8f9fa"

TARGET = 'Children in HHS Care'
FEATURES = [
    'Children apprehended and placed in CBP custody', 'Children in CBP custody',
    'Children transferred out of CBP custody', 'Children discharged from HHS Care',
    'Net_Pressure', 'Transfer_Discharge_Ratio', 'HHS_Daily_Change', 'HHS_Growth_Rate',
    'lag_1', 'lag_7', 'lag_14', 'lag_30',
    'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30',
    'rolling_std_7', 'rolling_std_14', 'rolling_std_30',
    'day_of_week', 'day_of_month', 'month', 'quarter', 'week_of_year', 'year'
]

MODELS = ['Gradient Boosting', 'XGBoost', 'Random Forest', 'ARIMA', 'SARIMA', 'Moving Average', 'Naïve Forecast']

# --- CACHED DATA & MODELING FUNCTIONS ---
@st.cache_data
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(how='all').reset_index(drop=True)
    df.columns = df.columns.str.strip().str.replace('*', '', regex=False)
    df['Date'] = pd.to_datetime(df['Date'])
    
    if df[TARGET].dtype == object:
        df[TARGET] = df[TARGET].astype(str).str.replace(',', '', regex=False).str.strip()
        df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
        
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    
    # Updated pandas forward-fill syntax
    return df.ffill().fillna(0)

@st.cache_resource
def train_and_predict(_train, _test, target_col, features, horizon=30):
    """Trains all models and caches results for fast multi-page switching."""
    results = {}
    X_train, y_train = _train[features], _train[target_col]
    X_test, y_test = _test[features], _test[target_col]
    
    # 1. Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5)
    gb.fit(X_train, y_train)
    results['Gradient Boosting'] = {'pred': gb.predict(X_test), 'model': gb}
    
    # 2. Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=5)
    rf.fit(X_train, y_train)
    results['Random Forest'] = {'pred': rf.predict(X_test), 'model': rf}
    
    # 3. XGBoost
    if XGBOOST_AVAILABLE:
        xgb = XGBRegressor(n_estimators=100, max_depth=5)
        xgb.fit(X_train, y_train)
        results['XGBoost'] = {'pred': xgb.predict(X_test), 'model': xgb}
        
    # 4. Naive Forecast
    results['Naïve Forecast'] = {'pred': np.full(len(_test), y_train.iloc[-1]), 'model': None}
    
    # 5. Moving Average (7-day)
    ma_val = y_train.tail(7).mean()
    results['Moving Average'] = {'pred': np.full(len(_test), ma_val), 'model': None}
    
    # 6. ARIMA (Simplified for speed)
    try:
        arima = ARIMA(y_train, order=(1,1,1)).fit()
        results['ARIMA'] = {'pred': arima.forecast(steps=len(_test)), 'model': arima}
    except:
        results['ARIMA'] = {'pred': np.full(len(_test), ma_val), 'model': None}

    # 7. SARIMA
    try:
        sarima = SARIMAX(y_train, order=(1,1,1), seasonal_order=(0,0,0,0)).fit(disp=False)
        results['SARIMA'] = {'pred': sarima.forecast(steps=len(_test)), 'model': sarima}
    except:
        results['SARIMA'] = {'pred': np.full(len(_test), ma_val), 'model': None}
        
    # Calculate Metrics
    for m in results:
        actual = y_test[:len(results[m]['pred'])]
        pred = results[m]['pred']
        mape = mean_absolute_percentage_error(actual, pred)
        results[m]['metrics'] = {
            'MAE': mean_absolute_error(actual, pred),
            'RMSE': np.sqrt(mean_squared_error(actual, pred)),
            'MAPE': mape,
            'Accuracy': max(0, (1 - mape) * 100)
        }
    return results

def create_plotly_ts(train, test, predictions, title, ci_lower=None, ci_upper=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train[TARGET], mode='lines', name='Historical Data', line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=test.index, y=test[TARGET], mode='lines', name='Actual', line=dict(color=COLOR_PRIMARY)))
    fig.add_trace(go.Scatter(x=test.index, y=predictions, mode='lines', name='Forecast', line=dict(color=COLOR_DANGER, dash='dash')))
    
    if ci_lower is not None and ci_upper is not None:
        fig.add_trace(go.Scatter(x=test.index.tolist() + test.index[::-1].tolist(),
                                 y=ci_upper.tolist() + ci_lower[::-1].tolist(),
                                 fill='toself', fillcolor='rgba(220,53,69,0.2)', line=dict(color='rgba(255,255,255,0)'),
                                 hoverinfo="skip", showlegend=True, name='Confidence Interval'))
    
    fig.update_layout(title=title, template="plotly_white", hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# --- MAIN APP ARCHITECTURE ---
st.sidebar.markdown(f"<h2 style='color: {COLOR_PRIMARY};'>UAC Analytics</h2>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload UAC Dataset", type=['csv'])

if uploaded_file:
    df = load_and_clean_data(uploaded_file)
    available_features = [f for f in FEATURES if f in df.columns]
    
    # Global Train/Test Split
    split_idx = int(len(df) * 0.8)
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]
    
    # Train all models once
    with st.spinner("Initializing models and compiling data..."):
        model_results = train_and_predict(train, test, TARGET, available_features)
    
    # Navigation
    page = st.sidebar.radio("Navigation", [
        "Executive Dashboard", "Forecast Explorer", "Model Comparison", 
        "Feature Importance", "Historical Trends", "Capacity Risk", "Download Reports"
    ])
    
    st.sidebar.markdown("---")
    
    # Calculate dates and lag
    today = datetime.date.today()
    last_data_date = df.index.max().date()
    data_lag = (today - last_data_date).days
    
    st.sidebar.info(
        f"📅 **Report Generated:** {today.strftime('%Y-%m-%d')}\n\n"
        f"🔄 **Data Last Updated:** {last_data_date.strftime('%Y-%m-%d')}\n\n"
        f"⏳ **Data Lag:** {data_lag} Days\n\n"
        f"📊 **Total Records:** {len(df):,}"
    )

    # -----------------------------------------
    # PAGE 1: EXECUTIVE DASHBOARD
    # -----------------------------------------
    if page == "Executive Dashboard":
        st.title("Executive Dashboard")
        default_model = 'Gradient Boosting'
        res = model_results[default_model]
        
        current_val = int(train[TARGET].iloc[-1])
        forecast_val = int(res['pred'][-1])
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Current HHS Care", f"{current_val:,}")
        col2.metric("Forecasted (End of Horizon)", f"{forecast_val:,}", delta=f"{forecast_val - current_val:,}", delta_color="inverse")
        col3.metric("Selected Model", default_model)
        col4.metric("Forecast Accuracy", f"{res['metrics']['Accuracy']:.1f}%")
        col5.metric("RMSE", f"{res['metrics']['RMSE']:,.0f}")
        
        st.plotly_chart(create_plotly_ts(train, test, res['pred'], "Executive Overview: HHS Capacity Forecast"), use_container_width=True)
        
        st.markdown("### Executive Summary")
        st.info(f"The primary forecast generated by **{default_model}** indicates a change of **{forecast_val - current_val:,}** children over the test horizon. The model operates with a baseline accuracy of **{res['metrics']['Accuracy']:.1f}%**.")

    # -----------------------------------------
    # PAGE 2: FORECAST EXPLORER
    # -----------------------------------------
    elif page == "Forecast Explorer":
        st.title("Forecast Explorer")
        
        col_sel1, col_sel2, col_sel3 = st.columns([1, 1, 2])
        selected_model = col_sel1.selectbox("Select Model", MODELS, index=MODELS.index('Gradient Boosting'))
        horizon = col_sel2.selectbox("Forecast Horizon", [7, 14, 30], index=2)
        
        res = model_results.get(selected_model, model_results['Gradient Boosting'])
        
        # Display metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('MAE', f"{res['metrics']['MAE']:.2f}")
        m2.metric('RMSE', f"{res['metrics']['RMSE']:.2f}")
        m3.metric('MAPE', f"{res['metrics']['MAPE']:.4f}")
        m4.metric('Confidence / Accuracy', f"{res['metrics']['Accuracy']:.2f}%")
        
        # Slice test for horizon
        h_test = test.head(horizon)
        h_pred = res['pred'][:horizon]
        
        st.plotly_chart(create_plotly_ts(train, h_test, h_pred, f"{selected_model} - {horizon} Day Forecast"), use_container_width=True)

    # -----------------------------------------
    # PAGE 3: MODEL COMPARISON
    # -----------------------------------------
    elif page == "Model Comparison":
        st.title("Model Comparison")
        
        comp_data = []
        for m, data in model_results.items():
            if 'metrics' in data:
                row = data['metrics'].copy()
                row['Model'] = m
                comp_data.append(row)
                
        comp_df = pd.DataFrame(comp_data).sort_values(by='Accuracy', ascending=False)
        
        st.markdown(f"**Best Performing Model:** :green[{comp_df.iloc[0]['Model']}]")
        st.dataframe(comp_df.set_index('Model').style.highlight_max(subset=['Accuracy'], color='lightgreen').highlight_min(subset=['RMSE', 'MAE', 'MAPE'], color='lightgreen'), use_container_width=True)
        
        c1, c2 = st.columns(2)
        fig_acc = px.bar(comp_df, x='Model', y='Accuracy', title='Accuracy Comparison', color='Accuracy', color_continuous_scale='Blues')
        c1.plotly_chart(fig_acc, use_container_width=True)
        
        fig_rmse = px.bar(comp_df, x='Model', y='RMSE', title='RMSE Comparison', color='RMSE', color_continuous_scale='Reds')
        c2.plotly_chart(fig_rmse, use_container_width=True)

    # -----------------------------------------
    # PAGE 4: FEATURE IMPORTANCE
    # -----------------------------------------
    elif page == "Feature Importance":
        st.title("Feature Importance")
        fi_models = ['Gradient Boosting', 'Random Forest', 'XGBoost']
        sel_fi = st.selectbox("Select Model for Interpretation", [m for m in fi_models if m in model_results])
        
        model_obj = model_results[sel_fi]['model']
        if hasattr(model_obj, 'feature_importances_'):
            fi_df = pd.DataFrame({'Feature': available_features, 'Importance': model_obj.feature_importances_})
            fi_df = fi_df.sort_values(by='Importance', ascending=True).tail(10)
            
            fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title=f'Top 10 Drivers - {sel_fi}')
            fig.update_layout(template="plotly_white")
            fig.update_traces(marker_color=COLOR_PRIMARY)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("**Interpretation:** The features listed at the top have the highest mathematical weight in determining the final forecast output.")
        else:
            st.warning("Feature importance is not available for this model type.")

    # -----------------------------------------
    # PAGE 5: HISTORICAL TRENDS
    # -----------------------------------------
    elif page == "Historical Trends":
        st.title("Historical Trends")
        sel_feat = st.selectbox("Select Metric", available_features, index=available_features.index(TARGET) if TARGET in available_features else 0)
        roll_window = st.slider("Rolling Average Window (Days)", 1, 60, 7)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[sel_feat], mode='lines', name=sel_feat, line=dict(color='lightgray')))
        fig.add_trace(go.Scatter(x=df.index, y=df[sel_feat].rolling(roll_window).mean(), mode='lines', name=f'{roll_window}-Day Avg', line=dict(color=COLOR_PRIMARY, width=2)))
        fig.update_layout(title=f"{sel_feat} Over Time", template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------
    # PAGE 6: CAPACITY RISK DASHBOARD
    # -----------------------------------------
    elif page == "Capacity Risk":
        st.title("Capacity Risk Dashboard")
        st.markdown("Decision-support matrix based on default Gradient Boosting operational forecasts.")
        
        current_pop = train[TARGET].iloc[-1]
        forecasted_pop = model_results['Gradient Boosting']['pred'][-1]
        pct_change = ((forecasted_pop - current_pop) / current_pop) * 100
        
        st.metric("Forecasted 30-Day Population Change", f"{pct_change:+.2f}%")
        
        if pct_change > 30:
            st.error(f"🔴 **HIGH RISK:** Population projected to increase by {pct_change:.1f}%.")
            st.markdown("- **Action:** Increase shelter capacity immediately.\n- **Action:** Scale healthcare and casework staffing.\n- **Action:** Prepare emergency placement resources.")
        elif pct_change > 15:
            st.warning(f"🟡 **WATCH:** Population projected to increase by {pct_change:.1f}%.")
            st.markdown("- **Action:** Audit current bed availability.\n- **Action:** Issue standby notices to reserve caseworkers.")
        else:
            st.success(f"🟢 **NORMAL:** Population stable (Projected change: {pct_change:.1f}%).")
            st.markdown("- **Action:** Maintain standard operations and regular monitoring.")

    # -----------------------------------------
    # PAGE 7: DOWNLOAD REPORTS
    # -----------------------------------------
    elif page == "Download Reports":
        st.title("Download Center")
        st.write("Export analytics and model artifacts.")
        
        # Prepare CSVs
        gb_res = pd.DataFrame({'Date': test.index, 'Actual': test[TARGET], 'Forecast': model_results['Gradient Boosting']['pred']})
        csv_forecast = gb_res.to_csv(index=False).encode('utf-8')
        
        st.download_button(label="Download Gradient Boosting Forecast (CSV)", data=csv_forecast, file_name='gb_forecast.csv', mime='text/csv')
        st.download_button(label="Download Cleaned Dataset (CSV)", data=df.to_csv().encode('utf-8'), file_name='cleaned_uac_data.csv', mime='text/csv')

else:
    # Landing Screen when no file is uploaded
    st.markdown(f"<h1 style='text-align: center; color: {COLOR_PRIMARY}; margin-top: 100px;'>UAC Custody & Care Analytics Platform</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Please upload your historical dataset via the sidebar to initialize the dashboard environment.</p>", unsafe_allow_html=True)