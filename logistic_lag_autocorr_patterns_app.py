
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

st.set_page_config(page_title="GOOG Lag Prediction", layout="wide")
st.title("📈 GOOG Lag Prediction with Time Series Insights")

# 📷 Quick preview of expected CSV format
from PIL import Image
image = Image.open("Goog.JPG")
st.image(image, caption="CSV Format: Stocks, SP500", use_container_width=True, output_format="JPEG")

with st.sidebar:
    st.header("1. Upload Files")
    goog_file = st.file_uploader("GOOG CSV (semicolon-separated)", type="csv")
    sp500_file = st.file_uploader("S&P500 CSV (semicolon-separated)", type="csv")

    st.header("2. Model Options")
    lags = st.multiselect("Select lags (days)", [1, 2, 3, 5, 10], default=[1, 2, 5])
    model_type = st.selectbox("Model", ["Logistic Regression", "Decision Tree", "Random Forest"])
    splits = st.slider("Time Series Splits", min_value=3, max_value=10, value=5)
    run_button = st.button("Run Model")

def load_data(gfile, sfile):
    goog = pd.read_csv(gfile, sep=";")[["Date", "Adj.Close"]].rename(columns={"Adj.Close": "goog_price"})
    sp500 = pd.read_csv(sfile, sep=";")[["Date", "Adj.Close"]].rename(columns={"Adj.Close": "sp_price"})
    for df in [goog, sp500]:
        for col in df.columns:
            if col != 'Date':
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

    df = pd.merge(goog, sp500, on="Date")
    df["Date"] = pd.to_datetime(df["Date"], format='%d/%m/%Y')
    df = df.sort_values("Date", ascending=False).reset_index(drop=True)

    df["goog_ret"] = df["goog_price"].pct_change()
    df["sp_ret"] = df["sp_price"].pct_change()
    return df.dropna().reset_index(drop=True)

def prepare_features(df, lags):
    df = df.copy()
    for lag in lags:
        df[f"goog_lag{lag}"] = df["goog_ret"].shift(-lag)
        df[f"sp_lag{lag}"] = df["sp_ret"].shift(-lag)
    df["goog_up"] = (df["goog_ret"] >= 0).astype(int)
    df = df.dropna().reset_index(drop=True)
    features = [f"goog_lag{lag}" for lag in lags] + [f"sp_lag{lag}" for lag in lags]
    return df, features

def select_model(name):
    if name == "Logistic Regression":
        return LogisticRegression()
    elif name == "Decision Tree":
        return DecisionTreeClassifier()
    elif name == "Random Forest":
        return RandomForestClassifier()

if run_button and goog_file and sp500_file:
    df_raw = load_data(goog_file, sp500_file)

    st.header("🔍 Time Series Pattern Detection")

    # --- Mean Reversion: ADF Test ---
    st.subheader("📉 Mean Reversion (ADF Test)")
    adf_result = adfuller(df_raw["goog_ret"].dropna())
    st.write(f"ADF Statistic: {adf_result[0]:.3f}")
    st.write(f"p-value: {adf_result[1]:.3f}")
    if adf_result[1] < 0.05:
        st.success("Likely Mean Reverting (stationary series)")
    else:
        st.warning("Not Mean Reverting (non-stationary series)")

    # --- Momentum: Autocorrelation ---
    st.subheader("⚡ Momentum (Autocorrelation)")
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(df_raw["goog_ret"].dropna(), lags=20, ax=ax)
    st.pyplot(fig)

    # --- Seasonality by Weekday/Month ---
    st.subheader("📅 Seasonality Patterns")

    df_raw["Weekday"] = df_raw["Date"].dt.dayofweek
    df_raw["Month"] = df_raw["Date"].dt.month

    col1, col2 = st.columns(2)

    with col1:
        weekday_avg = df_raw.groupby("Weekday")["goog_ret"].mean()
        weekday_avg.plot(kind="bar", title="Average Return by Weekday")
        st.pyplot(plt.gcf())

    with col2:
        month_avg = df_raw.groupby("Month")["goog_ret"].mean()
        month_avg.plot(kind="bar", title="Average Return by Month")
        st.pyplot(plt.gcf())

    df_lagged, features = prepare_features(df_raw, lags)
    X = df_lagged[features].values
    y = df_lagged["goog_up"].values

    tscv = TimeSeriesSplit(n_splits=splits)
    model = select_model(model_type)

    metrics = []
    preds_df = pd.DataFrame()

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        metrics.append({
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_proba)
        })

        temp = pd.DataFrame({
            "Actual": y_test,
            "Predicted": y_pred,
            "Probability": y_proba
        })
        preds_df = pd.concat([preds_df, temp], ignore_index=True)

    df_metrics = pd.DataFrame(metrics).mean().round(3)

    tab1, tab2, tab3 = st.tabs(["📊 Metrics", "📉 ROC Curve", "📌 Feature Importance"])

    with tab1:
        st.subheader("Average Cross-Validation Metrics")
        st.write(df_metrics.T)
        st.dataframe(preds_df.head(10))
        csv = preds_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions", data=csv, file_name="cv_predictions.csv")

    with tab2:
        st.subheader("Mean ROC Curve")
        model.fit(X, y)
        proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
        fpr, tpr, _ = roc_curve(y, proba)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, proba):.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        st.pyplot(plt.gcf())

    with tab3:
        if hasattr(model, "feature_importances_"):
            st.subheader("Feature Importances")
            fi = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
            fi = fi.sort_values("Importance", ascending=False)
            plt.figure(figsize=(8, 4))
            sns.barplot(x="Importance", y="Feature", data=fi)
            plt.title("Feature Importance")
            st.pyplot(plt.gcf())
        else:
            st.info("Feature importance not available for this model.")
else:
    st.info("⬅️ Upload files, choose settings, and run the model.")
