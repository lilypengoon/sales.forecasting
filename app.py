# Upload a CSV with two columns: ds (date) and y (daily sales),
# and the app trains a model and forecasts the next 300 days.

import streamlit as st
import pandas as pd
from neuralprophet import NeuralProphet
from sklearn.metrics import r2_score

st.title("📈 Sales Forecasting App")
st.write("Upload your sales CSV file and get an interactive forecast!")

uploaded_file = st.file_uploader("Upload your `sales_raw.csv`", type="csv")

if uploaded_file:
    # 1. Load and clean the data
    df = pd.read_csv(uploaded_file)
    if "Unnamed: 0" in df.columns:          # drop the leftover index column Excel adds
        df.drop(columns=["Unnamed: 0"], inplace=True)
    df.dropna(subset=["ds", "y"], inplace=True)   # remove days with missing dates or sales
    df["ds"] = pd.to_datetime(df["ds"])           # make sure dates are real dates, not text
    df = df[["ds", "y"]]                          # keep only date + sales

    st.write("📊 Preview of your data", df.tail())

    # 2. Train the model
    with st.spinner("Training model..."):
        model = NeuralProphet(
            yearly_seasonality=True,   # learns patterns across the year (holidays, summer, etc.)
            weekly_seasonality=True,   # learns patterns across the week (weekends vs weekdays)
            learning_rate=1.0,
        )
        model.fit(df, freq="D", epochs=100)   # daily data, 100 passes over the history

        # 3. Forecast 300 days ahead (and include past fit for comparison)
        future = model.make_future_dataframe(df, periods=300, n_historic_predictions=True)
        forecast = model.predict(future)

    st.success("✅ Forecast Complete!")

    # 4. Interactive chart: actual sales (dots) vs model predictions (blue line)
    fig = model.plot(forecast)
    st.plotly_chart(fig, use_container_width=True)

    # 5. Score the model: how much of the day-to-day variation it explains
    eval_df = forecast[["y", "yhat1"]].dropna()
    r2 = r2_score(eval_df["y"], eval_df["yhat1"])
    st.write(f"R² Score (coefficient of determination): {r2:.4f}")
