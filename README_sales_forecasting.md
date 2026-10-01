# AI Sales Forecasting App

A time series forecasting tool that predicts future daily store sales so retail teams can plan inventory ahead of demand.

I built this as a proof of concept during my AI Strategy internship at Value Retail (The Bicester Collection), to support an AI forecasting strategy I pitched. It was selected for a pilot.

## What it does

1. **Upload** a CSV of past daily sales
2. **Clean:** removes blank rows and formatting errors
3. **Train:** a NeuralProphet model learns the long-term trend, yearly seasonality (holidays, peak seasons), and weekly cycles (busy weekends)
4. **Forecast:** predicts the next 300 days and shows an interactive chart of predicted vs. actual sales
5. **Score:** reports R² so users know how much to trust the forecast

## Why NeuralProphet

I first tried a hand-built regression model and an LSTM neural network. The regression was too rigid to capture overlapping weekly and yearly patterns, and the LSTM was hard to tune and hard to explain to a business team. NeuralProphet learns its weights automatically but breaks forecasts into readable components (trend, yearly, weekly), so non-technical users can see what's driving a prediction.

## Run it

```
pip install -r requirements.txt
streamlit run app.py
```

Then upload a CSV with two columns:

| ds | y |
|---|---|
| 2024-01-01 | 15000 |
| 2024-01-02 | 12500 |

`ds` is the date, `y` is that day's sales. Company data is not included in this repo.

## Built with

Python, NeuralProphet, pandas, scikit-learn, Streamlit
