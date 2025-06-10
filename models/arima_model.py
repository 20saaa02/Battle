from pmdarima import auto_arima
import pandas as pd

def arima_forecast(series, horizon):
    print(len(series))
    model = auto_arima(series, seasonal=True, m=7, suppress_warnings=True)
    forecast = model.predict(n_periods=horizon)
    return pd.Series(forecast), model
