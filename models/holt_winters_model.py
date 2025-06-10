from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd

def holt_winters_forecast(series, horizon):
    model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12)
    fit = model.fit()
    forecast = fit.forecast(horizon)
    return pd.Series(forecast), fit