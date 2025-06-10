from tbats import TBATS
import pandas as pd

def tbats_forecast(series, horizon):
    estimator = TBATS(seasonal_periods=[7])
    model = estimator.fit(series)
    forecast = model.forecast(steps=horizon)
    return pd.Series(forecast), model