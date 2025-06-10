import pandas as pd

def ensemble_forecast(forecasts_dict):
    # forecasts_dict: {'arima': Series, 'prophet': Series, ...}
    df = pd.DataFrame(forecasts_dict)
    return df.mean(axis=1) 