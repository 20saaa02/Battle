from prophet import Prophet
import pandas as pd

def prophet_forecast(series, horizon):
    df = pd.DataFrame({'ds': series.index, 'y': series.values})
    model = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False)
    model.add_seasonality(
    name='custom_season',
    period=12,             # сезонность (например, 12 месяцев)
    fourier_order=5)
    model.fit(df)
    future = model.make_future_dataframe(periods=horizon, freq='D')
    forecast = model.predict(future)
    return pd.Series(forecast['yhat'][-horizon:].reset_index(drop=True)), model