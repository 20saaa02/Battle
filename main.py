import pandas as pd
import argparse
from models.arima_model import arima_forecast
from models.holt_winters_model import holt_winters_forecast
from models.prophet_model import prophet_forecast
from models.tbats_model import tbats_forecast
from models.ensemble_model import ensemble_forecast
from utils.preprocessing import preprocess_series

def main():
    i = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV')
    parser.add_argument('--horizon', type=int, required=True, help='Forecast horizon')
    parser.add_argument('--models', type=str, default='arima,holt,prophet,tbats,ensemble',
    help='Comma-separated list of models to run')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    series = preprocess_series(df)

    model_list = [m.strip().lower() for m in args.models.split(',')]
    forecasts = {}
    trained_models = {}

    for model_name in model_list:
        print(f"Running model: {model_name}")
        if model_name == 'arima':
            forecast, model = arima_forecast(series, args.horizon)
            forecasts['arima'] = forecast
            trained_models['arima'] = model
        elif model_name == 'holt':
            forecast, model = holt_winters_forecast(series, args.horizon)
            forecasts['holt'] = forecast
            trained_models['holt'] = model
        elif model_name == 'prophet':
            forecast, model = prophet_forecast(series, args.horizon)
            forecasts['prophet'] = forecast
            trained_models['prophet'] = model
        elif model_name == 'tbats':
            forecast, model = tbats_forecast(series, args.horizon)
            forecasts['tbats'] = forecast
            trained_models['tbats'] = model
        elif model_name == 'ensemble':
            continue
        else:
            print(f"⚠️ Unknown model: {model_name}")

    for name, forecast in forecasts.items():
        forecast.index = range(1, len(forecast) + 1) 

    if 'ensemble' in model_list:
        print("Running ensemble model")
        ensemble = ensemble_forecast(forecasts)
        forecasts['ensemble'] = ensemble



    forecast_df = pd.DataFrame(forecasts)
    forecast_df.to_csv(f'forecast{i}.csv', index=False)
    print("✅ Forecast saved to forecast.csv")

if __name__ == "__main__":
    main()
