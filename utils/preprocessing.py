import pandas as pd

def preprocess_series(df):
    if df.shape[1] == 1:
        series = df.iloc[:, 0]
        series.index = pd.date_range(start='2000-01-01', periods=len(series), freq='D')
    elif 'date' in df.columns and 'value' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        series = df['value']
    else:
        raise ValueError("Unknown format")
    return series