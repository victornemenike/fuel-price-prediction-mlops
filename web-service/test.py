import predict
import pandas as pd

data_path = '../data/2024_val_data.parquet'
data = pd.read_parquet(data_path)
recent_data = data[-100:]
num_forecast_steps = 48
mode = 'local'

pred = predict.predict(recent_data, num_forecast_steps, mode)
print(pred)
print(len(pred[1]))