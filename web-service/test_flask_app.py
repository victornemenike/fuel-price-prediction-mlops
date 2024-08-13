#pylint: disable=duplicate-code
import pandas as pd
import requests

data_path = '../data/2024_val_data.parquet'
data = pd.read_parquet(data_path)
recent_data = data[-100:]
num_forecast_steps = 2


# Convert DataFrame to dictionary
recent_data_dict = recent_data.to_dict(orient = 'records')

# Create the payload
payload = {
    'recent_data': recent_data_dict,
    'num_forecast_steps': num_forecast_steps,
}

url = 'http://127.0.0.1:9696/predict'

response = requests.post(url, json = payload)
print(response.json())