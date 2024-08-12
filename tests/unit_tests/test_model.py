import sys
import os

sys.path.append(os.path.abspath(os.path.join('..', 'src')))


from model_output import *
from predict import local_model_forecast
import pandas as pd
import pickle


def test_model_forecast():
    data_path = '../data/2024_val_data.parquet'
    data = pd.read_parquet(data_path)
    recent_data = data[-100:]
    num_forecast_steps = 0

    model_docker_path = '../models/fuel_price_lstm.pickle' 
    with open(model_docker_path, 'rb') as f_in:
        model = pickle.load(f_in)

    prediction_horizon, _, forecasted_values, _, _ = local_model_forecast(model, 
                                                              '', 
                                                     recent_data, 
                                                     num_forecast_steps)
    actual_horizon = prediction_horizon
    actual_forecast = forecasted_values

    expected_horizon = model_output['predicted_steps']
    expected_forecast = model_output['forecasted_prices']
    
    assert actual_horizon == expected_horizon
    assert actual_forecast == expected_forecast