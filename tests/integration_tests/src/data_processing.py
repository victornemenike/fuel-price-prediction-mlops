import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from utils import *
import matplotlib.pyplot as plt
import torch
from data_collection import *


def read_dataframe(path):
    data_path = path
    df = pd.read_parquet(data_path)
    return df

def convert_to_timeseries(df):
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df.set_index('date', inplace=True)
    return df


def resample_timeseries(df, col_name, sampling_freq):
    sampling_freq = sampling_freq
    df_resampled = df[col_name].resample(sampling_freq).mean().interpolate(method = "linear")
    df_resampled = pd.DataFrame(df_resampled)
    return df_resampled

def prepare_data(path, col_name, sampling_freq):
    df = read_dataframe(path)
    df = convert_to_timeseries(df)
    df_resampled = resample_timeseries(df, col_name, sampling_freq)
    return df_resampled


def prepare_X_y(name, data, sequence_length):

    dataset = data.values
    dataset = dataset[~np.isnan(dataset)]
    # reshaping 1D to 2D array
    dataset = np.reshape(dataset, (-1, 1))

    # Create sequences and labels for training data
    X, y = [], []
    for i in range(len(dataset) - sequence_length):
        X.append(dataset[i:i+sequence_length])
        y.append(dataset[i+1:i+sequence_length+1])
    X, y = np.array(X), np.array(y)

    # Convert data to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    print(f'The {name} set has {dataset.shape[0]} samples')

    return X, y
   



if __name__ == '__main__':
    file_name = '../data/2024_globus_gas_prices.parquet'
    col_name = 'e5'
    sampling_freq = '1h'
    data = prepare_data(file_name, col_name, sampling_freq)

    train_data = data[:'2024-05-31']
    train_data_path = '../data/2024_train_data.parquet'
    save_data(train_data, train_data_path)

    val_data = data['2024-06-01':'2024-07-13']
    val_data_path = '../data/2024_val_data.parquet'
    save_data(val_data, val_data_path)

    test_data = data['2024-07-14':]
    test_data_path = '../data/2024_test_data.parquet'
    save_data(test_data, test_data_path)
    
    
