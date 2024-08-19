# pylint: disable=trailing-whitespace
import os

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from prefect import flow, task

#from data_collection import load_data
from data_processing import (
    read_dataframe,
    convert_to_timeseries,
    prepare_X_y,
    resample_timeseries
)
from model_registry import register_model, transition_model
from plotting import plot_learning_curve
from train import create_dataloader, save_model, train_model


@task
def load_data(file_path:str):
    df = read_dataframe(file_path)
    print(f'The dataset has {len(df)} samples.')
    return df

@task
def preprocess_data(df: pd.DataFrame):
    col_name = 'e5'
    df = convert_to_timeseries(df)
    sampling_freq = '1h'
    df_resampled = resample_timeseries(df, col_name, sampling_freq)
    return df_resampled


@task
def split_data(df: pd.DataFrame):
    df_train = df[:'2024-05-31']
    df_val = df['2024-06-01':'2024-07-13']
    df_test = df['2024-07-14':]
    return df_train, df_val, df_test


@task
def prepare_X_y_train_val(df_train: pd.DataFrame, df_val: pd.DataFrame):
    X_train, y_train = prepare_X_y('training', df_train, sequence_length=48)
    X_val, y_val = prepare_X_y('training', df_val, sequence_length=24)

    return X_train, y_train, X_val, y_val


@task
def create_torch_dataloader(
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size: int = 16,
):

    train_loader, val_loader = create_dataloader(
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size,
    )

    return train_loader, val_loader


@task
def train_torch_model(
    train_loader, val_loader, num_epochs: int = 50, learning_rate: float = 1e-3
):
    data_loader = {'train_loader': train_loader, 'val_loader': val_loader}
    model, run, train_hist, val_hist = train_model(
        data_loader,
        num_epochs,
        learning_rate,
    )

    return model, run, train_hist, val_hist, num_epochs


@task
def save_model_locally(model):
    model_dir = "models"
    model_format = "pickle"
    save_model(model, model_dir, model_format)


@task
def mlflow_registry(client, run, model_name: str = "fuel-price-predictor"):

    register_model(run.info.run_id, model_name)
    print('Model registered in MLflow')

    latest_versions = client.get_latest_versions(name=model_name)
    version = latest_versions[-1].version
    stage = "Production"
    transition_model(client, model_name, version, stage)

    print(f'The model version {version} was transitioned to {stage}')


@flow(log_prints=True)
def ml_pipeline(data_path: str, MLFLOW_mode: str = 'local'):

    df = load_data(data_path)

    df_processed = preprocess_data(df)

    df_train, df_val, _ = split_data(df_processed)

    X_train, y_train, X_val, y_val = prepare_X_y_train_val(df_train, df_val)

    train_loader, val_loader = create_torch_dataloader(X_train, y_train, X_val, y_val)

    if MLFLOW_mode == 'local':
        MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if MLFLOW_mode == 'aws':
        TRACKING_SERVER_HOST = "ec2-3-255-95-173.eu-west-1.compute.amazonaws.com"
        MLFLOW_TRACKING_URI = f"http://{TRACKING_SERVER_HOST}:5000"
        os.environ["AWS_PROFILE"] = "default"
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    MLFLOW_EXPERIMENT_NAME = "fuel-price-experiment"
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    model, run, train_hist, val_hist, num_epochs = train_torch_model(
        train_loader, val_loader
    )

    save_model_locally(model)

    print(f'Most recent MLflow run id: {run.info.run_id}')
    mlflow_registry(client, run)

    plot_learning_curve(num_epochs, train_hist, val_hist)


if __name__ == "__main__":
    data_directory = 'data/2024_globus_gas_prices.parquet'
    mlflow_mode = 'local'
    ml_pipeline(data_directory, mlflow_mode)
