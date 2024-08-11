import sys
import os

# Correcting the typo from os.path.json to os.path.join
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

import datetime
import time
import random
import logging
import uuid
import pytz
import pandas as pd
import io
import psycopg
import mlflow
import numpy as np
from predict import forecast

from prefect import task, flow

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric


logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
DROP TABLE IF EXISTS dummy_metrics;
CREATE TABLE dummy_metrics(
	timestamp TIMESTAMP,
	prediction_drift FLOAT,
	num_drifted_columns INTEGER,
	share_missing_values FLOAT
)
"""

reference_data = pd.read_parquet('../data/monitoring-reference.parquet')

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("fuel-price-experiment")

run_id = '337ff4b11daf4118a3c9a64263073c4b'
logged_model = f'runs:/{run_id}/LSTM-model'
loaded_model = mlflow.pyfunc.load_model(logged_model)

raw_data = pd.read_parquet('../data/2024_test_data.parquet')
begin = datetime.datetime(2024,8, 11, 0, 0)
begin_utc = begin.replace(tzinfo=pytz.UTC)

column_mapping = ColumnMapping(
    prediction='e5_forecasted',
    target = None
)

report = Report(metrics = [
    ColumnDriftMetric(column_name='e5_forecasted'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric()
])

@task
def prep_db():
	with psycopg.connect("host=localhost port=5432 user=postgres password=example", 
                      autocommit=True) as conn:
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
		if len(res.fetchall()) == 0:
			conn.execute("create database test;")
		with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
			conn.execute(create_table_statement)

@task
def calculate_metrics_postgresql(curr, i):
	sequence_length= 48
	current_data = raw_data.iloc[i:sequence_length+i]
	data_name = 'current data'
	num_forecast_steps = 24
	_, _, forecasted_values, combined_index, sequence_to_plot = forecast(loaded_model,
                                                                            data_name, 
                                                                            current_data, 
                                                                            num_forecast_steps)
	original_cases = np.expand_dims(sequence_to_plot[-1], axis=0).flatten()
	last_npoints = 24
	monitored_data = pd.DataFrame({'timestamp': combined_index[-last_npoints:],
                         'e5': original_cases,
                         'e5_forecasted': forecasted_values[-last_npoints:]})
	report.run(reference_data=reference_data, current_data=monitored_data,
               column_mapping=column_mapping)
	result = report.as_dict()
	prediction_drift = result['metrics'][0]['result']['drift_score']
	num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
	share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']
	curr.execute(
		"insert into dummy_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values) values (%s, %s, %s, %s)",
		(begin + datetime.timedelta(hours=i), prediction_drift, num_drifted_columns, share_missing_values)
	)

@flow
def batch_monitoring_backfill():
	prep_db()
	last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
	with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", 
					  autocommit=True) as conn:
		for i in range(0, 27):
			with conn.cursor() as curr:
				calculate_metrics_postgresql(curr, i)

			new_send = datetime.datetime.now()
			seconds_elapsed = (new_send - last_send).total_seconds()
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			while last_send < new_send:
				last_send = last_send + datetime.timedelta(seconds=10)
			logging.info("data sent")


if __name__ == "__main__":
    batch_monitoring_backfill()

