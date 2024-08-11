import sys
import os

sys.path.append(os.path.abspath(os.path.join('..', 'src')))


from data_processing import read_dataframe
from data_processing import convert_to_timeseries
import pandas as pd
from deepdiff import DeepDiff

def test_read_dataframe():
    data_path = '../data/2024_test_data.parquet'
    actual_df = read_dataframe(data_path)
    expected_df = pd.read_parquet(data_path)

    actual_df_list_dicts = actual_df.to_dict('records')

    expected_df_list_dicts = expected_df.to_dict('records')

    assert actual_df_list_dicts == expected_df_list_dicts

    diff = DeepDiff(actual_df_list_dicts, expected_df_list_dicts)
    print(f'diff = {diff}')

    assert 'type_changes' not in diff

def test_convert_to_timeseries():
    data_path = '../data/2024_globus_gas_prices.parquet'
    actual_df = read_dataframe(data_path)
    actual_df = convert_to_timeseries(actual_df)

    expected_df = read_dataframe(data_path)
    expected_df['date'] = pd.to_datetime(expected_df['date'], utc=True)
    expected_df.set_index('date', inplace=True)

    actual_df_list_dicts = actual_df.to_dict('records')

    expected_df_list_dicts = expected_df.to_dict('records')

    assert actual_df_list_dicts == expected_df_list_dicts