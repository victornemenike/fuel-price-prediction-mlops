import sys
import os

sys.path.append(os.path.abspath(os.path.join('..', 'src')))


from data_processing import read_dataframe
import pandas as pd

def test_read_dataframe():
    data_path = '../data/2024_test_data.parquet'
    actual_df = read_dataframe(data_path)
    expected_df = pd.read_parquet(data_path)

    actual_df_list_dicts = actual_df.to_dict('records')

    expected_df_list_dicts = expected_df.to_dict('records')

    assert actual_df_list_dicts == expected_df_list_dicts