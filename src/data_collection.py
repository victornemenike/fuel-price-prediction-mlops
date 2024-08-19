# pylint: disable=missing-module-docstring
import os

import pandas as pd


def load_data(data_dir, station_id):
    all_dataframes = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                print(f"processing: {file_path}")

                # read the csv file
                dataframe = pd.read_csv(file_path)
                dataframe = dataframe[dataframe.station_uuid == station_id]

                # Optional: add a column to identify the source file or subfolder
                dataframe['source'] = os.path.relpath(file_path, data_dir)

                # Append  the dataframe to the list
                all_dataframes.append(dataframe)

    # Combine all dataframes into a single dataframe
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    print(combined_df.head())
    print(f'Total rows: {len(combined_df)}')

    return combined_df


def save_data(dataframe, file_path):
    print(f'saving to file: {file_path}')
    dataframe.to_parquet(file_path)


if __name__ == '__main__':
    root_directory = 'data/2024_prices/2024'
    station_uuid = '28d2efc8-a5e6-47d6-9d37-230fbcefcf70'
    file_name = 'data/2024_globus_gas_prices.parquet'

    df = load_data(root_directory, station_uuid)
    save_data(df, file_name)
