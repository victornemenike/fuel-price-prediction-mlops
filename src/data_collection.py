import pandas as pd
import os

def load_data(root_directory, station_uuid):
    all_dataframes = []

    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                print(f"processing: {file_path}")

                # read the csv file
                df = pd.read_csv(file_path)
                df = df[df.station_uuid == station_uuid]

                # Optional: add a column to identify the source file or subfolder
                df['source'] = os.path.relpath(file_path, root_directory)

                # Append  the dataframe to the list
                all_dataframes.append(df)

    # Combine all dataframes into a single dataframe
    combined_df = pd.concat(all_dataframes, ignore_index= True)

    print(combined_df.head())
    print(f'Total rows: {len(combined_df)}')


    return combined_df

def save_data(df, file_name):
    print(f'saving to file: {file_name}')
    df.to_parquet(file_name)




if __name__ == '__main__':
    root_directory = '../data/2024_prices/2024'
    station_uuid = '28d2efc8-a5e6-47d6-9d37-230fbcefcf70'
    file_name = '../data/2024_globus_gas_prices.parquet'

    df = load_data(root_directory, station_uuid)
    
    save_data(df, file_name)
    
