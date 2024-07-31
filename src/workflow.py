from prefect import flow, task
from data_collection import load_data

@task
def collect_data(root_directory: str, station_uuid: str):
    df = load_data(root_directory, station_uuid)
    print(f'The dataset has {len(df)} samples.')
    return df

@flow(log_prints= True)  
def ml_pipeline(root_directory: str = '../data/2024_prices/2024',
                      station_uuid: str = '28d2efc8-a5e6-47d6-9d37-230fbcefcf70'):
    df = collect_data(root_directory, station_uuid)

if __name__ == "__main__":
    ml_pipeline()

    
