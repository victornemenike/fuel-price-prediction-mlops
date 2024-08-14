# pylint: disable=duplicate-code
import json

import pandas as pd
import requests
from deepdiff import DeepDiff

data_path = "2024_val_data.parquet"
data = pd.read_parquet(data_path)
recent_data = data[-100:]
num_forecast_steps = 0


# Convert DataFrame to dictionary
recent_data_dict = recent_data.to_dict(orient="records")

# Create the payload
payload = {
    "recent_data": recent_data_dict,
    "num_forecast_steps": num_forecast_steps,
}

url = "http://127.0.0.1:8080/predict"

actual_response = requests.post(url, json=payload)
actual_response = actual_response.json()

print(json.dumps(actual_response, indent=2))

expected_response = {
    "forecasted_prices": [
        1.8297425508499146,
        1.8407069444656372,
        1.8516478538513184,
        1.862565279006958,
        1.8734580278396606,
        1.8187555074691772,
        1.8279130458831787,
        1.8095823526382446,
        1.8049904108047485,
        1.7950280904769897,
        1.7850486040115356,
        1.7696630954742432,
        1.7727432250976562,
        1.7758218050003052,
        1.7727432250976562,
        1.7704333066940308,
        1.7623416185379028,
        1.7542399168014526,
        1.7614421844482422,
        1.7686359882354736,
        1.7758218050003052,
        1.7742828130722046,
        1.7727432250976562,
        1.7634981870651245,
    ],
    "predicted_steps": 24,
}

diff = DeepDiff(actual_response, expected_response, significant_digits=1)
print(diff)

assert actual_response == expected_response

assert "type_changes" not in diff

assert "values_changed" not in diff
