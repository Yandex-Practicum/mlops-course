# run.py

import pandas as pd
from preprocessing import preprocess_data

# Исходные данные, которые мы не хотим менять
original_data = pd.DataFrame(
    {
        "fare_amount": [10.0, 20.0],
        "passenger_count": [1, 2],
        "pickup_longitude": [-73.9, -73.9],
        "pickup_latitude": [40.7, 40.7],
        "dropoff_longitude": [-74.0, -74.0],
        "dropoff_latitude": [40.8, 40.8],
    }
)
print("Original data before processing:\n", original_data.columns)

# Вызываем функцию предобработки
processed_data = preprocess_data(original_data)

# Проверяем, что случилось с исходными данными
print("\nOriginal data after processing:\n", original_data.columns)
print(original_data)

