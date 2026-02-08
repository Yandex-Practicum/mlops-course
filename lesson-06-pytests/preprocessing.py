# src/preprocessing.py
import pandas as pd
import numpy as np


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Выполняет предобработку данных о поездках Uber."""
    # 1. Фильтрация по стоимости и пассажирам
    mask = (
        (df["fare_amount"] > 0)
        & (df["passenger_count"] > 0)
        & (df["passenger_count"] <= 6)
    )

    df.drop(index=df.loc[~mask].index, inplace=True)

    # 2. Создание нового признака 'distance'
    df["distance"] = np.sqrt(
        (df["dropoff_longitude"] - df["pickup_longitude"]) ** 2
        + (df["dropoff_latitude"] - df["pickup_latitude"]) ** 2
    )
    final_features = df[["distance", "passenger_count"]]
    return final_features

