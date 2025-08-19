import pandas as pd
# Datetime.datetime импортирована, но не используется. Плохо влияет на чистоту кода и може привести к проблемам с производительностью.
from datetime import datetime


def add_time_features(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    return df.drop('pickup_datetime', axis=1)
    # Нет новой строки в конце файла. Влияет на совместимость и читабельность
