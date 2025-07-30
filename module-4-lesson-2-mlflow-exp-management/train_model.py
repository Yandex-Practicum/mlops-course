import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

# Загрузка данных
df = pd.read_csv("uber.csv")

# Предобработка
df = df.dropna()
df = df[
    (df['fare_amount'] > 0) & 
    (df['passenger_count'] > 0) & 
    (df['passenger_count'] <= 6)
]

# Фичи и таргет
X = df[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'passenger_count']]
y = df['fare_amount']

# Обучающая и тестовая выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Инициализация и обучение модели
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Предсказания на тестовой выборке
lr_preds = lr_model.predict(X_test)

# Вычисление метрик
rmse = mean_squared_error(y_test, lr_preds)
mae = mean_absolute_error(y_test, lr_preds)
r2 = r2_score(y_test, lr_preds)

print(f"Эксперимент завершён. Метрики модели: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")