import pandas as pd

try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

# Загрузка и предобработка данных
df = pd.read_csv("uber.csv")

# Удаление пропусков и выбросов
df = df.dropna()
df = df[
    (df["fare_amount"] > 0) & (df["passenger_count"] > 0) & (df["passenger_count"] <= 6)
]

# Выбор признаков
X = df[
    [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
        "passenger_count",
    ]
]
y = df["fare_amount"]

params = {
    "data": {
        "test_size": 0.2,
        "random_state": 42,
    },
    "models": {
        "GradientBoostingRegressor": {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 5,
            "random_state": 42,
        },
        "KNeighborsRegressor": {"n_neighbors": 5},
        "LinearRegression": {},
    },
}

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, **params["data"])

# Создание и обучение GradientBoostingRegressor
gbr_reg = GradientBoostingRegressor(**params["models"]["GradientBoostingRegressor"])
gbr_reg.fit(X_train, y_train)
joblib.dump(gbr_reg, "gbr_reg.pkl", compress=True)
gbr_preds = gbr_reg.predict(X_test)

# Создание и обучение KNeighborsRegressor
knn_reg = KNeighborsRegressor(**params["models"]["KNeighborsRegressor"])
knn_reg.fit(X_train, y_train)
joblib.dump(knn_reg, "knn_reg.pkl", compress=True)
knn_preds = knn_reg.predict(X_test)

# Создание и обучение LinearRegression
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
joblib.dump(lr_reg, "lr_reg.pkl", compress=True)
lr_preds = lr_reg.predict(X_test)


def print_metrics(name, y_true, y_pred):
    print(f"\n{name}")
    print(f"RMSE: {mean_squared_error(y_true, y_pred):.3f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.3f}")
    print(f"R2: {r2_score(y_true, y_pred):.3f}")


print_metrics("GradientBoostingRegressor", y_test, gbr_preds)
print_metrics("KNeighborsRegressor", y_test, knn_preds)
print_metrics("LinearRegression", y_test, lr_preds)
