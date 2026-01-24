import json

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

with open("config.json") as f:
    config = json.load(f)

df = pd.read_csv(config["data_path"])
df.dropna(inplace=True)
df["wind_humidity_ratio"] = df["wind_speed"] / df["humidity"].replace(0, 0.1)

X = df[config["features"] + ["wind_humidity_ratio"]]
y = df[config["target"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=config["random_state"]
)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, config["output_model"])

score = model.score(X_test, y_test)
print(f"Model R^2 score: {score}")
