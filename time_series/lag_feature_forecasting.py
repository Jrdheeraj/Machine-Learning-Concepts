import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

rng = np.random.default_rng(42)
periods = 240
t = np.arange(periods)
series = 20 + 0.06 * t + 2.5 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.8, periods)

df = pd.DataFrame({"y": series})
for lag in [1, 2, 3, 24]:
    df[f"lag_{lag}"] = df["y"].shift(lag)

df = df.dropna().reset_index(drop=True)
X = df[["lag_1", "lag_2", "lag_3", "lag_24"]]
y = df["y"]

split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MAE:", round(float(mean_absolute_error(y_test, y_pred)), 3))
print("R2:", round(float(r2_score(y_test, y_pred)), 3))

plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test.values, label="Actual")
plt.plot(y_test.index, y_pred, label="Forecast")
plt.title("Lag-Feature Time Series Forecasting")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()
