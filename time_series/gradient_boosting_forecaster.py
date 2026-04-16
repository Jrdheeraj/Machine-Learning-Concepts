import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

rng = np.random.default_rng(123)
t = np.arange(260)
series = 15 + 0.05 * t + 1.8 * np.sin(2 * np.pi * t / 18) + rng.normal(0, 0.5, len(t))

lag = 16
X = []
y = []
for i in range(lag, len(series)):
    X.append(series[i - lag : i])
    y.append(series[i])
X = np.array(X)
y = np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MAE:", round(float(mean_absolute_error(y_test, y_pred)), 3))

idx = np.arange(len(y_test))
plt.figure(figsize=(9, 5))
plt.plot(idx, y_test, label="Actual")
plt.plot(idx, y_pred, label="Forecast")
plt.title("Gradient Boosting Time Series Forecast")
plt.xlabel("Test Step")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()
