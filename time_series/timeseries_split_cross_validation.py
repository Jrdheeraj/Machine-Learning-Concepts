import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

rng = np.random.default_rng(42)
t = np.arange(220)
series = 30 + 0.04 * t + 2.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.6, len(t))

lags = 12
X = []
y = []
for i in range(lags, len(series)):
    X.append(series[i - lags : i])
    y.append(series[i])
X = np.array(X)
y = np.array(y)

tscv = TimeSeriesSplit(n_splits=5)
model = LinearRegression()
mae_scores = []

for train_idx, test_idx in tscv.split(X):
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    mae_scores.append(mean_absolute_error(y[test_idx], pred))

print("Fold MAE:", np.round(mae_scores, 3).tolist())
print("Mean MAE:", round(float(np.mean(mae_scores)), 3))
