import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
X = np.linspace(0, 10, 250).reshape(-1, 1)
y = np.sin(X[:, 0]) + 0.2 * rng.normal(size=X.shape[0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("R2:", round(float(r2_score(y_test, y_pred)), 3))
print("MSE:", round(float(mean_squared_error(y_test, y_pred)), 3))

X_plot = np.linspace(0, 10, 300).reshape(-1, 1)
y_plot = model.predict(X_plot)

plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], y, s=18, alpha=0.55)
plt.plot(X_plot[:, 0], y_plot, color="darkorange", linewidth=2)
plt.title("Gradient Boosting Regression")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.tight_layout()
plt.show()
