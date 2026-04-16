import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

X = np.array([800, 1000, 1200, 1500, 1800, 2000, 2300, 2500, 3200, 4000]).reshape(-1, 1)
y = np.array([30, 35, 40, 55, 65, 70, 75, 80, 85, 95])

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

X_test = np.linspace(700, 4100, 200).reshape(-1, 1)
y_pred = model.predict(X_test)

print("Predictions:", np.round(model.predict(X[:5]), 2))
print("Feature importance:", model.feature_importances_)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="black")
plt.plot(X_test, y_pred, color="darkgreen")
plt.xlabel("Size")
plt.ylabel("Price")
plt.title("Random Forest Regression")
plt.tight_layout()
plt.show()
