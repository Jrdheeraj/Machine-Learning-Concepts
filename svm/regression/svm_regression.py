import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([2.0, 2.8, 3.6, 4.5, 5.5, 6.3])

model = SVR(kernel="rbf", C=100, gamma=0.5, epsilon=0.1)
model.fit(X, y)

X_test = np.linspace(0.5, 6.5, 200).reshape(-1, 1)
y_pred = model.predict(X_test)

print("Predictions:", np.round(model.predict(X), 2))

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="black")
plt.plot(X_test, y_pred, color="steelblue")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("SVM Regression")
plt.tight_layout()
plt.show()
