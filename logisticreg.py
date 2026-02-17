import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Dataset
# -----------------------------
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([0, 0, 0, 1, 1, 1])

# -----------------------------
# Train Logistic Regression Model
# -----------------------------
model = LogisticRegression()
model.fit(X, y)

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X)
probabilities = model.predict_proba(X)

# -----------------------------
# Visualization
# -----------------------------
X_test = np.linspace(0, 7, 100).reshape(-1, 1)
y_prob = model.predict_proba(X_test)[:, 1]

plt.figure()
plt.scatter(X, y)
plt.plot(X_test, y_prob)
plt.axhline(0.5)
plt.xlabel("Study Hours")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression Classification")
plt.show()
