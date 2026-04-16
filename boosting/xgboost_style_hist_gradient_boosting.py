import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=600,
    n_features=6,
    n_informative=4,
    n_redundant=0,
    class_sep=1.2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", round(float(accuracy_score(y_test, y_pred)), 3))
print("ROC-AUC:", round(float(roc_auc_score(y_test, y_prob)), 3))

x0, x1 = 0, 1
x_min, x_max = X[:, x0].min() - 1.0, X[:, x0].max() + 1.0
y_min, y_max = X[:, x1].min() - 1.0, X[:, x1].max() + 1.0
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 220), np.linspace(y_min, y_max, 220))

X_grid = np.zeros((xx.size, X.shape[1]))
X_grid[:, x0] = xx.ravel()
X_grid[:, x1] = yy.ravel()
for col in range(2, X.shape[1]):
    X_grid[:, col] = X[:, col].mean()

zz = model.predict(X_grid).reshape(xx.shape)

plt.figure(figsize=(8, 5))
plt.contourf(xx, yy, zz, alpha=0.25, cmap="coolwarm")
plt.scatter(X[:, x0], X[:, x1], c=y, cmap="coolwarm", s=24)
plt.title("XGBoost-Style Histogram Gradient Boosting")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.show()
