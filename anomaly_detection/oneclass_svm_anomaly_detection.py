import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import OneClassSVM

rng = np.random.default_rng(7)
normal = rng.normal(loc=0.0, scale=1.0, size=(260, 2))
outliers = rng.normal(loc=4.2, scale=0.6, size=(25, 2))
X = np.vstack([normal, outliers])

model = OneClassSVM(kernel="rbf", gamma=0.08, nu=0.09)
labels = model.fit_predict(X)

inliers = X[labels == 1]
anomalies = X[labels == -1]

print("Total points:", len(X))
print("Detected anomalies:", len(anomalies))

plt.figure(figsize=(8, 5))
plt.scatter(inliers[:, 0], inliers[:, 1], s=24, label="Inlier")
plt.scatter(anomalies[:, 0], anomalies[:, 1], s=42, c="red", label="Anomaly")
plt.title("One-Class SVM Anomaly Detection")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.show()
