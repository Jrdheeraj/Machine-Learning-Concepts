import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(42)
normal = rng.normal(loc=0.0, scale=1.0, size=(280, 2))
outliers = rng.normal(loc=5.5, scale=0.5, size=(20, 2))
X = np.vstack([normal, outliers])

model = IsolationForest(contamination=0.07, random_state=42)
labels = model.fit_predict(X)

inliers = X[labels == 1]
anomalies = X[labels == -1]

print("Total points:", len(X))
print("Detected anomalies:", len(anomalies))

plt.figure(figsize=(8, 5))
plt.scatter(inliers[:, 0], inliers[:, 1], s=25, label="Inlier")
plt.scatter(anomalies[:, 0], anomalies[:, 1], s=45, c="red", label="Anomaly")
plt.title("Isolation Forest Anomaly Detection")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.show()
