import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

rng = np.random.default_rng(21)
core = rng.normal(loc=0.0, scale=0.9, size=(300, 2))
edge = rng.normal(loc=3.8, scale=0.55, size=(18, 2))
X = np.vstack([core, edge])

model = LocalOutlierFactor(n_neighbors=20, contamination=0.06)
labels = model.fit_predict(X)

inliers = X[labels == 1]
anomalies = X[labels == -1]

print("Total points:", len(X))
print("Detected anomalies:", len(anomalies))

plt.figure(figsize=(8, 5))
plt.scatter(inliers[:, 0], inliers[:, 1], s=24, label="Inlier")
plt.scatter(anomalies[:, 0], anomalies[:, 1], s=42, c="crimson", label="Anomaly")
plt.title("Local Outlier Factor")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.show()
