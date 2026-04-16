import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.7, random_state=42)

model = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = model.fit_predict(X)
centers = model.cluster_centers_
score = silhouette_score(X, labels)

print("Silhouette score:", round(float(score), 3))

plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=35)
plt.scatter(centers[:, 0], centers[:, 1], c="red", s=180, marker="X")
plt.title("KMeans Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.show()
