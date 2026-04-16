import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=320, centers=4, cluster_std=0.72, random_state=52)

model = AgglomerativeClustering(n_clusters=4, linkage="ward")
labels = model.fit_predict(X)
score = silhouette_score(X, labels)

print("Silhouette score:", round(float(score), 3))

plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=28)
plt.title("Agglomerative Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.show()
