import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import silhouette_score

X, _ = make_moons(n_samples=350, noise=0.08, random_state=42)

model = DBSCAN(eps=0.22, min_samples=6)
labels = model.fit_predict(X)

unique_labels = np.unique(labels)
cluster_count = len(unique_labels[unique_labels != -1])
noise_count = int(np.sum(labels == -1))

print("Clusters:", cluster_count)
print("Noise points:", noise_count)

if cluster_count > 1:
    valid = labels != -1
    if np.unique(labels[valid]).size > 1:
        score = silhouette_score(X[valid], labels[valid])
        print("Silhouette score:", round(float(score), 3))

plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=30)
plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.show()
