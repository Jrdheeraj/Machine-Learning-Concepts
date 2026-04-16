import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D

X, y = make_classification(
    n_samples=5000,
    n_features=3,
    n_classes=3,
    n_clusters_per_class=1,
    n_redundant=0,
    random_state=42
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300),
)

z_fixed = np.full(xx.ravel().shape[0], X[:, 2].mean())

W = knn.predict(np.c_[xx.ravel(), yy.ravel(), z_fixed])
W = W.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, W, alpha=0.3, cmap=plt.cm.Set1)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Set1)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KNN Decision Boundary (Features 1 & 2, Feature 3 fixed)')
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.Set1, edgecolors='k')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('3D Scatter of Synthetic Data (3 Classes)')
plt.show()
