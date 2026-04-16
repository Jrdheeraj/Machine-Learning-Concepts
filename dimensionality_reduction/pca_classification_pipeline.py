import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("PCA explained variance ratio:", np.round(pca.explained_variance_ratio_, 3))
print("Accuracy:", round(float(accuracy_score(y_test, y_pred)), 3))

plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="Set1", s=35)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection of Iris Dataset")
plt.tight_layout()
plt.show()
