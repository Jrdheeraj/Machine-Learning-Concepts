import numpy as np
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = make_circles(n_samples=600, noise=0.08, factor=0.45, random_state=42)

kpca = KernelPCA(n_components=2, kernel="rbf", gamma=12)
X_kpca = kpca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_kpca, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Kernel PCA + Logistic accuracy:", round(float(accuracy_score(y_test, y_pred)), 3))
