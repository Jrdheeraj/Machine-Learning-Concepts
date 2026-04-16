import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

X, y = load_digits(return_X_y=True)
X_subset = X[:1200]
y_subset = y[:1200]

embedding = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42)
X_2d = embedding.fit_transform(X_subset)

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_subset, cmap="tab10", s=14)
plt.title("t-SNE Embedding (Digits)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.tight_layout()
plt.show()
