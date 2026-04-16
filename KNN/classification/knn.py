import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier

                                       
np.random.seed(42)
n_samples = 40

                         
X0 = np.random.normal(loc=[0, 0, 0], scale=0.6, size=(n_samples, 3))
y0 = np.zeros(n_samples)

                         
X1 = np.random.normal(loc=[2, 2, 2], scale=0.6, size=(n_samples, 3))
y1 = np.ones(n_samples)

X = np.vstack([X0, X1])
y = np.hstack([y0, y1])

                         
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)

                                   
query = np.array([[1.0, 1.0, 0.5]])                                  

distances, indices = knn.kneighbors(query)

                      
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

                      
ax.scatter(X0[:, 0], X0[:, 1], X0[:, 2], c='blue', label='Class 0', alpha=0.7)
ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2], c='red', label='Class 1', alpha=0.7)

                  
ax.scatter(query[0, 0], query[0, 1], query[0, 2],
           c='green', s=120, edgecolor='k', label='Query point')

                     
neighbors = X[indices[0]]
neighbor_labels = y[indices[0]]
for i, (px, py, pz) in enumerate(neighbors):
    color = 'blue' if neighbor_labels[i] == 0 else 'red'
    ax.scatter(px, py, pz, c=color, s=80, edgecolor='k')
                                 
    ax.plot([query[0, 0], px],
            [query[0, 1], py],
            [query[0, 2], pz],
            c='grey', linestyle='--', alpha=0.5)

                 
pred = knn.predict(query)[0]
title = f"k-NN (k={k}) - Predicted class for query: {int(pred)}"
ax.set_title(title)

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.legend()
plt.tight_layout()
plt.show()
