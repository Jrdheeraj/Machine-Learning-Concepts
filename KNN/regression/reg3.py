import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
                                        
X, y = make_regression(n_samples=257, n_features=1, noise=15, random_state=42)
print("Feature (X) values:\n", X[:100])
print("\nTarget (y) values:\n", y[:100])
                     
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                  
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)
            
X_plot = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
y_pred = knn.predict(X_plot)
         
plt.scatter(X_train, y_train, color="blue", label="Training data")
plt.plot(X_plot, y_pred, color="red", label="KNN Regression (k=15)")
plt.title("KNN Regression")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.show()