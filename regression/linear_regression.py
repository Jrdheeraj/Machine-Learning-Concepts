import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

                               
         
                               
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([35, 40, 50, 60, 65])

                               
                               
                               
model = LinearRegression()
model.fit(X, y)

                               
             
                               
y_pred = model.predict(X)

print("Slope (β1):", model.coef_[0])
print("Intercept (β0):", model.intercept_)

                               
               
                               
plt.figure()
plt.scatter(X, y)
plt.plot(X, y_pred)
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Simple Linear Regression")
plt.show()
