import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.model_selection import train_test_split
data = {
    "size": [800, 1000, 1200, 1500, 1800, 2000, 2300, 2500, 3200, 4000],
    "price": [30, 35, 40, 55, 65, 70, 75, 80, 85, 95]
}
df=pd.DataFrame(data)
X=df[["size"]]
y=df["price"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=DecisionTreeRegressor(max_depth=4,random_state=42)
model.fit(X_train,y_train)
print("Predicted prices:",model.predict(X_test))
plt.figure(figsize=(10,6))
plot_tree(model,feature_names=["size"],filled=True)
plt.title("Decision Tree for House Price Prediction")
plt.show()