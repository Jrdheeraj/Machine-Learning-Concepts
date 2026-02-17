import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
iris=load_iris()
X,y=iris.data,iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
clf=RandomForestClassifier(n_estimators=100,random_state=42)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred))
results_class=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print("Random Forest Classification Results:")
print(results_class)
importances=clf.feature_importances_
features=iris.feature_names
plt.figure(figsize=(8,6))
plt.barh(features,importances,color="blue")
plt.title("Feature Importances in Random Forest")
plt.show()