import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

selector_model = LogisticRegression(max_iter=5000, penalty="l1", solver="liblinear", C=0.15)
selector_model.fit(X_train, y_train)

coef = np.abs(selector_model.coef_[0])
mask = coef > 1e-6
selected = X.columns[mask]

X_train_sel = X_train[selected]
X_test_sel = X_test[selected]

final_model = LogisticRegression(max_iter=5000)
final_model.fit(X_train_sel, y_train)
y_pred = final_model.predict(X_test_sel)

print("Selected feature count:", len(selected))
print("Selected features:")
for name in selected:
    print("-", name)
print("Accuracy:", round(float(accuracy_score(y_test, y_pred)), 3))
