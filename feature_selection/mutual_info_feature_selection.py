import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

selector = SelectKBest(score_func=mutual_info_classif, k=8)
X_selected = selector.fit_transform(X, y)
selected_columns = X.columns[selector.get_support()].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Selected features:")
for name in selected_columns:
    print("-", name)
print("Accuracy:", round(float(accuracy_score(y_test, y_pred)), 3))
