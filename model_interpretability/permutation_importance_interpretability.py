import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

result = permutation_importance(model, X_test, y_test, n_repeats=15, random_state=42, scoring="accuracy")
ranking = pd.DataFrame({"feature": X.columns, "importance": result.importances_mean}).sort_values(
    by="importance", ascending=False
)

print("Accuracy:", round(float(accuracy_score(y_test, y_pred)), 3))
print("Top features by permutation importance:")
print(ranking.head(10).to_string(index=False))
