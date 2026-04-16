import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = load_wine(return_X_y=True)
feature_names = load_wine().feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=300, random_state=42, oob_score=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
importances = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_}).sort_values(
    by="importance", ascending=False
)

print("Test accuracy:", round(float(accuracy_score(y_test, y_pred)), 3))
print("OOB score:", round(float(model.oob_score_), 3))
print(importances.head(8).to_string(index=False))
