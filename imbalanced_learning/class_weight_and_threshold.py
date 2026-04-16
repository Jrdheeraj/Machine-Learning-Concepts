import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=1200,
    n_features=12,
    n_informative=5,
    n_redundant=2,
    weights=[0.93, 0.07],
    class_sep=1.1,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

model = LogisticRegression(max_iter=3000, class_weight="balanced")
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
default_pred = (proba >= 0.5).astype(int)
adjusted_pred = (proba >= 0.3).astype(int)

print("F1 default threshold 0.5:", round(float(f1_score(y_test, default_pred)), 3))
print("F1 adjusted threshold 0.3:", round(float(f1_score(y_test, adjusted_pred)), 3))
print("Confusion matrix (0.3 threshold):")
print(confusion_matrix(y_test, adjusted_pred))
print("Classification report (0.3 threshold):")
print(classification_report(y_test, adjusted_pred))
