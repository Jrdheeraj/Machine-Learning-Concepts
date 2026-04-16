import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=1500,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    weights=[0.94, 0.06],
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

baseline = LogisticRegression(max_iter=3000)
weighted = LogisticRegression(max_iter=3000, class_weight={0: 1.0, 1: 6.0})

baseline.fit(X_train, y_train)
weighted.fit(X_train, y_train)

pred_baseline = baseline.predict(X_test)
pred_weighted = weighted.predict(X_test)

print("Baseline F1:", round(float(f1_score(y_test, pred_baseline)), 3))
print("Cost-sensitive F1:", round(float(f1_score(y_test, pred_weighted)), 3))
print("Cost-sensitive report:")
print(classification_report(y_test, pred_weighted))
