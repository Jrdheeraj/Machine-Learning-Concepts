import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)

X, y = make_classification(
    n_samples=1600,
    n_features=12,
    n_informative=5,
    n_redundant=2,
    weights=[0.93, 0.07],
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

major_idx = np.where(y_train == 0)[0]
minor_idx = np.where(y_train == 1)[0]
major_sample = rng.choice(major_idx, size=len(minor_idx) * 2, replace=False)
train_idx = np.concatenate([major_sample, minor_idx])
rng.shuffle(train_idx)

X_bal = X_train[train_idx]
y_bal = y_train[train_idx]

model = LogisticRegression(max_iter=3000)
model.fit(X_bal, y_bal)
y_pred = model.predict(X_test)

print("Balanced train size:", len(y_bal))
print("Minority ratio in balanced train:", round(float(np.mean(y_bal)), 3))
print("F1:", round(float(f1_score(y_test, y_pred)), 3))
print(classification_report(y_test, y_pred))
