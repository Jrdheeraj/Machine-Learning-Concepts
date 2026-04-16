import numpy as np
from scipy.stats import randint, uniform
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

base = RandomForestClassifier(random_state=42)

grid_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 6, 10],
    "min_samples_split": [2, 5],
}

grid = GridSearchCV(base, grid_params, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

grid_pred = grid.best_estimator_.predict(X_test)

random_params = {
    "n_estimators": randint(100, 350),
    "max_depth": randint(3, 14),
    "min_samples_split": randint(2, 10),
    "max_features": uniform(0.4, 0.6),
}

random_search = RandomizedSearchCV(
    base,
    random_params,
    n_iter=20,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
)
random_search.fit(X_train, y_train)

random_pred = random_search.best_estimator_.predict(X_test)

print("Grid best params:", grid.best_params_)
print("Grid test accuracy:", round(float(accuracy_score(y_test, grid_pred)), 3))
print("Random best params:", random_search.best_params_)
print("Random test accuracy:", round(float(accuracy_score(y_test, random_pred)), 3))
