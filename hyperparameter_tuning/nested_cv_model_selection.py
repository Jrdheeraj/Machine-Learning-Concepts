import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)

inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    "C": [0.1, 1, 10],
    "gamma": [0.001, 0.01, 0.1],
    "kernel": ["rbf"],
}

grid = GridSearchCV(SVC(), param_grid=param_grid, cv=inner_cv, scoring="accuracy", n_jobs=-1)

scores = cross_val_score(grid, X, y, cv=outer_cv, scoring="accuracy", n_jobs=-1)
print("Nested CV fold scores:", np.round(scores, 4).tolist())
print("Nested CV mean accuracy:", round(float(np.mean(scores)), 4))
