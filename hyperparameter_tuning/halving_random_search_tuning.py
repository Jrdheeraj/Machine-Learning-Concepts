from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import HalvingRandomSearchCV, train_test_split
from scipy.stats import randint, uniform

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

model = RandomForestClassifier(random_state=42)
param_dist = {
    "n_estimators": randint(80, 350),
    "max_depth": randint(2, 18),
    "min_samples_split": randint(2, 12),
    "max_features": uniform(0.4, 0.6),
}

search = HalvingRandomSearchCV(
    model,
    param_distributions=param_dist,
    factor=2,
    resource="n_estimators",
    max_resources=350,
    min_resources=80,
    random_state=42,
    cv=4,
    n_jobs=-1,
)
search.fit(X_train, y_train)

pred = search.best_estimator_.predict(X_test)
print("Best params:", search.best_params_)
print("Test accuracy:", round(float(accuracy_score(y_test, pred)), 3))
