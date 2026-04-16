import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

black_box = RandomForestClassifier(n_estimators=300, random_state=42)
black_box.fit(X_train, y_train)

prob = black_box.predict_proba(X_test)[:, 1]

surrogate = DecisionTreeRegressor(max_depth=4, random_state=42)
surrogate.fit(X_test, prob)
prob_hat = surrogate.predict(X_test)

fidelity = r2_score(prob, prob_hat)
importance = pd.DataFrame({"feature": X.columns, "importance": surrogate.feature_importances_}).sort_values(
    by="importance", ascending=False
)

print("Surrogate fidelity (R2):", round(float(fidelity), 3))
print(importance.head(10).to_string(index=False))
