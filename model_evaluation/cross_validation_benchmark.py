import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

iris = load_iris()
X, y = iris.data, iris.target

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "SVM": SVC(kernel="rbf", gamma="scale"),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

rows = []
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    rows.append(
        {
            "Model": name,
            "Mean Accuracy": round(float(np.mean(scores)), 4),
            "Std": round(float(np.std(scores)), 4),
            "Fold Scores": np.round(scores, 4).tolist(),
        }
    )

result = pd.DataFrame(rows).sort_values(by="Mean Accuracy", ascending=False)
print(result.to_string(index=False))
