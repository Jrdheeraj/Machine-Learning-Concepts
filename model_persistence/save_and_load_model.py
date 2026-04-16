import pathlib
import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

model_path = pathlib.Path("model_persistence") / "iris_random_forest.joblib"
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_path)

loaded_model = joblib.load(model_path)
y_pred = loaded_model.predict(X_test)

print("Saved model path:", model_path)
print("Accuracy after loading:", round(float(accuracy_score(y_test, y_pred)), 3))
