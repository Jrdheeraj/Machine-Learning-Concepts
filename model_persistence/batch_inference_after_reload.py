import pathlib
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

X, y = load_iris(return_X_y=True)

artifact_dir = pathlib.Path("model_persistence")
artifact_dir.mkdir(parents=True, exist_ok=True)
model_path = artifact_dir / "iris_batch_inference_model.joblib"

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)
joblib.dump(model, model_path)

loaded = joblib.load(model_path)
new_batch = np.array([[5.1, 3.5, 1.4, 0.2], [6.7, 3.1, 4.7, 1.5], [7.2, 3.0, 5.8, 1.6]])
pred = loaded.predict(new_batch)
prob = loaded.predict_proba(new_batch)

print("Predicted classes:", pred.tolist())
print("Predicted probabilities:")
print(np.round(prob, 3))
