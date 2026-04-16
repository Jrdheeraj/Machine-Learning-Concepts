import json
import pathlib
import joblib
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

pipeline = Pipeline([("scale", StandardScaler()), ("model", SVC(probability=True, random_state=42))])
pipeline.fit(X_train, y_train)

artifact_dir = pathlib.Path("model_persistence")
artifact_dir.mkdir(parents=True, exist_ok=True)
model_path = artifact_dir / "wine_svc_pipeline.joblib"
meta_path = artifact_dir / "wine_svc_pipeline_meta.json"

joblib.dump(pipeline, model_path)
meta = {"dataset": "wine", "algorithm": "SVC", "pipeline_steps": ["scale", "model"]}
meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

loaded = joblib.load(model_path)
pred = loaded.predict(X_test)

print("Saved model:", model_path)
print("Saved metadata:", meta_path)
print("Accuracy after reload:", round(float(accuracy_score(y_test, pred)), 3))
