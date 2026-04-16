import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

rng = np.random.default_rng(42)
rows = 300

df = pd.DataFrame(
    {
        "age": rng.integers(18, 65, rows),
        "income": rng.normal(55000, 12000, rows),
        "city": rng.choice(["A", "B", "C"], rows),
        "segment": rng.choice(["new", "active", "churn"], rows),
    }
)

df.loc[rng.choice(rows, 20, replace=False), "income"] = np.nan
df.loc[rng.choice(rows, 15, replace=False), "city"] = np.nan

y = ((df["age"].fillna(df["age"].median()) > 35) & (df["segment"] != "churn")).astype(int)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=42, stratify=y)

numeric_features = ["age", "income"]
categorical_features = ["city", "segment"]

numeric_pipeline = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

categorical_pipeline = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]
)

preprocessor = ColumnTransformer(
    transformers=[("num", numeric_pipeline, numeric_features), ("cat", categorical_pipeline, categorical_features)]
)

model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=2000))])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", round(float(accuracy_score(y_test, y_pred)), 3))
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
