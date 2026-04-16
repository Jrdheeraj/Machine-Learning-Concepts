import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

rng = np.random.default_rng(10)
rows = 180

X = pd.DataFrame(
    {
        "age": rng.integers(18, 65, rows),
        "income": rng.normal(52000, 11000, rows),
        "experience": rng.integers(0, 25, rows),
        "city": rng.choice(["A", "B", "C", "D"], rows),
    }
)

num_features = ["age", "income", "experience"]
cat_features = ["city"]

num_pipeline = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)), ("scale", StandardScaler())])
cat_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer([("num", num_pipeline, num_features), ("cat", cat_pipeline, cat_features)])
Xt = preprocessor.fit_transform(X)

print("Input shape:", X.shape)
print("Transformed shape:", Xt.shape)
