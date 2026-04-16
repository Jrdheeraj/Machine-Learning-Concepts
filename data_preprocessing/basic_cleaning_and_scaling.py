import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

rng = np.random.default_rng(42)
df = pd.DataFrame(
    {
        "hours": rng.integers(1, 10, 15).astype(float),
        "attendance": rng.integers(50, 100, 15).astype(float),
        "score": rng.integers(35, 95, 15).astype(float),
    }
)

df.loc[[2, 6, 11], "attendance"] = np.nan

imputer = SimpleImputer(strategy="median")
scaler = MinMaxScaler()

X_imputed = imputer.fit_transform(df[["hours", "attendance"]])
X_scaled = scaler.fit_transform(X_imputed)

processed = pd.DataFrame(X_scaled, columns=["hours_scaled", "attendance_scaled"])
processed["score"] = df["score"].values

print("Original shape:", df.shape)
print("Processed shape:", processed.shape)
print(processed.head().to_string(index=False))
