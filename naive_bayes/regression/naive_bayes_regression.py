import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

data = {
    "StudyHours": [1, 2, 3, 4, 5, 6],
    "Attendance": [55, 60, 68, 72, 85, 92],
    "Score": [42, 48, 56, 63, 78, 90],
}

df = pd.DataFrame(data)

feature_encoder = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform")
target_encoder = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform")

X = feature_encoder.fit_transform(df[["StudyHours", "Attendance"]]).astype(int)
y = target_encoder.fit_transform(df[["Score"]]).astype(int).ravel()

model = CategoricalNB()
model.fit(X, y)

sample = pd.DataFrame({"StudyHours": [5], "Attendance": [88]})
sample_X = feature_encoder.transform(sample).astype(int)
predicted_bin = model.predict(sample_X)[0]

bin_edges = target_encoder.bin_edges_[0]
score_value = (bin_edges[predicted_bin] + bin_edges[predicted_bin + 1]) / 2

print("Predicted score:", round(float(score_value), 2))
