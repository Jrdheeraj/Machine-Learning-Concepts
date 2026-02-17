import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
# -----------------------------
# 1. Create Student Dataset
# -----------------------------
data = {
    'StudyHours': ['Low', 'Low', 'Medium', 'Medium', 'High', 'High'],
    'Attendance': ['Poor', 'Good', 'Poor', 'Good', 'Poor', 'Good'],
    'PassExam': ['No', 'No', 'No', 'Yes', 'Yes', 'Yes']
}
df = pd.DataFrame(data)
# -----------------------------
# 2. Encode categorical features
# -----------------------------
encoders = {}
for column in df.columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    encoders[column] = encoder   # store encoders
# Features and target
X = df[['StudyHours', 'Attendance']]
y = df['PassExam']
# -----------------------------
# 3. Train Naïve Bayes Model
# -----------------------------
model = CategoricalNB()
model.fit(X, y)
# -----------------------------
# 4. Predict New Student Result
# -----------------------------
sample = pd.DataFrame({
    'StudyHours': ['High'],
    'Attendance': ['Good']
})

# Encode input sample
sample['StudyHours'] = encoders['StudyHours'].transform(sample['StudyHours'])
sample['Attendance'] = encoders['Attendance'].transform(sample['Attendance'])

prediction = model.predict(sample)

# Decode prediction
result = encoders['PassExam'].inverse_transform(prediction)

print("Prediction:", result[0])
