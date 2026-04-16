import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
                               
                           
                               
data = {
    'StudyHours': ['Low', 'Low', 'Medium', 'Medium', 'High', 'High'],
    'Attendance': ['Poor', 'Good', 'Poor', 'Good', 'Poor', 'Good'],
    'PassExam': ['No', 'No', 'No', 'Yes', 'Yes', 'Yes']
}
df = pd.DataFrame(data)
                               
                                
                               
encoders = {}
for column in df.columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    encoders[column] = encoder                   
                     
X = df[['StudyHours', 'Attendance']]
y = df['PassExam']
                               
                            
                               
model = CategoricalNB()
model.fit(X, y)
                               
                               
                               
sample = pd.DataFrame({
    'StudyHours': ['High'],
    'Attendance': ['Good']
})

                     
sample['StudyHours'] = encoders['StudyHours'].transform(sample['StudyHours'])
sample['Attendance'] = encoders['Attendance'].transform(sample['Attendance'])

prediction = model.predict(sample)

                   
result = encoders['PassExam'].inverse_transform(prediction)

print("Prediction:", result[0])
