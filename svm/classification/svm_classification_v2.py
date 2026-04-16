import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import numpy as np

                               
         
                               
data = {
    'StudyHours': ['Low','Low','Medium','Medium','High','High'],
    'Attendance': ['Poor','Good','Poor','Good','Poor','Good'],
    'PassExam': ['No','No','No','Yes','Yes','Yes']
}

df = pd.DataFrame(data)

                               
                             
                               
encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df[['StudyHours', 'Attendance']]
y = df['PassExam']

                               
                      
                               
model = SVC(kernel='linear', probability=True)
model.fit(X, y)

                               
            
                               
sample = pd.DataFrame({
    'StudyHours': ['High'],
    'Attendance': ['Good']
})

sample['StudyHours'] = encoders['StudyHours'].transform(sample['StudyHours'])
sample['Attendance'] = encoders['Attendance'].transform(sample['Attendance'])

prediction = model.predict(sample)
probabilities = model.predict_proba(sample)

result = encoders['PassExam'].inverse_transform(prediction)

print("Prediction:", result[0])

                               
                                  
                               
x_min, x_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
y_min, y_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y)
plt.xlabel("Study Hours")
plt.ylabel("Attendance")
plt.title("SVM Classification Decision Boundary")
plt.show()
