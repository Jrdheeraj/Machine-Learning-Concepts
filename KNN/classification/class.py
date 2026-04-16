from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

                      
iris = load_iris()
X = iris.data                                                            
y = iris.target                                       

                     
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

                    
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

                                    
knn_clf = KNeighborsClassifier(
    n_neighbors=5,
    metric="minkowski",
    p=2                               
)
knn_clf.fit(X_train_scaled, y_train)

            
y_pred = knn_clf.predict(X_test_scaled)

             
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
