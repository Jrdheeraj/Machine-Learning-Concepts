from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

                                 
diabetes = load_diabetes()
X = diabetes.data                           
y = diabetes.target                                      

                     
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

                    
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

                                   
knn_reg = KNeighborsRegressor(
    n_neighbors=5,
    metric="minkowski",
    p=2                         
)
knn_reg.fit(X_train_scaled, y_train)

            
y_pred = knn_reg.predict(X_test_scaled)

             
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R² score:", r2)
