import matplotlib.pyplot as plt
import numpy as np

X=np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y=np.array([[0], [1], [1], [0]])
def sigmoid(X):
    return 1/(1+np.exp(-X))

def sigmoid_derivative(X):
        return X*(1-X)

np.random.seed(0)
w1=np.random.rand(2,2)
b1=np.zeros((1,2))
w2=np.random.rand(2,1)
b2=np.zeros((1,1))
epochs=5000
lr=0.1
losses=[]
pred_history=[]
hidden_history=[]
for epoch in range(epochs):
    z1=np.dot(X,w1)+b1
    A1=sigmoid(z1)
    hidden_history.append(A1.copy())
    z2=np.dot(A1,w2)+b2
    y_pred=sigmoid(z2)
    loss=np.mean((y-y_pred)**2)
    losses.append(loss)
    pred_history.append(y_pred.copy())
    error=y_pred-y
    d2=error*sigmoid_derivative(y_pred)
    dw2=np.dot(A1.T,d2)
    d1=np.dot(d2,w2.T)*sigmoid_derivative(A1)
    dw1=np.dot(X.T,d1)
    w2-=lr*dw2
    w1-=lr*dw1
    b2 -= lr * np.sum(d2, axis=0, keepdims=True)
    b1 -= lr * np.sum(d1, axis=0, keepdims=True)
    
print("Final Predictions:\n",y_pred)
plt.plot(losses)
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

    

