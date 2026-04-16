import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)

                                    
input_dim = 3
hidden_dim = 5
output_dim = 1

                         
W1 = np.random.randn(input_dim, hidden_dim)
b1 = np.random.randn(hidden_dim)
W2 = np.random.randn(hidden_dim, output_dim)
b2 = np.random.randn(output_dim)

def relu(x):
    return np.maximum(0, x)

                                       
x = np.array([0.5, -1.0, 1.5])                
h = relu(x @ W1 + b1)                               
y = h @ W2 + b2                         

print("Hidden activations:", h)
print("Output:", y)

                                                     
              
                                                    
                                                         

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

                                    
ax.scatter(x[0], x[1], x[2], c='green', s=120, edgecolor='k', label='Input (x)')

                                                                    
hidden_positions = []
z_layer = 3.0                                    
xs = np.linspace(-1, 1, hidden_dim)
ys = np.zeros(hidden_dim)

for i in range(hidden_dim):
    hidden_positions.append((xs[i], ys[i], z_layer))

hidden_positions = np.array(hidden_positions)

                                                          
norm_h = (h - h.min()) / (np.ptp(h) + 1e-8)
colors = plt.cm.viridis(norm_h)

ax.scatter(hidden_positions[:, 0],
           hidden_positions[:, 1],
           hidden_positions[:, 2],
           c=colors, s=120, edgecolor='k',
           label='Hidden neurons')

                                          
for j in range(hidden_dim):
    ax.plot([x[0], hidden_positions[j, 0]],
            [x[1], hidden_positions[j, 1]],
            [x[2], hidden_positions[j, 2]],
            c='gray', alpha=0.4)

          
ax.text(x[0], x[1], x[2], "  Input x", color='green')
for j in range(hidden_dim):
    ax.text(hidden_positions[j, 0],
            hidden_positions[j, 1],
            hidden_positions[j, 2] + 0.1,
            f"h{j+1}", color='black', fontsize=8)

ax.set_title("3D Illustration of Input → Hidden Layer (Simple Neural Network)")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3 / Layer axis")
ax.legend()
plt.tight_layout()
plt.show()
