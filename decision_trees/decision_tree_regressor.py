import sys
import os
print("=== SCRIPT STARTED ===", file=sys.stderr)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')                           
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import confusion_matrix

print("Libraries imported successfully")

                                                  
data = {
    'Size': [800, 1000, 1200, 1500, 1800, 2000, 2300, 2500, 3200, 4000],
    'Price': [30, 35, 40, 55, 65, 70, 75, 80, 85, 95]
}

df = pd.DataFrame(data)
print("Dataframe created:", df.shape)

                                                         
X = df[['Size']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeRegressor(max_depth=4, random_state=42)
model.fit(X_train, y_train)

                                                        
y_test_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("\n=== REGRESSION RESULTS ===")
print("Predicted:", y_test_pred.tolist())
print("Actual:", y_test.tolist())
print("MSE:", mse)
print("MAE:", mae)
print("R² Score:", r2)

                                                                          
print("\nCreating confusion matrix using ALL 10 samples...")

y_all_pred = model.predict(X)

n_bins = 4
discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')

y_true_binned = discretizer.fit_transform(y.values.reshape(-1, 1)).astype(int)
y_pred_binned = discretizer.transform(y_all_pred.reshape(-1, 1)).astype(int)

cm = confusion_matrix(y_true_binned, y_pred_binned)

print("Confusion Matrix (Binned Full Data):")
print(cm)

                                                   

                            
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=X.columns, filled=True, rounded=True)
plt.title("Decision Tree Regressor")
plt.savefig('tree.png', dpi=150, bbox_inches='tight')
plt.close()

                                    
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

              
X_np = np.array(data['Size'])
Y_np = np.array(data['Price'])

ax1.scatter(X_np, Y_np, s=100, label="Data")
ax1.scatter(X_test, y_test, s=150, marker='^', label="Test Points")
ax1.scatter(X_test, y_test_pred, s=150, marker='x', label="Predictions")
ax1.axvline(1750, linestyle="--", label="Best Split", linewidth=2)

ax1.set_xlabel("Size")
ax1.set_ylabel("Price")
ax1.legend()
ax1.grid(True, alpha=0.3)

                          
im = ax2.imshow(cm, interpolation='nearest', aspect='auto')
plt.colorbar(im, ax=ax2)

edges = discretizer.bin_edges_[0]
actual_bins = cm.shape[0]

tick_labels = [f'Bin {i}\n({edges[i]:.0f}-{edges[i+1]:.0f})' for i in range(actual_bins)]

ax2.set(
    xticks=np.arange(actual_bins),
    yticks=np.arange(actual_bins),
    xticklabels=tick_labels,
    yticklabels=tick_labels,
    title='Confusion Matrix (All Data)',
    ylabel='True Label',
    xlabel='Predicted Label'
)

thresh = cm.max() / 2.
for i in range(actual_bins):
    for j in range(actual_bins):
        ax2.text(
            j, i, format(cm[i, j], 'd'),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=14, fontweight='bold'
        )

plt.tight_layout()
plt.savefig('regression_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✅ FILES SAVED:")
print("   - tree.png")
print("   - regression_analysis.png")
print("\nBin ranges:", edges)
print("=== SCRIPT FINISHED ===")
