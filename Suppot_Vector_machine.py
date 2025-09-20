import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data  # All 4 features
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(y)

# Train-test split (for prediction model)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM model (for user prediction)
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Get user input
print("Enter flower features:")
sepal_length = float(input("Sepal length (cm): "))
sepal_width = float(input("Sepal width (cm): "))
petal_length = float(input("Petal length (cm): "))
petal_width = float(input("Petal width (cm): "))

# Predict flower type
input_features = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_features)
flower_name = target_names[prediction[0]]
print("Predicted flower type:", flower_name)

# ---- Plotting Decision Boundary using only first two features ----

# Use only first two features (sepal length & width) for graph
X_plot = iris.data[:, :2]
y_plot = iris.target

# Train a model for plotting
plot_model = SVC(kernel='linear')
plot_model.fit(X_plot, y_plot)

# Create mesh grid for plotting
import numpy as np
x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))
Z = plot_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title("SVM Decision Boundary (2 features)")
plt.grid(True)
plt.show()
