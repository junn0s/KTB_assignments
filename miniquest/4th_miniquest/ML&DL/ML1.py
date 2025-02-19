import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = np.array([[1.4, 0.2], [1.3, 0.2], [1.5, 0.2], [4.5, 1.5], [4.1, 1.0], [5.1, 1.8]])
y = np.array([0, 0, 0, 1, 1, 1])

model = KNeighborsClassifier()
model.fit(X, y) 

X_test = np.array([[11.1, 2.5], [1.2, 0.3], [1.3, 2.1], [3.3, 0.8], [6.1, 0.2]])
y_pred = model.predict(X_test) 

print("예측값:", y_pred) 