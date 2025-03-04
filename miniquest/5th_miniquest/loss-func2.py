import numpy as np

X = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])

w = np.random.randn()
b = np.random.randn()

learning_rate = 0.01

def compute_loss(X, y, w, b):
    y_pred = w * X + b
    return np.mean((y_pred - y) ** 2)


def update_weights(X, y, w, b, learning_rate):
    y_pred = w * X + b  

    dw = -2 * np.mean(X * (y - y_pred))
    db = -2 * np.mean(y - y_pred)

    w -= learning_rate * dw
    b -= learning_rate * db

    return w, b


num_epochs = 1000 
for epoch in range(num_epochs):
    w, b = update_weights(X, y, w, b, learning_rate)
    if epoch % 100 == 0:
        loss = compute_loss(X, y, w, b)
        print(f"Epoch {epoch}: Loss = {loss}")

print(f"Trained weights: w = {w}, b = {b}")