import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=500, noise=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test, dtype=torch.long)


class DeepMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=2):
        super(DeepMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

model = DeepMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


epochs = 500
for epoch in range(1, epochs+1):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        with torch.no_grad():
            pred = model(X_train_tensor).argmax(dim=1)
            acc = (pred == y_train_tensor).float().mean()
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, Train Acc: {acc.item():.4f}")



with torch.no_grad():
    pred_test = model(X_test_tensor).argmax(dim=1)
    test_acc = (pred_test == y_test_tensor).float().mean()
print(f"Test Accuracy: {test_acc.item():.4f}")


############# 시각화 #############

import numpy as np

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32)
    
    with torch.no_grad():
        Z = model(grid_tensor).argmax(dim=1).numpy()
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Spectral)
    plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap=plt.cm.Spectral, edgecolors='k')

plt.figure(figsize=(6,5))
plot_decision_boundary(model, X_test, y_test)
plt.title("MLP Decision Boundary on make_moons")
plt.show()