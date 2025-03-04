import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

input_size = 28 * 28
hidden_size = 128
output_size = 10

np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def forward(X):
    Z1 = X.dot(W1) + b1
    A1 = relu(Z1)
    Z2 = A1.dot(W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def compute_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

def backward(X, y_true, Z1, A1, Z2, A2):
    global W1, b1, W2, b2
    m = X.shape[0]

    dZ2 = A2 - y_true
    dW2 = A1.T.dot(dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = dZ2.dot(W2.T)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T.dot(dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

def update_weights(dW1, db1, dW2, db2, lr=0.1):
    global W1, b1, W2, b2
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

def train_manual(train_loader, epochs=10, lr=0.1):
    global W1, b1, W2, b2
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            X = images.view(-1, 28*28).numpy()
            y = np.eye(10)[labels.numpy()]

            Z1, A1, Z2, A2 = forward(X)
            loss = compute_loss(y, A2)
            epoch_loss += loss * len(X)

            dW1, db1, dW2, db2 = backward(X, y, Z1, A1, Z2, A2)
            update_weights(dW1, db1, dW2, db2, lr)

            correct += (np.argmax(A2, axis=1) == labels.numpy()).sum()
            total += len(X)

        print(f"Manual Epoch {epoch+1}, Loss: {epoch_loss/total:.4f}, Accuracy: {correct/total:.4f}")

def evaluate_manual(test_loader):
    correct = 0
    total = 0
    for images, labels in test_loader:
        X = images.view(-1, 28*28).numpy()
        _, _, _, A2 = forward(X)
        pred = np.argmax(A2, axis=1)
        correct += (pred == labels.numpy()).sum()
        total += len(labels)
    return correct / total

########## pytorch ###########

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_pytorch(model, train_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.view(-1, 28 * 28)

            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        print(f"PyTorch Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {correct/total:.4f}")

def evaluate_pytorch(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28 * 28)
            outputs = model(images)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total


if __name__ == '__main__':
    train_manual(train_loader, epochs=10, lr=0.1)
    manual_accuracy = evaluate_manual(test_loader)
    print(f"직접 구현한 모델 정확도: {manual_accuracy:.4f}")

    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    train_pytorch(model, train_loader, criterion, optimizer, epochs=10)
    pytorch_accuracy = evaluate_pytorch(model, test_loader)
    print(f"PyTorch 모델 정확도: {pytorch_accuracy:.4f}")