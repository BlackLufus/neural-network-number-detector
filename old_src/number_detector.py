import cv2
import numpy as np
import os

# ---------------------------
# Help Functions
# ---------------------------
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp, axis=0, keepdims=True)

def cross_entropy(pred, target):
    return -np.sum(target * np.log(pred + 1e-9))

# ---------------------------
# Load Data
# ---------------------------
def load_data(folder='data'):
    X, Y = [], []
    for file in os.listdir(folder):
        if file.endswith('png'):
            label = int(file.split('_')[0])
            # print("file", "'", file, "'")
            img = cv2.imread('data/' + file, cv2.IMREAD_GRAYSCALE)
            _, img = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)
            arr = img.reshape(-1, 1)
            # print("img")
            # print(np.asarray(img))
            one_hot = np.zeros((10, 1))
            one_hot[label] = 1
            X.append(arr)
            Y.append(one_hot)
    return np.hstack(X), np.hstack(Y)

# ---------------------------
# Network Parameter
# ---------------------------
input_size = 784 # image size 28x28
hidden1 = 128
hidden2 = 64
output_size = 10 # number from 0-9
lr = 0.025

# initinal weights
W1 = np.random.randn(hidden1, input_size) * 0.01
b1 = np.zeros((hidden1, 1))
W2 = np.random.randn(hidden2, hidden1) * 0.01
b2 = np.zeros((hidden2, 1))
W3 = np.random.randn(output_size, hidden2) * 0.01
b3 = np.zeros((output_size, 1))

print(W1, b1)
print(W2, b2)
print(W3, b3)

# ---------------------------
# Forward + Backward
# ---------------------------
def forward(x):
    z1 = np.dot(W1, x) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = relu(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = softmax(z3)
    return (z1, a1, z2, a2, z3, a3)

def backward(x, y, z1, a1, z2, a2, z3, a3):
    global W1, b1, W2, b2, W3, b3
    m = x.shape[1]

    dz3 = a3 - y
    dW3 = np.dot((1/m) * dz3, a2.T)
    db3 = (1/m) * np.sum(dz3, axis=1, keepdims=True)

    dz2 = np.dot(W3.T, dz3) * relu_deriv(z2)
    dW2 = np.dot((1/m) * dz2, a1.T)
    db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)

    dz1 = np.dot(W2.T, dz2) * relu_deriv(z1)
    dW1 = np.dot((1/m) * dz1, x.T)
    db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)

    # Update
    W3 -= lr * dW3
    b3 -= lr * db3
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

# ---------------------------
# Training
# ---------------------------
def train(X, Y, epochs=5000):
    for epoch in range(epochs):
        z1, a1, z2, a2, z3, a3 = forward(X)
        loss = cross_entropy(a3, Y)
        backward(X, Y, z1, a1, z2, a2, z3, a3)
        if epoch % 50 == 0:
            preds = np.argmax(a3, axis=0)
            truth = np.argmax(Y, axis=0)
            acc = np.mean(preds == truth) * 100
            print(f"Epoche {epoch:4d}: Loss={loss:.4f}, Genauigkeit={acc:.2f}%")

# ---------------------------
# Predict
# ---------------------------
def predict(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)
    arr = img.reshape(-1, 1)

    _, _, _, _, _, a3 = forward(arr)
    label = int(np.argmax(a3, axis=0))
    probs = a3.flatten()  # in 1D Array umwandeln, 10 Elemente
    return label, probs

# ---------------------------
# Save
# ---------------------------
def save(filename):
    np.savez(filename, W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
    print("Weights are stored")

# ---------------------------
# Load
# ---------------------------
def load(filename):
    data = np.load(filename)
    global W1, b1, W2, b2, W3, b3
    W1 = data['W1']
    b1 = data['b1']
    W2 = data['W2']
    b2 = data['b2']
    W3 = data['W3']
    b3 = data['b3']
    print("Weights are loaded")

# ---------------------------
# Hauptprogramm
# ---------------------------
if __name__ == "__main__":
    # X_train, Y_train = load_data("data")
    # print("X shape:", X_train.shape)   # sollte (2500, N)
    # print("Y shape:", Y_train.shape)   # sollte (10, N)
    # print("min/max X:", X_train.min(), X_train.max())
    # print("unique labels count:", np.unique(np.argmax(Y_train, axis=0), return_counts=True))
    
    # # load("number_detector.npz")
    
    # train(X_train, Y_train, epochs=10000)

    # save("number_detector.npz")


    load("number_detector.npz")

    label, probs = predict("temp/temp.png")
    print("Prediction Label:", label)
    print("probabilities:", probs)
