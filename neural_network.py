from typing import Any
import cv2
import numpy as np
import os

import activation_function as af

import os
os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"

from concurrent.futures import ThreadPoolExecutor

class NeuralNetwork:

    # ---------------------------
    # Network Parameter
    # ---------------------------
    # input_size = 784 # image size 28x28
    # hidden_layers_size = [] # hidden layers
    # output_size = 10 # number from 0-9
    # lr = 0.025

    W = []
    b = []

    # ---------------------------
    # Initinal Function
    # ---------------------------
    def __init__(self, input_size, hidden_layers_size, output_size, lr=0.025, decay=0.001, min_lr=0.0001, activation='relu'):

        self.input_size = input_size
        self.hidden_layers_size = hidden_layers_size
        self.output_size = output_size
        self.lr = lr
        self.decay = decay
        self.min_lr = min_lr
        self.activation_name = activation

        self.__activation_function = getattr(af, self.activation_name)
        self.__activation_deriv_func = getattr(af, f"{self.activation_name}_deriv")
        
        self.total_layers = 1 + len(hidden_layers_size)

        self.W = [None] * self.total_layers
        self.b = [None] * self.total_layers

        # initinal weights
        self.W[0] = np.random.randn(self.hidden_layers_size[0], self.input_size) * 0.01
        self.b[0] = np.zeros((self.hidden_layers_size[0], 1))
        if len(hidden_layers_size) > 1:
            for i, hidden_layer in enumerate(hidden_layers_size[:-1]):
                self.W[1+i] = np.random.randn(self.hidden_layers_size[i+1], hidden_layer) * 0.01
                self.b[1+i] = np.zeros((self.hidden_layers_size[i+1], 1))
        self.W[-1] = np.random.randn(self.output_size, self.hidden_layers_size[-1]) * 0.01
        self.b[-1] = np.zeros((self.output_size, 1))

    # ---------------------------
    # Load Data
    # ---------------------------
    def load_data(self, folders=['data/train/0', 'data/train/1', 'data/train/2']):
        X, Y = [], []
        print(folders)
        for index, folder in enumerate(folders):
            print(folder)
            for file in os.listdir(folder):
                if file.endswith('png'):
                    # print("file", "'", file, "'")
                    img = cv2.imread(folder + "/" + file, cv2.IMREAD_GRAYSCALE)
                    _, img = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)
                    arr = img.reshape(-1, 1).astype(np.float32)
                    # print("img")
                    # print(np.asarray(img))
                    one_hot = np.zeros((len(folders), 1))
                    one_hot[index] = 1
                    X.append(arr)
                    Y.append(one_hot)
        return np.hstack(X), np.hstack(Y)
    
    # ---------------------------
    # Forward + Backward
    # ---------------------------
    def forward(self, x):

        Z = np.empty((self.total_layers,), dtype=object)
        A = np.empty((self.total_layers,), dtype=object)


        for i in range(self.total_layers):
            Z[i] = np.dot(self.W[i], (x if i == 0 else A[i-1])) + self.b[i]
            if i == self.total_layers - 1:
                A[i] = af.softmax(Z[i])
            else:
                A[i] = self.__activation_function(Z[i])
        
        return Z, A

    def backward(self, X, Y, Z: list, A: list):
        m = X.shape[1]

        dz = np.empty((self.total_layers,), dtype=object)
        dW = np.empty((self.total_layers,), dtype=object)
        db = np.empty((self.total_layers,), dtype=object)

        for l in reversed(range(self.total_layers)):
            # dz
            if l == self.total_layers - 1:
                dz[l] = A[l] - Y
            else:
                dz[l] = np.dot(self.W[l+1].T, dz[l+1]) * self.__activation_deriv_func(Z[l])

            # dW
            A_prev = X if l == 0 else A[l-1]
            dW[l] = (1/m) * np.dot(dz[l], A_prev.T)

            # db
            db[l] = (1/m) * np.sum(dz[l], axis=1, keepdims=True)
        
        # Update weights and biases
        for l in reversed(range(self.total_layers)):
            self.W[l] -= self.lr * dW[l]
            self.b[l] -= self.lr * db[l]
        
        # Reduce learning rate
        self.lr = max(self.min_lr, self.lr / (1 + self.decay))

    # ---------------------------
    # Training
    # ---------------------------
    def train(self, X, Y, epochs=5000, batch_size=32):

        m = X.shape[1] # (n_features, n_examples)

        for epoch in range(epochs):
            
            # Shuffle the training examples randomly for this epoch
            permutation = np.random.permutation(m)
            X_shuffled = X[:, permutation]
            Y_shuffled = Y[:, permutation]

            # Process data in mini-batches
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[:, i:i + batch_size]
                Y_batch = Y_shuffled[:, i:i + batch_size]

                Z, A = self.forward(X_batch)
                self.backward(X_batch, Y_batch, Z, A)
            if epoch % 50 == 0:
                _, A_full = self.forward(X)
                loss_full = af.cross_entropy(A_full[-1], Y)
                preds = np.argmax(A_full[-1], axis=0)
                truth = np.argmax(Y, axis=0)
                acc = np.mean(preds == truth) * 100
                print(f"Epoche {epoch:4d}: Loss={loss_full:.4f}, Genauigkeit={acc:.2f}%")

    # ---------------------------
    # Predict
    # ---------------------------
    def predict(self, image:str|cv2.Mat| np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]] | None):
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        elif image is None:
            raise Exception("None value for image is not allowed!")
        
        _, img = cv2.threshold(image, 128, 1, cv2.THRESH_BINARY)
        arr = img.reshape(-1, 1)

        _, A = self.forward(arr)
        label = int(np.argmax(A[-1], axis=0))
        probs = A[-1].flatten()  # in 1D Array umwandeln, 10 Elemente
        return label, probs

    # ---------------------------
    # Save
    # ---------------------------
    def save(self, filename):
        np.savez(filename, W=np.array(self.W, dtype=object), b=np.array(self.b, dtype=object))
        print("Weights are stored")

    # ---------------------------
    # Load
    # ---------------------------
    def load(self, filename):
        data = np.load(filename, allow_pickle=True)
        self.W = list(data['W'])
        self.b = list(data['b'])
        print("Weights are loaded")

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # nn = NeuralNetwork(input_size=784, hidden_layers_size=[256, 128, 64], output_size=10, lr=0.25, decay=0.00001, min_lr=0.0001, activation='relu')
    # nn = NeuralNetwork(input_size=784, hidden_layers_size=[256, 128, 64], output_size=10, lr=0.1, decay=0.0001, min_lr=0.01, activation='relu')
    nn = NeuralNetwork(
        input_size=784,
        hidden_layers_size=[256, 128, 64],
        output_size=11,
        lr=0.025,
        decay=0.0001,
        min_lr=0.01,
        activation='relu'
    )

    X_train, Y_train = nn.load_data(['data/train/0', 'data/train/1', 'data/train/2', 'data/train/3', 'data/train/4', 'data/train/5', 'data/train/6', 'data/train/7', 'data/train/8', 'data/train/9', 'data/train/10'])
    print("X shape:", X_train.shape)   # sollte (784, N)
    print("Y shape:", Y_train.shape)   # sollte (10, N)
    print("min/max X:", X_train.min(), X_train.max())
    print("unique labels count:", np.unique(np.argmax(Y_train, axis=0), return_counts=True))
    
    # nn.load("models/nn_number_detector_large_0007.npz")
    
    nn.train(X_train, Y_train, epochs=4000, batch_size=64)

    nn.save("models/nn_number_detector_large_extra_0001.npz")

    nn.load("models/nn_number_detector_large_extra_0001.npz")

    label, probs = nn.predict("data/train/0/0027.png")
    print(len(probs))
    print("Prediction Label:", label)
    print("probabilities:", np.round(probs, 5))
