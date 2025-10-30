from typing import Any
import cv2
import numpy as np
import os

import activation_function as af

class NeuralNetwork:

    # ---------------------------
    # Network Parameter
    # ---------------------------
    W = []
    b = []

    # ---------------------------
    # Initinal Function
    # ---------------------------
    def __init__(self, input_size, hidden_layers, output_size, lr=0.025, decay=0.001, min_lr=0.0001, activation='relu', dropout_rate=0.2):
        # Network architecture parameters
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        # Learning rate configuration with decay
        self.lr = lr # Initial learning rate
        self.decay = decay # Learning rate decay factor
        self.min_lr = min_lr # Minimum learning rate threshold
        
        # Activation and regularization settings
        self.activation_name = activation
        self.dropout_rate = dropout_rate # Dropout rate for regularization

        # Dynamically load activation functions
        self.__activation_function = getattr(af, self.activation_name)
        self.__activation_deriv_func = getattr(af, f"{self.activation_name}_deriv")
        
        # Initialize network structure
        self.total_layers = 1 + len(hidden_layers) # Input + hidden layers
        self.weights = [] # Weight matrices storage
        self.bias = [] # Bias vectors storage

        # He initial random factor for weights
        init_value = np.sqrt(2. / self.input_size)

        # Input to Hidden Layers Network
        self.weights.append(init_value * np.random.randn(hidden_layers[0], input_size))
        self.bias.append(np.zeros((hidden_layers[0], 1)))

        # Hidden Layers Network
        for i in range(len(hidden_layers)-1):
            self.weights.append(init_value * np.random.randn(hidden_layers[i+1], hidden_layers[i]))
            self.bias.append(np.zeros((hidden_layers[i+1], 1)))

        # Hidden Layers Network to Output
        self.weights.append(init_value * np.random.randn(output_size, hidden_layers[-1]))
        self.bias.append(np.zeros((output_size, 1)))

    # ---------------------------
    # Save
    # ---------------------------
    def save(self, filename):
        np.savez(filename, W=np.array(self.weights, dtype=object), b=np.array(self.bias, dtype=object))
        print("Weights are stored")

    # ---------------------------
    # Load
    # ---------------------------
    def load(self, filename):
        data = np.load(filename, allow_pickle=True)
        self.weights = list(data['W'])
        self.bias = list(data['b'])
        print("Weights are loaded")

    # ---------------------------
    # Split Date to Train, Test and Validation
    # ---------------------------
    def train_val_test_split(self, X, Y, train_ratio=0.8, val_ratio=0.1):
        # Total number of samples
        m = X.shape[1]

        # Shuffle indices for random train/val/test split
        indices = np.random.permutation(m)

        # Calculate split boundaries
        train_end = int(m * train_ratio)
        val_end = int(m * (train_ratio + val_ratio))

        # Split into training set
        X_train = X[:, indices[:train_end]]
        Y_train = Y[:, indices[:train_end]]

        # Split into validation set  
        X_val = X[:, indices[train_end:val_end]]
        Y_val = Y[:, indices[train_end:val_end]]

        # Split into test set
        X_test = X[:, indices[val_end:]]
        Y_test = Y[:, indices[val_end:]]

        return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)

    ##
    def __get_normalized_image(self, image: str | cv2.Mat| np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]] | None):
        # Checks if image is a string or already an array
        if isinstance(image, str):
            # Load image in grayscale mode
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        elif img is None:
            raise Exception("None value for image is not allowed!")

        # Apply Gaussian blur to reduce noise and details
        img_blur = cv2.GaussianBlur(img, (9, 9), 2)

        # Flatten image to column vector and convert to float32
        arr = img_blur.reshape(-1, 1).astype(np.float32)

        # Normalize pixel values from [0,255] to [0,1]
        normalized_array = arr / 255

        return normalized_array

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
                    
                    # Load image in grayscale mode
                    normalized_array = self.__get_normalized_image(folder + "/" + file)

                    # Create one-hot encoded label vector
                    one_hot = np.zeros((len(folders), 1))
                    one_hot[index] = 1  # Set the corresponding class index to 1

                    X.append(normalized_array)
                    Y.append(one_hot)
        return np.hstack(X), np.hstack(Y)
    
    # ---------------------------
    # Forward + Backward
    # ---------------------------
    def forward(self, x, training=False):

        Z, A = [], []

        for i in range(self.total_layers):
            print(self.bias)
            Z.append(np.dot(self.weights[i], (x if i == 0 else A[i-1])) + self.bias[i])
            A.append(af.softmax(Z[i]) if i == self.total_layers - 1 else self.__activation_function(Z[i]))
            
            if training and i < self.total_layers - 1 and self.dropout_rate > 0.0:
                mask = np.random.rand(*A[i].shape) > self.dropout_rate
                A[i] = A[i] * mask / (1.0 - self.dropout_rate)
        
        return Z, A

    def backward(self, X, Y, Z: list, A: list):
        m = X.shape[1]

        dz_prev = None
        dW = [None] * self.total_layers
        db = [None] * self.total_layers

        for i in reversed(range(self.total_layers)):
            # dz
            dz = A[i] - Y if dz_prev is None else np.dot(self.weights[i+1].T, dz_prev) * self.__activation_deriv_func(Z[i])
            dz_prev = dz

            # dW
            A_prev = X if i == 0 else A[i-1]
            dW[i] = (1/m) * np.dot(dz, A_prev.T)

            # db
            db[i] = (1/m) * np.sum(dz, axis=1, keepdims=True)
        
        # Update weights and biases
        for i in reversed(range(self.total_layers)):
            self.weights[i] -= self.lr * dW[i]
            self.bias[i] -= self.lr * db[i]

    # ---------------------------
    # Validate Training
    # ---------------------------
    def __validate_check(self, X, Y):
        _, A = self.forward(X)
        loss = af.cross_entropy(A[-1], Y)
        preds = np.argmax(A[-1], axis=0)
        truth = np.argmax(Y, axis=0)
        acc = np.mean(preds == truth) * 100

        return loss, acc
    
    # ---------------------------
    # Print Data
    # ---------------------------
    def __dump_train_info(self, epoche, epochs, train_loss=None, train_acc=None, val_loss=None, val_acc=None, test_loss=None, test_acc=None):
        print(f"| ---------- [Epoche: {epoche}/{epochs} ({((epoche/epochs)*100):.2f}%)] ----------")
        print(f"| Learning rate:   {self.lr:10.8f}")
        if train_loss is not None:
            print(f"| Training Loss:   {train_loss:10.4f}")
        if train_acc is not None:
            print(f"| Training Acc:    {train_acc:8.2f}%")
        if val_loss is not None:
            print(f"| Validation Loss: {val_loss:10.4f}")
        if val_acc is not None:
            print(f"| Validation Acc:  {val_acc:8.2f}%")
        if test_loss is not None:
            print(f"| Test Loss:       {test_loss:10.4f}")
        if test_acc is not None:
            print(f"| Test Acc:        {test_acc:8.2f}%")
        print("| -------------------------------------------------\n")

    # ---------------------------
    # Training
    # ---------------------------
    def train(self, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=5000, batch_size=32):

        m = X_train.shape[1] # (n_features, n_examples)

        for epoch in range(epochs):
            
            # Shuffle the training examples randomly for this epoch
            permutation = np.random.permutation(m)
            X_shuffled = X_train[:, permutation]
            Y_shuffled = Y_train[:, permutation]

            # Process data in mini-batches
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[:, i:i + batch_size]
                Y_batch = Y_shuffled[:, i:i + batch_size]

                Z, A = self.forward(X_batch, training=True)
                self.backward(X_batch, Y_batch, Z, A)
            
            # Reduce learning rate
            self.lr = max(self.min_lr, self.lr / (1 + self.decay))

            # Validation Check
            if (epoch + 1) % 50 == 0:
                train_loss, train_acc = self.__validate_check(X_train, Y_train)
                if X_val is not None:
                    val_loss, val_acc = self.__validate_check(X_val, Y_val)
                else:
                    val_loss, val_acc = None, None
                self.__dump_train_info(epoche=epoch+1, epochs=epochs, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)
        
        if X_test is not None:
            test_loss, test_acc = self.__validate_check(X_test, Y_test)
        else:
            test_loss, test_acc = None, None
        self.__dump_train_info(epoche=epochs, epochs=epochs, test_loss=test_loss, test_acc=test_acc)

    # ---------------------------
    # Predict
    # ---------------------------
    def predict(self, image:str|cv2.Mat| np.ndarray[Any, np.dtype[np.integer[Any] | np.floating[Any]]] | None):
        
        normalized_array = self.__get_normalized_image(image)

        _, A = self.forward(normalized_array)
        label = int(np.argmax(A[-1], axis=0))
        probs = A[-1].flatten()  # in 1D Array umwandeln, 10 Elemente
        return label, probs

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":

    nn = NeuralNetwork(
        input_size=784,
        hidden_layers=[256, 256],
        output_size=10,
        lr=0.0075,
        decay=0.0005,
        min_lr=0.001,
        activation='relu',
        dropout_rate=0.0
    )

    X, Y = nn.load_data(['data/0', 'data/1', 'data/2', 'data/3', 'data/4', 'data/5', 'data/6', 'data/7', 'data/8', 'data/9'])
    # (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = nn.train_val_test_split(X, Y, train_ratio=0.8, val_ratio=0.1)
    X_train = X
    Y_train = Y
    X_val = None
    Y_val = None
    X_test = None
    Y_test = None

    print("X shape:", X_train.shape)   # sollte (784, N)
    print("Y shape:", Y_train.shape)   # sollte (10, N)
    print("min/max X:", X_train.min(), X_train.max())
    print("unique labels count:", np.unique(np.argmax(Y_train, axis=0), return_counts=True))
    
    # nn.load("models/nn_number_detector_large_extra_0009.npz")
    
    nn.train(X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=2000, batch_size=64)

    # nn.trainParallel(X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=2000, batch_size=128, workers=64)

    nn.save("models/nn_number_detector_0001.npz")

    nn.load("models/nn_number_detector_0001.npz")

    label, probs = nn.predict("data/0/0027.png")
    print(len(probs))
    print("Prediction Label:", label)
    print("probabilities:", np.round(probs, 5))
