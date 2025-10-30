import cv2
import torch
import datetime
import os

import torch_activation_function as af

class NeuralNetworkPyTorch():

    # ---------------------------
    # Network Parameter
    # ---------------------------
    W = []
    b = []

    # ---------------------------
    # Initinal Function
    # ---------------------------
    def __init__(self, input_size, hidden_layers, output_size, lr=0.025, decay=0.001, min_lr=0.0001, dropout_rate=0.3, activation='relu'):
        # Network architecture parameters
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        # Learning rate configuration with decay
        self.lr = lr # Initial learning rate
        self.decay = decay # Learning rate decay factor
        self.min_lr = min_lr # Minimum learning rate threshold
        self.dropout_rate = dropout_rate
        
        # Activation and regularization settings
        self.activation_name = activation

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Dynamically load activation functions
        self.__activation_function = getattr(af, self.activation_name)
        self.__activation_deriv_func = getattr(af, f"{self.activation_name}_deriv")
        
        # Initialize network structure
        self.total_layers = 1 + len(hidden_layers) # Input + hidden layers
        self.weights = [] # Weight matrices storage
        self.bias = [] # Bias vectors storage

        # Input to Hidden Layers Network
        self.weights.append(0.01 * torch.randn(hidden_layers[0], input_size, device=self.device))
        self.bias.append(torch.zeros((hidden_layers[0], 1), device=self.device))

        # Hidden Layers Network
        for i in range(len(hidden_layers)-1):
            self.weights.append(0.01 * torch.randn(hidden_layers[i+1], hidden_layers[i], device=self.device))
            self.bias.append(torch.zeros((hidden_layers[i+1], 1), device=self.device))

        # Hidden Layers Network to Output
        self.weights.append(0.01 * torch.randn(output_size, hidden_layers[-1], device=self.device))
        self.bias.append(torch.zeros((output_size, 1), device=self.device))

    # ---------------------------
    # Save
    # ---------------------------
    def save(self, filename):
        torch.save({
            'weights': self.weights,
            'bias': self.bias
        }, filename) 
        print("Weights are stored")

    # ---------------------------
    # Load
    # ---------------------------
    def load(self, filename):
        data = torch.load(filename, map_location=self.device)
        self.weights = list(data['weights'])
        self.bias = list(data['bias'])
        print("Weights are loaded")

    # ---------------------------
    # Save as Numpy
    # ---------------------------
    def save_npz(self, filename):
        import numpy as np
        # Konvertiere jeden Tensor zu NumPy Array (auf CPU)
        weights_np = [w.detach().cpu().numpy() for w in self.weights]
        bias_np    = [b.detach().cpu().numpy() for b in self.bias]

        # Speichern als .npz
        np.savez(filename, weights=np.array(weights_np, dtype=object),
                bias=np.array(bias_np, dtype=object))

        print(f"Weights and bias saved to {filename}.npz")

    # ---------------------------
    # Split Date to Train, Test and Validation
    # ---------------------------
    def train_val_test_split(self, X, Y, train_ratio=0.8, val_ratio=0.1):
        # Total number of samples
        m = X.shape[1]

        # Shuffle indices for random train/val/test split
        indices = torch.permute(m)

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
    def __get_normalized_image(self, image: str | cv2.Mat | None):
        # Checks if image is a string or already an array
        if isinstance(image, str):
            # Load image in grayscale mode
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        elif img is None:
            raise Exception("None value for image is not allowed!")

        # Apply Gaussian blur to reduce noise and details
        img_blur = cv2.GaussianBlur(img, (9, 9), 2)

        # Flatten image to column vector and convert to float32
        arr = torch.tensor(img_blur, dtype=torch.float32, device=self.device).view(-1, 1)

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
                    one_hot = torch.zeros((len(folders), 1))
                    one_hot[index] = 1  # Set the corresponding class index to 1

                    X.append(normalized_array)
                    Y.append(one_hot)
        return torch.hstack(X), torch.hstack(Y)
    
    # ---------------------------
    # Forward + Backward
    # ---------------------------
    def forward(self, x):

        Z, A = [], []

        x = x.to(self.device)

        for i in range(self.total_layers):
            # print(self.bias)
            input_tensor = x if i == 0 else A[i-1]
            input_tensor = input_tensor.to(self.device) 
            Z.append(torch.matmul(self.weights[i], input_tensor) + self.bias[i],)
            A.append(torch.softmax(Z[i], dim=0) if i == self.total_layers - 1 else torch.relu(Z[i]))

            if i < self.total_layers - 1 and self.dropout_rate > 0.0:
                mask = (torch.rand(*A[i].shape) > self.dropout_rate).to(self.device)
                A[i] = A[i] * mask / (1.0 - self.dropout_rate)
        
        return Z, A

    def backward(self, X, Y, Z: list, A: list):
        m = X.shape[1]

        dz_prev = None
        dW = [None] * self.total_layers
        db = [None] * self.total_layers

        Y = Y.to(self.device)

        for i in reversed(range(self.total_layers)):
            # dz
            Y = Y.to(self.device)
            dz = A[i] - Y if dz_prev is None else torch.matmul(self.weights[i+1].T, dz_prev) * self.__activation_deriv_func(Z[i])
            dz_prev = dz

            # dW
            A_prev = X if i == 0 else A[i-1]
            dW[i] = (1/m) * torch.matmul(dz, A_prev.T)

            # db
            db[i] = (1/m) * torch.sum(dz, axis=1, keepdims=True)
        
        # Update weights and biases
        for i in reversed(range(self.total_layers)):
            self.weights[i] -= self.lr * dW[i]
            self.bias[i] -= self.lr * db[i]

    # ---------------------------
    # Validate Training
    # ---------------------------
    def __validate_check(self, X, Y):
        _, A = self.forward(X)
        Y = Y.to(self.device)
        loss = af.cross_entropy(A[-1], Y)
        preds = torch.argmax(A[-1], axis=0)
        truth = torch.argmax(Y, axis=0)
        acc = (preds == truth).float().mean() * 100

        return loss, acc
    
    # ---------------------------
    # Print Data
    # ---------------------------
    def __dump_train_info(self, epoche, epochs, loss, acc):
        time_passed = datetime.datetime.now() - self.start_time
        minutes, seconds = divmod(time_passed.total_seconds(), 60)
        hours, minutes = divmod(minutes, 60)
        bar_length = 32
        progress = epoche / epochs
        filled_length = int(bar_length * progress)
        progress *= 100
        bar_length = 32
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f"Epoche {epoche:4d}/{epochs}: {progress:3.0f}%[{bar:<32}] [{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}, loss={loss:10.4f}, acc={acc:3.2f}%]\r", end="")

    # ---------------------------
    # Training
    # ---------------------------
    def train(self, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=5000, batch_size=32):

        m = X_train.shape[1] # (n_features, n_examples)

        self.start_time = datetime.datetime.now()

        for epoch in range(epochs):
            
            # Shuffle the training examples randomly for this epoch
            permutation = torch.randperm(m)
            X_shuffled = X_train[:, permutation]
            Y_shuffled = Y_train[:, permutation]

            # Process data in mini-batches
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[:, i:i + batch_size]
                Y_batch = Y_shuffled[:, i:i + batch_size]

                Z, A = self.forward(X_batch)
                self.backward(X_batch, Y_batch, Z, A)
            
            # Reduce learning rate
            self.lr = max(self.min_lr, self.lr / (1 + self.decay))

            # Validation Check
            # if (epoch + 1) % 50 == 0:
            loss, acc = self.__validate_check(X_train, Y_train)
            self.__dump_train_info(epoche=epoch+1, epochs=epochs, loss=loss, acc=acc)

    # ---------------------------
    # Predict
    # ---------------------------
    def predict(self, image:str|cv2.Mat | None):
        
        normalized_array = self.__get_normalized_image(image)

        _, A = self.forward(normalized_array)
        label = int(torch.argmax(A[-1], axis=0))
        probs = A[-1].flatten()  # in 1D Array umwandeln, 10 Elemente
        return label, probs

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":

    nn = NeuralNetworkPyTorch(
        input_size=784,
        hidden_layers=[512, 256],
        output_size=10,
        lr=0.01,
        decay=0.001,
        min_lr=0.005,
        activation='relu'
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
    print("unique labels count:", torch.unique(torch.argmax(Y_train, axis=0), return_counts=True))
    
    # nn.load("models/nn_number_detector_large_extra_0009.npz")
    
    nn.train(X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=5000, batch_size=32)

    nn.save("models/nn_number_detector_pytorch_0003.npz")
    nn.save_npz("models/nn_number_detector_pytorch_numpy_0003.npz")

    label, probs = nn.predict("data/0/0027.png")
    print(len(probs))
    print("Prediction Label:", label)
    print("probabilities:", torch.round(probs * 1e5) / 1e5)
