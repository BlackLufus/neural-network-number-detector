import cv2
import torch
import datetime
import os

import activation_pytorch as af

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

        # Dynamically load activation functions
        self.__activation_function = getattr(af, self.activation_name)
        self.__activation_deriv_func = getattr(af, f"{self.activation_name}_deriv")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Initialize network structure
        self.total_layers = 1 + len(hidden_layers) # Input + hidden layers
        self.weights = [] # Weight matrices storage
        self.bias = [] # Bias vectors storage

        # Input to Hidden Layers Network
        self.weights.append(0.01 * torch.randn(input_size, hidden_layers[0], device=self.device))
        self.bias.append(torch.zeros(hidden_layers[0], device=self.device))

        # Hidden Layers Network
        for i in range(len(hidden_layers)-1):
            self.weights.append(0.01 * torch.randn(hidden_layers[i], hidden_layers[i+1], device=self.device))
            self.bias.append(torch.zeros(hidden_layers[i+1], device=self.device))

        # Hidden Layers Network to Output
        self.weights.append(0.01 * torch.randn(hidden_layers[-1], output_size, device=self.device))
        self.bias.append(torch.zeros(output_size, device=self.device))

        print(len(self.weights[:3]))
        print(len(self.bias[:3]))

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

    def test_data_loading(self):
        X, Y = self.load_data()
        print(f"Data shapes - X: {X.shape}, Y: {Y.shape}")
        print(f"X range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"Y sample: {Y[0]}")
        
        # Test forward pass mit kleinen Daten
        test_output, test_Z, test_A = self.forward(X[:5])  # Nur 5 samples testen
        print(f"Forward pass successful! Output shape: {test_output.shape}")
        
        return X, Y

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
                    normalized_array = self.__get_normalized_image(os.path.join(folder, file))

                    # Create one-hot encoded label vector
                    one_hot = torch.zeros(len(folders))
                    one_hot[index] = 1  # Set the corresponding class index to 1

                    X.append(normalized_array)
                    Y.append(one_hot)
        return torch.stack(X), torch.stack(Y)
    
    # ---------------------------
    # Forward
    # ---------------------------
    def forward(self, x):

        Z, A = [], []

        x = x.to(self.device)

        for i in range(self.total_layers):
            # print(self.bias)
            input_tensor = x if i == 0 else A[i-1]

            z = torch.matmul(input_tensor, self.weights[i].T) + self.bias[i]
            Z.append(z)

            if i == self.total_layers - 1:
                a = torch.softmax(z, dim=1)
            else:
                a = self.__activation_function(z)

                if self.training:
                    mask = (torch.rand_like(a) > self.dropout_rate).float().to(self.device)
                    a = a * mask / (1.0 - self.dropout_rate)
            
            A.append(a)
        
        return A[-1], Z, A  # Rückgabe für backward pass

    # ---------------------------
    # Backward
    # ---------------------------
    def backward(self, X, Y, Z: list, A: list):
        m = X.shape[0]

        dW = [None] * self.total_layers
        db = [None] * self.total_layers

        Y = Y.to(self.device)

        dz = A[-1] - Y

        for i in reversed(range(self.total_layers)):
            # dW
            A_prev = X if i == 0 else A[i-1]
            dW[i] = (1/m) * torch.matmul(A_prev.T, dz)

            # db
            db[i] = (1/m) * torch.sum(dz, axis=0)

            if i > 0:
                dz = torch.matmul(dz, self.weights[i].T) * self.__activation_deriv_func(Z[i-1])
        
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
    def train(self, X, Y, epochs=5000, batch_size=32):

        num_samples = X.shape[0] # (n_examples, n_features)
        self.start_time = datetime.datetime.now()

        print(f"🚀 Starting training with {num_samples} samples, batch_size={batch_size}")

        for epoch in range(epochs):
            
            # Shuffle the training examples randomly for this epoch
            permutation = torch.randperm(num_samples)
            X_shuffled = X[permutation]
            Y_shuffled = Y[permutation]

            # Process data in mini-batches
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size] # Shape: (batch_size, input_size)
                Y_batch = Y_shuffled[i:i + batch_size] # Shape: (batch_size, num_classes)

                output, Z, A = self.forward(X_batch) # (output, Z, A)
                self.backward(X_batch, Y_batch, Z, A)
            
            # Reduce learning rate
            self.lr = max(self.min_lr, self.lr / (1 + self.decay * (epoch + 1)))

            # Validation Check
            loss, acc = self.__validate_check(X, Y)
            self.__dump_train_info(epoche=epoch+1, epochs=epochs, loss=loss, acc=acc)
        
        print(f"✅ Training completed!")

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

    print("X shape:", X.shape)   # sollte (784, N)
    print("Y shape:", Y.shape)   # sollte (10, N)
    print("min/max X:", X.min(), X.max())
    print("unique labels count:", torch.unique(torch.argmax(Y, axis=0), return_counts=True))
    
    # nn.load("models/nn_number_detector_large_extra_0009.npz")
    
    nn.train(X, Y, epochs=5000, batch_size=32)

    nn.save("models/nn_number_detector_pytorch_0003.npz")
    nn.save_npz("models/nn_number_detector_pytorch_numpy_0003.npz")

    label, probs = nn.predict("data/0/0027.png")
    print(len(probs))
    print("Prediction Label:", label)
    print("probabilities:", torch.round(probs * 1e5) / 1e5)
