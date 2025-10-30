import numpy as np
import cv2
import os
import datetime

import activation_numpy as af

class NeuralNetwork:

    def __init__(self, input_size, hidden_layers, output_size, episoden=1000, lr=0.01, decay=0.0001, min_lr=0.001, batch_size=64, dropout_rate=0.2):
        # Network architecture parameters
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        # Learning rate configuration with decay
        self.episoden = episoden
        self.lr = lr # Initial learning rate
        self.decay = decay # Learning rate decay factor
        self.min_lr = min_lr # Minimum learning rate threshold
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate

        # Initialize network structure
        self.total_layers = 1 + len(hidden_layers) # Input + hidden layers
        self.weights = [] # Weight matrices storage
        self.bias = [] # Bias vectors storage

        # Input to Hidden Layers Network
        self.weights.append(np.random.randn(input_size, hidden_layers[0]) * np.sqrt(2.0 / input_size))
        self.bias.append(np.zeros(hidden_layers[0]))

        # Hidden Layers Network
        for i in range(len(hidden_layers)-1):
            self.weights.append(np.random.randn(hidden_layers[i], hidden_layers[i+1]) * np.sqrt(2.0 / hidden_layers[i]))
            self.bias.append(np.zeros(hidden_layers[i+1]))

        # Hidden Layers Network to Output
        self.weights.append(np.random.randn(hidden_layers[-1], output_size) * 0.01)
        self.bias.append(np.zeros(output_size))
    
    # ---------------------------
    # Save
    # ---------------------------
    def save(self, filename):
        np.savez(filename, weights=np.array(self.weights, dtype=object), bias=np.array(self.bias, dtype=object))
        print("Weights are stored")

    # ---------------------------
    # Load
    # ---------------------------
    def load(self, filename):
        data = np.load(filename, allow_pickle=True)
        self.weights = list(data['weights'])
        self.bias = list(data['bias'])
        print("Weights are loaded")
    
    # ---------------------------
    # Normalize Image
    # ---------------------------
    def __get_normalized_image(self, image: str | cv2.Mat | None):
        # Checks if image is a string or already an array
        if isinstance(image, str):
            # Load image in grayscale mode
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        elif image is None:
            raise Exception("None value for image is not allowed!")

        # Apply Gaussian blur to reduce noise and details
        img_blur = cv2.GaussianBlur(img, (9, 9), 2)

        # Flatten image to column vector and convert to float32
        arr = img_blur.reshape(-1).astype(np.float32)

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
                    one_hot = np.zeros(len(folders))
                    one_hot[index] = 1  # Set the corresponding class index to 1

                    X.append(normalized_array)
                    Y.append(one_hot)
        return np.array(X), np.array(Y)

    # ---------------------------
    # Forward
    # ---------------------------
    def forward(self, x, training=False):
        Z, A = [], []

        current_input = x

        for i in range(self.total_layers):

            z = np.matmul(current_input, self.weights[i]) + self.bias[i]
            Z.append(z)

            if i == self.total_layers - 1:
                a = af.softmax(z)
            else:
                a = af.relu(z)

                # 👉 Dropout nur während Training:
                if training and self.dropout_rate > 0:
                    dropout_mask = (np.random.rand(*a.shape) > self.dropout_rate).astype(np.float32)
                    a *= dropout_mask                     # "Ausschalten" zufälliger Neuronen
                    a /= (1.0 - self.dropout_rate)           # Skalieren, damit Erwartungswert gleich bleibt
                    
            A.append(a)
            current_input = a

        return Z, A

    # ---------------------------
    # Backward
    # ---------------------------
    def backward(self, X, Y, Z: list, A: list):

        m = X.shape[0]

        dW = [None] * self.total_layers
        db = [None] * self.total_layers

        dz = A[-1] - Y

        for i in reversed(range(self.total_layers)):
            # dW
            A_prev = X if i == 0 else A[i-1]
            dW[i] = (1/m) * np.matmul(A_prev.T, dz)

            # db
            db[i] = (1/m) * np.sum(dz, axis=0)

            # dz
            if i > 0:
                dz = np.matmul(dz, self.weights[i].T) * af.relu_deriv(Z[i-1])
        
        # Update weights and biases
        for i in reversed(range(self.total_layers)):
            self.weights[i] -= self.lr * dW[i]
            self.bias[i] -= self.lr * db[i]
    
    # ---------------------------
    # Training
    # ---------------------------
    def train(self, X, Y):
        self.start_time = datetime.datetime.now()

        m = X.shape[0] # (n_examples, n_features)

        for episode in range(self.episoden):
            # Shuffle the training examples randomly for this epoch
            indices = np.arange(m)
            np.random.shuffle(indices)

            num_batches = m // self.batch_size
            batches = np.array_split(indices, num_batches)

            # Process data in mini-batches
            for batch_idx in batches:
                X_batch = X[batch_idx]
                Y_batch = Y[batch_idx]

                Z, A = self.forward(X_batch, True)
                self.backward(X_batch, Y_batch, Z, A)
            
            # Reduce learning rate
            self.lr = max(self.min_lr, self.lr * (1 / (1 + self.decay)))

            # Validation Check
            loss, acc = self.__validate_check(X, Y)
            self.running_loss = 0.9 * self.running_loss + 0.1 * loss if hasattr(self, 'running_loss') else loss
            self.__dump_train_info(epoche=episode+1, epochs=self.episoden, loss=self.running_loss, acc=acc)

    # ---------------------------
    # Predict
    # ---------------------------
    def predict(self, image:str|cv2.Mat | None):
        
        normalized_array = self.__get_normalized_image(image)
        print(normalized_array)

        _, A = self.forward(normalized_array, training=False)
        probs = A[-1]
        label = int(np.argmax(probs))
        return label, probs
    
    # ---------------------------
    # Validate Training
    # ---------------------------
    def __validate_check(self, X, Y):
        _, A = self.forward(X)
        loss = af.cross_entropy(A[-1], Y)
        preds = np.argmax(A[-1], axis=1)
        truth = np.argmax(Y, axis=1)
        acc = np.mean(preds == truth) * 100

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
        if epoche % 10 == 0 or epoche == epochs:
            print()

if __name__ == "__main__":

    nn = NeuralNetwork(
        input_size=784,
        hidden_layers=[256, 256],
        output_size=10,
        episoden=2000,
        lr=0.01,
        decay=0.001,
        min_lr=0.005,
        batch_size=32,
        dropout_rate=0.1
    )

    X, Y = nn.load_data(['data/0', 'data/1', 'data/2', 'data/3', 'data/4', 'data/5', 'data/6', 'data/7', 'data/8', 'data/9'])

    print("X shape:", X.shape)   # sollte (N, 784)
    print("Y shape:", Y.shape)   # sollte (N, 10)
    print("min/max X:", X.min(), X.max())
    print("unique labels count:", np.unique(np.argmax(Y, axis=0), return_counts=True))

    nn.load("models/nn_number_detector_numpy_0003.npz")
    label, probs = nn.predict("data/2/0027.png")
    print(len(probs))
    print("Prediction Label:", label)
    print("probabilities:", np.round(probs, 5))

    nn.train(X, Y)

    nn.save("models/nn_number_detector_numpy_0004.npz")
    label, probs = nn.predict("data/0/0027.png")
    print(len(probs))
    print("Prediction Label:", label)
    print("probabilities:", np.round(probs, 5))