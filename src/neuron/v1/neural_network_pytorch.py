import torch
import cv2
import os
import datetime

import activation_pytorch as af

class NeuralNetwork():

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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Initialize network structure
        self.total_layers = 1 + len(hidden_layers) # Input + hidden layers
        self.weights = [] # Weight matrices storage
        self.bias = [] # Bias vectors storage

        # Input to Hidden Layers Network
        self.weights.append(torch.randn(input_size, hidden_layers[0], device=self.device) * torch.sqrt(torch.tensor(2.0 / input_size)))
        self.bias.append(torch.zeros(hidden_layers[0], device=self.device))

        # Hidden Layers Network
        for i in range(len(hidden_layers)-1):
            self.weights.append(torch.randn(hidden_layers[i], hidden_layers[i+1], device=self.device) * torch.sqrt(torch.tensor(2.0 / hidden_layers[i])))
            self.bias.append(torch.zeros(hidden_layers[i+1], device=self.device))

        # Hidden Layers Network to Output
        self.weights.append(torch.randn(hidden_layers[-1], output_size, device=self.device) * 0.01)
        self.bias.append(torch.zeros(output_size, device=self.device))
    
    # ---------------------------
    # Save
    # ---------------------------
    def save(self, filename, asNumpy=False):
        if asNumpy:
            import numpy as np

            weights_np = [w.detach().cpu().numpy() for w in self.weights]
            bias_np    = [b.detach().cpu().numpy() for b in self.bias]

            np.savez(filename, weights=np.array(weights_np, dtype=object),
                    bias=np.array(bias_np, dtype=object))
        else:
            torch.save({
                'weights': self.weights,
                'bias': self.bias
            }, filename) 
        print("Weights are stored")

    # ---------------------------
    # Load
    # ---------------------------
    def load(self, filename):
        data = torch.load(filename)
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
        arr = torch.tensor(img_blur, dtype=torch.float32, device=self.device).view(-1)

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
    def forward(self, x, training=False):
        # Calculation of layers plus bias
        Z = []

        # Applying of activation functions
        A = []

        current_activation = (x).to(self.device)

        for i in range(self.total_layers):

            z = torch.matmul(current_activation, self.weights[i]) + self.bias[i]
            Z.append(z)

            if i == self.total_layers - 1:
                a = torch.softmax(z, dim=1)
            else:
                a = torch.relu(z)

                # Apply dropout to hidden layers during the training mode and layer is not input layer
                if training and self.dropout_rate > 0:
                    # Create a random mask with values between 0 or 1
                    # Each neuron is dropped (set to 0) with probability of dropout_rate
                    dropout_mask = (torch.rand(*a.shape, device=a.device) > self.dropout_rate).float()
                    # Multiply activations by the mask
                    # This "turns off" (zeros out) the neurons that are dropped
                    a *= dropout_mask
                    # Scale remaining activations to keep the expected output the same
                    # Prevents shrinking of activation values due to dropout
                    a /= (1.0 - self.dropout_rate)
                    
            A.append(a)
            current_activation = a

        return Z, A

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

            # dz
            if i > 0:
                dz = torch.matmul(dz, self.weights[i].T) * af.relu_deriv(Z[i-1])
        
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
            indices = torch.randperm(m)
            num_batches = m // self.batch_size
            batches = torch.split(indices, num_batches)

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

        x = normalized_array.unsqueeze(0)
        _, A = self.forward(x, training=False)
        probs = A[-1][0]
        label = int(torch.argmax(probs))
        return label, probs
    
    # ---------------------------
    # Validate Training
    # ---------------------------
    def __validate_check(self, X, Y):
        _, A = self.forward(X)
        Y = Y.to(self.device)
        loss = af.cross_entropy(A[-1], Y)
        preds = torch.argmax(A[-1], axis=1)
        truth = torch.argmax(Y, axis=1)
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
        if epoche % 10 == 0 or epoche == epochs:
            print()

if __name__ == "__main__":

    nn = NeuralNetwork(
        input_size=784,
        hidden_layers=[512, 256],
        output_size=10,
        episoden=5000,
        lr=0.001,
        decay=0.0005,
        min_lr=0.005,
        batch_size=64,
        dropout_rate=0.01
    )

    X, Y = nn.load_data(['data/0', 'data/1', 'data/2', 'data/3', 'data/4', 'data/5', 'data/6', 'data/7', 'data/8', 'data/9'])

    print("X shape:", X.shape)   # sollte (N, 784)
    print("Y shape:", Y.shape)   # sollte (N, 10)
    print("min/max X:", X.min(), X.max())
    print("unique labels count:", torch.unique(torch.argmax(Y, axis=0), return_counts=True))

    # nn.load("models/nn_number_detector_0004.npz")
    label, probs = nn.predict("data/0/0027.png")
    print(len(probs))
    print("Prediction Label:", label)
    print("probabilities:", torch.round(probs * 1e5) / 1e5)

    nn.train(X, Y)

    nn.save("models/nn_number_detector_0004.npz")
    nn.save("models/nn_number_detector_numpy_0004.npz", True)
    label, probs = nn.predict("data/2/0027.png")
    print(len(probs))
    print("Prediction Label:", label)
    print("probabilities:", torch.round(probs * 1e5) / 1e5)