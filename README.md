# Number Detector (Neural Network)

This project serves to investigate **neural networks**. For this purpose, a small model was developed that can classify painted numbers.

The project itself was investigated and implemented in both `NumPy` and `PyTorch`. `NumPy` does not require a special graphics card (**GPU**), just a reasonably up-to-date CPU. Nevertheless, I was very curious to test the whole thing with a GPU as well, partly to see where the limits of both methods lie. Since I only had a current **AMD graphics card** (RX 9070 XT) available, I could only train on that. To do this, the graphics card first had to be made accessible via `ROCm`. `ROCm` provides an interface to the graphics card to enable calculations to be performed directly on the graphics card – similar to what `CUDA` does for Nvidia GPUs. Since Windows was not officially supported at the time, the experiment was carried out using the Windows Subsystem for Linux (**WSL**).

## 🔬 Install PyTorch

To install `PyTorch`, it was first necessary to create a current `Linux` operating system. `Ubuntu 24.04` was used for this purpose, as Linux itself officially supports the `ROCm` interface. However, the whole thing still ran under Windows, which is why the appropriate drivers had to be preinstalled.

### 💻 Install Radeon software for WSL with ROCm

The ROCm™ Software Stack and other Radeon™ software for Windows Subsystem for Linux (WSL) components are installed using the amdgpu-install script to assist you in the installation of a coherent set of stack components.

> [Install Radeon software for WSL with ROCm](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/install-radeon.html)

### ⚙️ Install PyTorch for ROCm

With the help of the instructions provided by **AMD**, the software for `PyTorch` could then be installed.

> [Install PyTorch for ROCm on wsl systems](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/install-pytorch.html)

`PyTorch` can be installed and used on various `Linux` distributions. Depending on your system and compute requirements, your experience with `PyTorch` on Linux may vary in terms of processing time.

> [Installing PyTorch on Linux (General)](https://pytorch.org/get-started/locally/)

## 📦 Install project requirements over pip

For the successful execution of the project, some **requirements** are necessary that are installed via `PIP`.

```bash
pip install -r requirements.txt
```

## 🚀 Launch project

The project itself runs on `Python 3.12.3`. In addition, the project contains several versions. The old versions are for reference only and are no longer up to date. The current version is `v1` and is structured as follows:

```bash
neural_network_number_detector
├── data/
├── models/
├── src/
├── neuron/
│   ├── v0/
│   │   ├── ...
│   └── v1/
│       ├── activation_numpy.py
│       ├── activation_pytorch.py
│       ├── canvas.py
│       ├── neural_network_numpy.py
│       └── neural_network_pytorch.py
├── temp/temp.png
├── venv/
├── .gitignore
├── requirements.txt
└── README.md
```

### 🧩 Activation

The Activation module contains all activation functions used in the neural network (e.g., `ReLU`, `Sigmoid`, `Softmax`, `Tanh`, etc.).  
These are modular in design and can be easily expanded or customized.

### 🧠 Neural network

The NeuralNetwork module implements a complete neural network that has been implemented with both `NumPy` (**neural_network_numpy.py**) and `PyTorch` (**neural_network_pytorch.py**).  
It provides all the necessary methods for training, saving, loading, and making predictions with models.

#### ✨ Initialization

```python
from neural_network_numpy import NeuralNetwork

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
```

#### 📂 Load Data

The `load_data()` method can be used to load image data including their associated labels.
Each subfolder corresponds to a label:

```python
X, Y = nn.load_data([
    'data/0', 'data/1', 'data/2', 'data/3', 'data/4',
    'data/5', 'data/6', 'data/7', 'data/8', 'data/9'
])
```

**Return**:

- `X`: List of input data (features)
- `Y`: List of associated labels

#### 💾 Load and Save Model

Trained models can be saved using the `save()` method and reloaded using `load()`.

```python
# Loading an existing model
nn.load("models/model_0001.npz")

# Training the model with new data
nn.train(X, Y)

# Saving the trained model
nn.save("models/model_0001.npz")
```

#### 🧮 Training

The training is started using the `train()` method.
Here, the input and output data that has been transferred is processed iteratively in order to optimize the network weights.

```python
nn.train(X, Y)
```

#### 🔍 Prediction

After training, the network can predict new data.
The `predict()` method accepts either a file path string or an already loaded cv2.Mat image object.

```python
result = nn.predict("images/sample_digit.png")
```

Return:

- The predicted label or class of the input image.

### 🎨 Canvas

The Canvas module can be used to interact with the model in order to draw numbers.
It can be used either for data acquisition or for classifying a drawn number.

```python
# Start training
dc = DrawCanvas(output_folder="data/train")

# Launch Model to predict input
dc = DrawCanvas(
    dc = DrawCanvas(output_folder="data/train")
    "models/model_0001.npz",
    input_layer_size=784,
    hidden_layer_size=[256, 256],
    output_layer_size=10
)

# Run and build a canvas
dc.build_and_run()
```

You can save your input using the keyboard shortcuts `CTRL` + `S` and clear the canvas using `CTRL` + `C`.