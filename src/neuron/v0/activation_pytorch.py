import numpy as np
import torch

# ==== ACTIVATION FUNCTIONS ====

def relu(x):
    """ReLU (Rectified Linear Unit) – sets all negative values to zero."""
    return torch.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    """Leaky ReLU – allows a small gradient when x < 0 to prevent dead neurons."""
    return np.where(x > 0, x, alpha * x)


def sigmoid(x):
    """Sigmoid – squashes input into range (0, 1), used for binary outputs."""
    return 1 / (1 + np.exp(-x))


def tanh(x):
    """Tanh – squashes input into range (-1, 1), centered around zero."""
    return np.tanh(x)


def elu(x, alpha=1.0):
    """ELU (Exponential Linear Unit) – smooth negative side, helps with vanishing gradients."""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


# ==== DERIVATIVES ====

def relu_deriv(x):
    """Derivative of ReLU – gradient is 1 for x > 0, else 0."""
    return (x > 0).float()


def leaky_relu_deriv(x, alpha=0.01):
    """Derivative of Leaky ReLU – small gradient (alpha) for x < 0."""
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx


def sigmoid_deriv(x):
    """Derivative of Sigmoid – s(x) * (1 - s(x))."""
    s = 1 / (1 + np.exp(-x))
    return s * (1 - s)


def tanh_deriv(x):
    """Derivative of Tanh – 1 - tanh(x)^2."""
    return 1 - np.tanh(x) ** 2


def elu_deriv(x, alpha=1.0):
    """Derivative of ELU – 1 for x > 0, alpha * exp(x) for x <= 0."""
    dx = np.ones_like(x)
    dx[x <= 0] = alpha * np.exp(x[x <= 0])
    return dx


def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp, axis=0, keepdims=True)

# def softmax(z):
#     # Subtrahiere das Maximum in jeder Spalte
#     z_max = np.max(z, axis=0, keepdims=True)
#     exp = np.exp(z - z_max)
#     return exp / np.sum(exp, axis=0, keepdims=True)

def cross_entropy(pred, target):
    return -torch.sum(target * torch.log(pred + 1e-9))