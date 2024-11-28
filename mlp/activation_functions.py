import numpy as np

def tanh(x):
    """Tanh activation function."""
    return np.tanh(x)

def dtanh(x):
    """Derivative of tanh activation function."""
    return 1.0 - x ** 2

def softmax(x):
    """Softmax activation function."""
    e = np.exp(x - np.max(x))
    return e / np.sum(e)
