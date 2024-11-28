import numpy as np
from .activation_functions import tanh, softmax, dtanh

class MLP:
    """Multi-layer perceptron class."""

    def __init__(self, *layers):
        self.shape = layers
        self.layers = [np.ones(n + (1 if i == 0 else 0)) for i, n in enumerate(layers)]
        self.weights = [np.random.uniform(-0.25, 0.25, (self.layers[i].size, self.layers[i + 1].size))
                        for i in range(len(layers) - 1)]
        self.dweights = [np.zeros_like(w) for w in self.weights]

    def propagate_forward(self, data):
        self.layers[0][:-1] = data  # input layer (excluding bias)
        for i in range(len(self.weights) - 1):
            self.layers[i + 1][...] = tanh(self.layers[i] @ self.weights[i])
        self.layers[-1][...] = softmax(self.layers[-2] @ self.weights[-1])
        return self.layers[-1]

    def backpropagate(self, target, lrate=0.1, lambda_=0):
        deltas = [self.layers[-1] - target]
        for i in range(len(self.weights) - 1, 0, -1):
            deltas.append((deltas[-1] @ self.weights[i].T) * dtanh(self.layers[i]))
        deltas.reverse()

        for i, delta in enumerate(deltas):
            self.dweights[i] = (self.layers[i][:, None] @ delta[None, :]) + lambda_ * self.weights[i]
            self.weights[i] -= lrate * self.dweights[i]
