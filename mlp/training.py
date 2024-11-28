import numpy as np

def train(network, samples, epochs=100, lrate=0.1, lambda_=0):
    for epoch in range(epochs):
        np.random.shuffle(samples)
        for sample in samples:
            network.propagate_forward(sample['input'])
            network.backpropagate(sample['output'], lrate, lambda_)
