import numpy as np
from mlp_project.mlp import MLP
from mlp_project.training import train
from mlp_project.utils import one_hot_encode, plot_roc_curve
from mlp_project.data_processing import load_data, preprocess_data

if __name__ == "__main__":
    data = load_data("data/task1.csv")
    inputs, mean, std = preprocess_data(data.iloc[:, :-1])
    outputs = one_hot_encode(data.iloc[:, -1].values, num_classes=3)

    samples = np.zeros(len(inputs), dtype=[('input', float, inputs.shape[1]), ('output', float, 3)])
    samples['input'] = inputs
    samples['output'] = outputs

    mlp = MLP(2, 20, 20, 3)
    train(mlp, samples, epochs=50, lrate=0.1)

    predictions = np.array([mlp.propagate_forward(s['input']) for s in samples])
    plot_roc_curve(outputs, predictions, 3, "MLP ROC Curve")
