### README: **MLP Neural Network from Scratch**

---

For a setp-by-step explanation, see the jupyter notebook `MLP.ipynb`.

### Overview

This project implements a **Multi-Layer Perceptron (MLP)** from scratch in Python, without using any external machine learning frameworks (e.g., TensorFlow, PyTorch). The MLP supports key features such as:
- **Tanh** and **Softmax** activation functions.
- **Dropout regularization** (training and testing modes).
- **L2 regularization**.
- Backpropagation with weight updates.
- Training and evaluation pipelines with a custom dataset.

The project structure is modular, with classes, utilities, and scripts split into separate files for maintainability.

---

### Project Structure

```
mlp_project/
├── README.md             # Project documentation
├── main.py               # Main script to train and evaluate the MLP
├── requirements.txt      # List of dependencies
├── data/
│   └── task1.csv         # Example dataset
├── mlp/
│   ├── __init__.py       # Marks the directory as a package
│   ├── activations.py    # Activation functions (tanh, softmax) and their derivatives
│   ├── dropout.py        # Dropout-related utilities
│   ├── model.py          # Implementation of the MLP class
│   ├── training.py       # Training and learning functions (with/without dropout)
│   ├── utils.py          # Helper functions (e.g., data pre-processing, one-hot encoding)
│   ├── evaluation.py     # ROC curve generation and test set evaluation
└── tests/
    ├── test_mlp.py       # Unit tests for the MLP class
    └── test_utils.py     # Unit tests for utility functions
```

---

### Prerequisites

Before running the project, ensure you have Python installed (version 3.7 or higher). The following libraries are required:
- **NumPy**: Numerical computations.
- **Pandas**: Data manipulation.
- **Matplotlib**: Visualization (for ROC curves).
- **scikit-learn**: Evaluation metrics like ROC and AUC.

Install the dependencies using:
```bash
pip install -r requirements.txt
```

---

### How to Use

#### 1. Prepare the Dataset
Place your dataset in the `data/` directory. An example dataset (`task1.csv`) is included. The dataset should have features as columns and the target label as the last column.

#### 2. Run the Training Script
To train the model without dropout or L2 regularization:
```bash
python main.py --mode train --regularization none
```

To train with L2 regularization:
```bash
python main.py --mode train --regularization l2 --lambda 0.1
```

To train with dropout:
```bash
python main.py --mode train --regularization dropout --dropout_input 0.95 --dropout_hidden 0.85
```

#### 3. Evaluate the Model
After training, evaluate the model using:
```bash
python main.py --mode test
```

#### 4. Visualize ROC Curves
The script generates and displays ROC curves to evaluate the model’s performance.

---

### Key Features

1. **Customizable MLP**:
   - Define input size, hidden layers, and output size easily in `MLP` initialization.

2. **Dropout Regularization**:
   - Implemented for both training and testing modes to prevent overfitting.

3. **L2 Regularization**:
   - Penalizes large weights during training for better generalization.

4. **Evaluation with ROC Curves**:
   - Uses scikit-learn's metrics to plot ROC curves and calculate AUC for multi-class classification.

5. **Pre-processing and Normalization**:
   - Normalizes and scales inputs for better convergence during training.

---

### Example Workflow

1. **Data Preprocessing**:
   - Normalize inputs and one-hot encode outputs.
   - Split the dataset into training and test sets.

2. **Model Initialization**:
   - Create an MLP instance with the desired architecture (e.g., `MLP(2, 20, 20, 3)` for 2 input nodes, two hidden layers with 20 nodes each, and 3 output nodes).

3. **Training**:
   - Train the model using backpropagation with or without regularization.

4. **Testing and ROC Curve Generation**:
   - Evaluate the model on a test set and visualize its performance.

---

### Example Commands

Train a network with 2 inputs, 20 hidden nodes in two layers, and 3 outputs:
```python
from mlp.model import MLP
from mlp.training import learn
from mlp.utils import import_data, pre_proc_input, one_hot_encode

# Load data
data = import_data("data/task1.csv")

# Pre-process and split data
train_data, test_data = split_data(data)
inputs, mean, std = pre_proc_input(train_data)
outputs = one_hot_encode(train_data)

# Initialize model
mlp = MLP(2, 20, 20, 3)

# Train the model
learn(mlp, train_samples)
```

---

### Future Improvements

1. **Extend Dataset Support**:
   - Add support for different dataset formats (e.g., JSON, Excel).

2. **Additional Features**:
   - Add more activation functions (ReLU, Leaky ReLU).
   - Implement early stopping for training.

3. **Optimization**:
   - Improve performance with vectorized computations.

4. **Model Persistence**:
   - Save and load trained models for reuse.

---

### Contributors

- **Author**: David Jimenez Barrero.

---

### License

This project is licensed under the MIT License. See the LICENSE file for details.