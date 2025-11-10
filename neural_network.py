import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import logging

# Initialize nnfs
nnfs.init()

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases
        logger.info(f"Layer_Dense outputs:\n{self.outputs}")

class Activation_ReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)
        logger.info(f"ReLU outputs:\n{self.outputs}")

class Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = probabilities
        logger.info(f"Softmax outputs:\n{self.outputs}")

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

# Dynamic dataset
samples = 100
classes = 3
X, y = spiral_data(samples=samples, classes=classes)

# Network configuration (can be dynamic)
input_size = X.shape[1]
hidden_neurons = 5  # Configurable
output_neurons = len(np.unique(y))

# Layers
dense1 = Layer_Dense(input_size, hidden_neurons)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(hidden_neurons, output_neurons)
activation2 = Softmax()

# Forward pass
dense1.forward(X)
activation1.forward(dense1.outputs)
dense2.forward(activation1.outputs)
activation2.forward(dense2.outputs)

# Loss calculation
lossfunc = Loss_CategoricalCrossentropy()
loss = lossfunc.calculate(activation2.outputs, y)
logger.info(f"Loss: {loss}")
