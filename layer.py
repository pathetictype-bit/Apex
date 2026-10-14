import math
import random
from Matrix import Matrix
class Layer:
    def __init__(self, input_size, output_size, activation, backend):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.backend = backend
        limit = math.sqrt(
            6 / (input_size + output_size)
        )
        weights = []
        for _ in range(output_size):
            rows = []
            for _ in range(input_size):
                rows.append(random.uniform(-limit, limit))
            weights.append(rows)
        self.weight_matrix = backend.create_matrix(weights)
        biases = []
        for _ in range(output_size):
            biases.append([random.uniform(-1, 1)])
        self.bias_matrix = backend.create_matrix(biases)
    def forward(self, inputs):
        z = self.backend.matmul(self.weight_matrix, inputs)
        z = self.backend.add(z, self.bias_matrix)
        if self.activation.__class__.__name__ == "method":
            a = self.backend.apply_softmax(z)
        else:
            a = self.backend.apply_activation(z, self.activation)

        cache = (inputs, z)
        return a, cache
    def backward(self, output_gradient, cache):
        input_cache, z_cache = cache
        # Check if activation is a backend method (like apply_softmax) or an activation instance
        is_softmax = (
            hasattr(self.activation, '__name__') and 
            self.activation.__name__ == "apply_softmax"
        ) or (
            hasattr(self.activation, '__class__') and 
            self.activation.__class__.__name__ == "method"
        )
        
        if is_softmax:
            delta = output_gradient
        else:
            activation_derivative = self.backend.apply_activation_derivative(z_cache, self.activation)
            delta = self.backend.element_wise_multiply(output_gradient, activation_derivative)
        weight_gradient = self.backend.matmul(delta, input_cache.transpose())
        bias_gradient = delta
        input_gradient = self.backend.matmul(self.weight_matrix.transpose(), delta)
        return input_gradient, weight_gradient, bias_gradient
    def update_parameters(self, weight_gradient, bias_gradient, learning_rate):
        weight_update = self.backend.scalar_multiplication(weight_gradient, -learning_rate)
        bias_update = self.backend.scalar_multiplication(bias_gradient, -learning_rate)
        self.weight_matrix = self.backend.add(self.weight_matrix, weight_update)
        self.bias_matrix = self.backend.add(self.bias_matrix, bias_update)