import random
from Matrix import Matrix
from acitvation_functions import linear_activation
class Layer:
    def __init__(self, input_size, output_size, activation, backend):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.backend = backend
        weights = []
        for _ in range(output_size):
            rows = []
            for j in range(input_size):
                rows.append(random.uniform(-1, 1))
            weights.append(rows)
        self.weight_matrix = Matrix(weights)
        biases = []
        for _ in range(output_size):
            biases.append([random.uniform(-1, 1)])
        self.bias_matrix = Matrix(biases)
    def forward(self, inputs):
        z = self.backend.matmul(self.weight_matrix, inputs)
        z = self.backend.add(z, self.bias_matrix)
        a = self.backend.apply_activation(z, self.activation)
        self.last_input = inputs
        self.last_z = z
        self.last_output = a
        return a
    def backward(self, output_gradient):
        activation_derivative = self.activation.derivative(self.last_z)
        delta = self.backend.element_wise_multiply(output_gradient, activation_derivative)
        weight_gradient = self.backend.matmul(delta, self.last_input.transpose())
        bias_gradient = delta
        input_gradient = self.backend.matmul(self.weight_matrix.transpose(), delta)
        return input_gradient, weight_gradient, bias_gradient
    def update_parameters(self, weight_gradient, bias_gradient, learning_rate):
        weight_update = self.backend.scalar_multiplication(weight_gradient, -learning_rate)
        bias_update = self.backend.scalar_multiplication(bias_gradient, -learning_rate)
        self.weight_matrix = self.backend.add(self.weight_matrix, weight_update)
        self.bias_matrix = self.backend.add(self.bias_matrix, bias_update)