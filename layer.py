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
        return a
