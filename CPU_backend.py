import math
import random
from Matrix import Matrix
class CPUBackend:
    name = "CPU"

    def matmul(self, a, b):
        return a.multiply(b)

    def add(self, a, b):
        return a.add(b)

    def apply_activation(self, matrix, function):
        return matrix.apply_function(function)
    
    def element_wise_multiply(self, a, b):
        return a.element_wise_multiply(b)

    def scalar_multiplication(self, matrix, scalar):
        return matrix.scalar_multiplication(scalar)

    def apply_function(self, matrix, function):
        return matrix.apply_function(function)
    def apply_softmax(self, matrix):
        maximum = max(max(row) for row in matrix.data)
        shifted_matrix = Matrix([
            [value - maximum for value in row]
            for row in matrix.data
        ])
        exp_matrix = shifted_matrix.apply_function(math.exp)
        sum_exp = sum(sum(row) for row in exp_matrix.data)
        softmax_matrix = exp_matrix.scalar_multiplication(1 / sum_exp)
        return softmax_matrix
    def create_matrix(self, data):
        return Matrix(data)
    def apply_activation_derivative(self, matrix, function):
        return matrix.apply_function(function.derivative)
    def subtract(self, a, b):
        return a.subtract(b)
    def loss(self, prediction, target, gradient_scale=1.0):
        if (
            isinstance(gradient_scale, bool)
            or not isinstance(gradient_scale, (int, float))
            or not math.isfinite(gradient_scale)
            or gradient_scale <= 0.0
        ):
            raise ValueError("gradient_scale must be a finite positive number.")
        # Network outputs are already softmax probabilities.
        # Supports either a target class index (int) or a one-hot/soft target matrix.
        prob_data = prediction.data

        if hasattr(target, 'data'):
            target_flat = [value for row in target.data for value in row]
            loss = 0.0
            grad_flat = []
            for p, t in zip([value for row in prob_data for value in row], target_flat):
                loss -= t * math.log(p + 1e-9)
                grad_flat.append((p - t) * gradient_scale)
            grad_matrix = [
                grad_flat[i * len(prob_data[0]): (i + 1) * len(prob_data[0])]
                for i in range(len(prob_data))
            ]
            return loss, Matrix(grad_matrix)

        target_index = int(target)
        flat_probs = [value for row in prob_data for value in row]
        loss = -math.log(flat_probs[target_index] + 1e-9)

        grad_flat = [0.0 for _ in flat_probs]
        grad_flat[target_index] = 1.0
        for i, (p, g) in enumerate(zip(flat_probs, grad_flat)):
            grad_flat[i] = (p - g) * gradient_scale

        grad_matrix = [
            grad_flat[i * len(prob_data[0]): (i + 1) * len(prob_data[0])]
            for i in range(len(prob_data))
        ]
        return loss, Matrix(grad_matrix)
    def embedding_lookup(self, embedding_matrix, token_ids):
        return Matrix([embedding_matrix.data[token_id] for token_id in token_ids])
    def embedding_backward(self, token_ids, output_gradient, vocab_size):
        embedding_gradient = [[0.0] * len(output_gradient.data[0]) for _ in range(vocab_size)]
        for i, token_id in enumerate(token_ids):
            for j in range(len(output_gradient.data[0])):
                embedding_gradient[token_id][j] += output_gradient.data[i][j]
        return Matrix(embedding_gradient)
    def randn(self, rows, cols):
        data = [[random.gauss(0, 1) for _ in range(cols)] for _ in range(rows)]
        return Matrix(data)
    def zeros(self, rows, cols):
        data = [[0.0 for _ in range(cols)] for _ in range(rows)]
        return Matrix(data)
    def zero_matrices(self, matrix):
        rows = matrix.rows
        cols = matrix.columns
        data = [[0.0 for _ in range(cols)] for _ in range(rows)]
        return Matrix(data)
    def clip(self, matrix, minimum, maximum):
        return Matrix([
            [
                max(minimum, min(value, maximum))
                for value in row
            ]
            for row in matrix.data
        ])