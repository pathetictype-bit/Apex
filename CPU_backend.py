from Matrix import Matrix
class CPUBackend:

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