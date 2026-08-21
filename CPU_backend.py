class CPUBackend:

    def matmul(self, a, b):
        return a.multiply(b)

    def add(self, a, b):
        return a.add(b)

    def apply_activation(self, matrix, function):
        return matrix.apply_function(function)