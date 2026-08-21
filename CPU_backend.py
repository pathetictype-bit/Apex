from Matrix import Matrix
class CPUBackend:

    def matmul(self, a, b):
        return a.multiply(b)

    def add(self, a, b):
        return a.add(b)

    def apply_activation(self, matrix, function):
        return matrix.apply_function(function)
    
    def element_wise_multiply(self, a, b):
        result = []
        for i in range(a.rows):
            rows = []
            for j in range(a.columns):
                rows.append(a.data[i][j] * b.data[i][j])
            result.append(rows)
        return Matrix(result)

    def apply_function(self, matrix, function):
        return matrix.apply_function(function)