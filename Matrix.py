class Matrix:
    def __init__(self, data):
        self.data = data
        self.rows = len(self.data)
        self.columns =  len(self.data[0])
    def add(self, other):
        result = []
        for i in range(self.rows):
            rows = []
            for j in range(self.columns):
                rows.append(self.data[i][j] + other.data[i][j])
            result.append(rows)
        return Matrix(result)
    def multiply(self, other):
        c = []
        for i in range(len(self.data)):
            rows_c = []
            for j in range(len(other.data[0])):   # every column of B
                row_A = self.data[i]
                column_B = [other.data[k][j] for k in range(len(other.data))]
                dot_product = sum(
                    row_A[k] * column_B[k]
                    for k in range(len(row_A))
                )
                rows_c.append(dot_product)
            c.append(rows_c)
        return Matrix(c)
    def subtract(self, other):
        result = []
        for i in range(self.rows):
            rows = []
            for j in range(self.columns):
                rows.append(self.data[i][j] - other.data[i][j])
            result.append(rows)
        return Matrix(result)
    def scalar_multiplication(self, scalar):
        result = []
        for i in range(self.rows):
            rows = []
            for j in range(self.columns):
                rows.append(self.data[i][j] * scalar)
            result.append(rows)
        return Matrix(result)
    def transpose(self):
        result = []
        for i in range(self.columns):
            row = []
            for j in range(self.rows):
                row.append(self.data[j][i])
            result.append(row)
        return Matrix(result)
    def apply_function(self, function):
        result = []
        for row in self.data:
            new_row = []
            for value in row:
                new_row.append(function(value))
            result.append(new_row)
        return Matrix(result)
    def __str__(self):
        rows = []

        for row in self.data:
            rows.append(str(row))

        return "\n".join(rows)

matrix_A = Matrix(
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
)
matrix_B = Matrix(
    [
        [2, 6],
        [4, 5],
        [3, 6]
    ]
)

result = Matrix.multiply(matrix_A, matrix_B)
print(result)
