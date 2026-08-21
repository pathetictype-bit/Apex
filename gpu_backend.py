import numpy as np
from numba import cuda
from Matrix import Matrix

@cuda.jit
def add_kernel(a, b, c):
    row, col = cuda.grid(2)

    if row < c.shape[0] and col < c.shape[1]:
        c[row][col] = a[row][col] + b[row][col]

@cuda.jit
def matmul_kernel(a, b, c):

    row, col = cuda.grid(2)

    if row < c.shape[0] and col < c.shape[1]:

        total = 0.0

        for k in range(a.shape[1]):
            total += a[row][k] * b[k][col]

        c[row][col] = total

@cuda.jit
def element_wise_multiply_kernel(a, b, c):
    row, col = cuda.grid(2)

    if row < c.shape[0] and col < c.shape[1]:
        c[row][col] = a[row][col] * b[row][col]

@cuda.jit
def scalar_multiply_kernel(matrix, scalar, result):
    row, col = cuda.grid(2)

    if row < result.shape[0] and col < result.shape[1]:
        result[row][col] = matrix[row][col] * scalar

class GPUBackend:

    def __init__(self):
        self.available = cuda.is_available()

        if not self.available:
            raise RuntimeError("CUDA is not available.")

    def to_gpu(self, data):
        return cuda.to_device(data)

    def to_cpu(self, data):
        return data.copy_to_host()

    def _to_array(self, matrix):
        if isinstance(matrix, Matrix):
            matrix = matrix.data
        return np.asarray(matrix, dtype=np.float32)

    def _to_matrix(self, array):
        return Matrix(array.tolist())

    def add(self, a, b):
        a = self._to_array(a)
        b = self._to_array(b)

        a_gpu = self.to_gpu(a)
        b_gpu = self.to_gpu(b)

        c_gpu = cuda.device_array_like(a_gpu)

        threads_per_block = (16, 16)
        blocks_per_grid = (
            (a.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
            (a.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        )

        add_kernel[blocks_per_grid, threads_per_block](
            a_gpu,
            b_gpu,
            c_gpu
        )

        return self._to_matrix(self.to_cpu(c_gpu))

    def scalar_multiplication(self, matrix, scalar):
        matrix = self._to_array(matrix)
        matrix_gpu = self.to_gpu(matrix)
        result_gpu = cuda.device_array_like(matrix_gpu)

        threads_per_block = (16, 16)
        blocks_per_grid = (
            (matrix.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
            (matrix.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        )

        scalar_multiply_kernel[blocks_per_grid, threads_per_block](
            matrix_gpu,
            np.float32(scalar),
            result_gpu
        )

        return self._to_matrix(self.to_cpu(result_gpu))

    def matmul(self, a, b):
        a = self._to_array(a)
        b = self._to_array(b)

        a_gpu = self.to_gpu(a)
        b_gpu = self.to_gpu(b)

        c_gpu = cuda.device_array(
            (a.shape[0], b.shape[1]),
            dtype=np.float32
        )

        threads_per_block = (16, 16)

        blocks_per_grid = (
            (a.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
            (b.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        )

        matmul_kernel[blocks_per_grid, threads_per_block](
            a_gpu,
            b_gpu,
            c_gpu
        )

        return self._to_matrix(self.to_cpu(c_gpu))

    def apply_activation(self, matrix, function):
        matrix = self._to_array(matrix)
        vectorized_function = np.vectorize(function)
        return self._to_matrix(vectorized_function(matrix))

    def element_wise_multiply(self, a, b):
        a = self._to_array(a)
        b = self._to_array(b)
        a_gpu = self.to_gpu(a)
        b_gpu = self.to_gpu(b)
        result_gpu = cuda.device_array_like(a_gpu)

        threads_per_block = (16, 16)
        blocks_per_grid = (
            (a.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
            (a.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        )

        element_wise_multiply_kernel[blocks_per_grid, threads_per_block](
            a_gpu,
            b_gpu,
            result_gpu
        )

        return self._to_matrix(self.to_cpu(result_gpu))

    def apply_function(self, matrix, function):
        return self.apply_activation(matrix, function)