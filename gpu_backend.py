import numpy as np
from numba import cuda

@cuda.jit
def add_kernel(a, b, c):
    i = cuda.grid(1)

    if i < a.size:
        c[i] = a[i] + b[i]

@cuda.jit
def matmul_kernel(a, b, c):

    row, col = cuda.grid(2)

    if row < c.shape[0] and col < c.shape[1]:

        total = 0.0

        for k in range(a.shape[1]):
            total += a[row][k] * b[k][col]

        c[row][col] = total

class GPUBackend:

    def __init__(self):
        self.available = cuda.is_available()

        if not self.available:
            raise RuntimeError("CUDA is not available.")

    def to_gpu(self, data):
        return cuda.to_device(data)

    def to_cpu(self, data):
        return data.copy_to_host()

    def add(self, a, b):

        a_gpu = self.to_gpu(a)
        b_gpu = self.to_gpu(b)

        c_gpu = cuda.device_array_like(a_gpu)

        threads_per_block = 256
        blocks_per_grid = (a.size + threads_per_block - 1) // threads_per_block

        add_kernel[blocks_per_grid, threads_per_block](
            a_gpu,
            b_gpu,
            c_gpu
        )

        return self.to_cpu(c_gpu)
    def matmul(self, a, b):

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

        return self.to_cpu(c_gpu)
    
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
], dtype=np.float32)

B = np.array([
    [7, 8],
    [9, 10],
    [11, 12]
], dtype=np.float32)

backend = GPUBackend()

C = backend.matmul(A, B)

print(C)