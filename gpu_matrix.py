from numba import cuda
@cuda.jit
def transpose_kernel(a, b):
        row, col = cuda.grid(2)
        if row < a.shape[0] and col < a.shape[1]:
            b[col, row] = a[row, col]
class GPU_matrix:
    
    def __init__(self, data, from_gpu=False):
        if from_gpu:
            self.data = data
        else:
            self.data = cuda.to_device(data)
        self.shape = self.data.shape

    def to_cpu(self):
        return self.data.copy_to_host()

    def to_gpu(self):
        return self.data

    def transpose(self):
        transposed_data = cuda.device_array((self.shape[1], self.shape[0]), dtype=self.data.dtype)
        threads_per_block = (16, 16)
        blocks_per_grid = (
            (self.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
            (self.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        )
        transpose_kernel[blocks_per_grid, threads_per_block](self.data, transposed_data)
        return GPU_matrix(transposed_data, from_gpu=True)