from numba import cuda
class GPU_matrix:
    
    def __init__(self, data):
        self.data = cuda.to_device(data)
        self.shape = data.shape

    def to_cpu(self):
        return self.data.copy_to_host()

    def to_gpu(self):
        return self.data
