import math
import numpy as np
from numba import cuda
from Matrix import Matrix
from gpu_matrix import GPU_matrix

@cuda.jit
def add_kernel(a, b, c):
    row, col = cuda.grid(2)

    if row < c.shape[0] and col < c.shape[1]:
        c[row][col] = a[row][col] + b[row][col]

@cuda.jit
def subtract_kernel(a, b, c):
    row, col = cuda.grid(2)

    if row < c.shape[0] and col < c.shape[1]:
        c[row][col] = a[row][col] - b[row][col]

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

@cuda.jit
def activation_kernel(matrix, activation_kind, result):
    row, col = cuda.grid(2)

    if row < result.shape[0] and col < result.shape[1]:
        value = matrix[row][col]

        if activation_kind == 1:
            result[row][col] = max(0.0, value)
        elif activation_kind == 2:
            result[row][col] = max(0.01 * value, value)
        elif activation_kind == 3:
            result[row][col] = math.tanh(value)
        elif activation_kind == 4:
            result[row][col] = math.exp(value) - 1.0 if value < 0.0 else value
        elif activation_kind == 5:
            result[row][col] = 1.0 if value >= 0.0 else 0.0
        else:
            result[row][col] = value
@cuda.jit
def activation_derivative_kernel(matrix, activation_kind, result):
    row, col = cuda.grid(2)

    if row < result.shape[0] and col < result.shape[1]:
        value = matrix[row][col]

        if activation_kind == 1:
            result[row][col] = 1.0 if value > 0.0 else 0.0
        elif activation_kind == 2:
            result[row][col] = 1.0 if value > 0.0 else 0.01
        elif activation_kind == 3:
            result[row][col] = 1.0 - math.tanh(value) ** 2
        elif activation_kind == 4:
            result[row][col] = math.exp(value) if value < 0.0 else 1.0
        elif activation_kind == 5:
            result[row][col] = 0.0
        else:
            result[row][col] = 1.0

@cuda.jit
def softmax_max_kernel(matrix, maximum):
    thread_id = cuda.threadIdx.x
    stride = cuda.blockDim.x
    shared_values = cuda.shared.array(1024, dtype=np.float32)
    element_count = matrix.shape[0] * matrix.shape[1]

    local_maximum = -math.inf
    for index in range(thread_id, element_count, stride):
        row = index // matrix.shape[1]
        col = index % matrix.shape[1]
        value = matrix[row][col]
        if value > local_maximum:
            local_maximum = value

    shared_values[thread_id] = local_maximum
    cuda.syncthreads()

    offset = stride // 2
    while offset > 0:
        if thread_id < offset and shared_values[thread_id + offset] > shared_values[thread_id]:
            shared_values[thread_id] = shared_values[thread_id + offset]
        cuda.syncthreads()
        offset //= 2

    if thread_id == 0:
        maximum[0] = shared_values[0]

@cuda.jit
def softmax_sum_kernel(matrix, maximum, total):
    thread_id = cuda.threadIdx.x
    stride = cuda.blockDim.x
    shared_values = cuda.shared.array(1024, dtype=np.float32)
    element_count = matrix.shape[0] * matrix.shape[1]
    matrix_maximum = maximum[0]

    local_total = np.float32(0.0)
    for index in range(thread_id, element_count, stride):
        row = index // matrix.shape[1]
        col = index % matrix.shape[1]
        local_total += math.exp(matrix[row][col] - matrix_maximum)

    shared_values[thread_id] = local_total
    cuda.syncthreads()

    offset = stride // 2
    while offset > 0:
        if thread_id < offset:
            shared_values[thread_id] += shared_values[thread_id + offset]
        cuda.syncthreads()
        offset //= 2

    if thread_id == 0:
        total[0] = shared_values[0]

@cuda.jit
def softmax_normalize_kernel(matrix, maximum, total, result):
    row, col = cuda.grid(2)

    if row < result.shape[0] and col < result.shape[1]:
        result[row][col] = math.exp(matrix[row][col] - maximum[0]) / total[0]

@cuda.jit
def loss_kernel(prediction, target, loss):
    thread_id = cuda.threadIdx.x
    stride = cuda.blockDim.x
    shared_values = cuda.shared.array(1024, dtype=np.float32)
    element_count = prediction.shape[0] * prediction.shape[1]

    local_loss = np.float32(0.0)
    for index in range(thread_id, element_count, stride):
        row = index // prediction.shape[1]
        col = index % prediction.shape[1]
        p = prediction[row][col]
        t = target[row][col]
        local_loss += -t * math.log(p + 1e-9)

    shared_values[thread_id] = local_loss
    cuda.syncthreads()

    offset = stride // 2
    while offset > 0:
        if thread_id < offset:
            shared_values[thread_id] += shared_values[thread_id + offset]
        cuda.syncthreads()
        offset //= 2

    if thread_id == 0:
        loss[0] = shared_values[0]
@cuda.jit
def embedding_lookup_kernel(embedding_matrix, token_ids, result):
    row, col = cuda.grid(2)

    if row < result.shape[0] and col < result.shape[1]:
        token_id = token_ids[row]
        result[row][col] = embedding_matrix[token_id][col]

@cuda.jit
def embedding_backward_kernel(output_gradient, token_ids, embedding_gradient):
    """Accumulate gradients from output_gradient into embedding_gradient using token_ids.
    Uses atomic operations to handle multiple gradients going to same embedding.
    """
    row, col = cuda.grid(2)
    
    if row < output_gradient.shape[0] and col < output_gradient.shape[1]:
        token_id = token_ids[row]
        # Atomic add to handle gradients accumulating for same token_id
        cuda.atomic.add(embedding_gradient, (token_id, col), output_gradient[row, col])
@cuda.jit
def clip_kernel(matrix, minimum, maximum, result):
    row, col = cuda.grid(2)

    if row < result.shape[0] and col < result.shape[1]:
        value = matrix[row][col]
        result[row][col] = max(minimum, min(value, maximum))
        
class GPUBackend:

    def __init__(self):
        self.available = cuda.is_available()

        if not self.available:
            raise RuntimeError("CUDA is not available.")

    def _activation_kind(self, function):
        name = getattr(function, "__name__", function.__class__.__name__).lower()

        if "leaky" in name:
            return 2
        if "relu" in name:
            return 1
        if "tanh" in name:
            return 3
        if "elu" in name:
            return 4
        if "step" in name:
            return 5
        if "linear" in name:
            return 6
        raise ValueError(f"Unsupported GPU element-wise activation: {name}")

    def add(self, a, b):
        a_gpu = a.to_gpu()
        b_gpu = b.to_gpu()
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

        return GPU_matrix(c_gpu, from_gpu=True)

    def scalar_multiplication(self, matrix, scalar):
        matrix_gpu = matrix.to_gpu()
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

        return GPU_matrix(result_gpu, from_gpu=True)

    def matmul(self, a, b):
        a_gpu = a.to_gpu()
        b_gpu = b.to_gpu()

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

        return GPU_matrix(c_gpu, from_gpu=True)

    def apply_activation(self, matrix, function):
        matrix_gpu = matrix.to_gpu()
        result_gpu = cuda.device_array_like(matrix_gpu)

        threads_per_block = (16, 16)
        blocks_per_grid = (
            (matrix.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
            (matrix.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        )

        activation_kernel[blocks_per_grid, threads_per_block](
            matrix_gpu,
            self._activation_kind(function),
            result_gpu
        )

        return GPU_matrix(result_gpu, from_gpu=True)

    def element_wise_multiply(self, a, b):

        a_gpu = a.to_gpu()
        b_gpu = b.to_gpu()
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

        return GPU_matrix(result_gpu, from_gpu=True)

    def apply_function(self, matrix, function):
        return self.apply_activation(matrix, function)
    
    def apply_softmax(self, matrix):
        matrix_gpu = matrix.to_gpu()
        result_gpu = cuda.device_array_like(matrix_gpu)
        maximum_gpu = cuda.device_array(1, dtype=np.float32)
        total_gpu = cuda.device_array(1, dtype=np.float32)

        reduction_threads = 1024
        softmax_max_kernel[1, reduction_threads](matrix_gpu, maximum_gpu)
        softmax_sum_kernel[1, reduction_threads](matrix_gpu, maximum_gpu, total_gpu)

        threads_per_block = (16, 16)
        blocks_per_grid = (
            (matrix.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
            (matrix.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        )

        softmax_normalize_kernel[blocks_per_grid, threads_per_block](
            matrix_gpu,
            maximum_gpu,
            total_gpu,
            result_gpu
        )

        return GPU_matrix(result_gpu, from_gpu=True)
    def create_matrix(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        return GPU_matrix(data)
    def apply_activation_derivative(self, matrix, function):
        matrix_gpu = matrix.to_gpu()
        result_gpu = cuda.device_array_like(matrix_gpu)

        threads_per_block = (16, 16)
        blocks_per_grid = (
            (matrix.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
            (matrix.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        )

        activation_derivative_kernel[blocks_per_grid, threads_per_block](
            matrix_gpu,
            self._activation_kind(function),
            result_gpu
        )

        return GPU_matrix(result_gpu, from_gpu=True)
    def subtract(self, a, b):
        a_gpu = a.to_gpu()
        b_gpu = b.to_gpu()
        c_gpu = cuda.device_array_like(a_gpu)

        threads_per_block = (16, 16)
        blocks_per_grid = (
            (a.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
            (a.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        )

        subtract_kernel[blocks_per_grid, threads_per_block](
            a_gpu,
            b_gpu,
            c_gpu
        )

        return GPU_matrix(c_gpu, from_gpu=True)
    def loss(self, prediction, target):
        prediction_gpu = prediction.to_gpu()

        # Create one-hot target for the loss calculation only
        target_data = [[0.0] for _ in range(prediction.shape[0])]
        target_data[target][0] = 1.0

        target_matrix = GPU_matrix(target_data)
        target_gpu = target_matrix.to_gpu()

        loss_gpu = cuda.device_array(1, dtype=np.float32)

        reduction_threads = 1024
        loss_kernel[1, reduction_threads](
            prediction_gpu,
            target_gpu,
            loss_gpu
        )

        loss_value = loss_gpu.copy_to_host()[0]

        # Softmax + cross-entropy gradient
        doutput_data = [[0.0] for _ in range(prediction.shape[0])]
        prediction_cpu = prediction.to_cpu()

        for i in range(prediction.shape[0]):
            doutput_data[i][0] = prediction_cpu[i][0]

        doutput_data[target][0] -= 1.0

        doutput = GPU_matrix(doutput_data)

        return loss_value, doutput
    def embedding_lookup(self, embedding_matrix, token_ids):
        embedding_matrix_gpu = embedding_matrix.to_gpu()
        token_ids_gpu = cuda.to_device(np.array(token_ids, dtype=np.int32))
        result_gpu = cuda.device_array((len(token_ids), embedding_matrix.shape[1]), dtype=np.float32)

        threads_per_block = (16, 16)
        blocks_per_grid = (
            (len(token_ids) + threads_per_block[0] - 1) // threads_per_block[0],
            (embedding_matrix.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        )

        embedding_lookup_kernel[blocks_per_grid, threads_per_block](
            embedding_matrix_gpu,
            token_ids_gpu,
            result_gpu
        )

        return GPU_matrix(result_gpu, from_gpu=True)
    def embedding_backward(self, token_ids, output_gradient, vocab_size):
        """Accumulate gradients for embeddings.
        
        Args:
            token_ids: indices of tokens in the forward pass
            output_gradient: gradients w.r.t. embedding outputs (shape: [num_tokens, embedding_size])
            vocab_size: total vocabulary size for gradient matrix
        
        Returns:
            GPU_matrix of shape [vocab_size, embedding_size] with accumulated gradients
        """
        output_gradient_gpu = output_gradient.to_gpu()
        token_ids_gpu = cuda.to_device(np.array(token_ids, dtype=np.int32))
        embedding_size = output_gradient_gpu.shape[1]
        
        # Initialize gradient matrix to zeros
        embedding_gradient_gpu = cuda.device_array(
            (vocab_size, embedding_size),
            dtype=np.float32
        )
        embedding_gradient_gpu.copy_to_device(
            np.zeros((vocab_size, embedding_size), dtype=np.float32)
        )
        
        threads_per_block = (16, 16)
        blocks_per_grid = (
            (len(token_ids) + threads_per_block[0] - 1) // threads_per_block[0],
            (embedding_size + threads_per_block[1] - 1) // threads_per_block[1]
        )
        
        embedding_backward_kernel[blocks_per_grid, threads_per_block](
            output_gradient_gpu,
            token_ids_gpu,
            embedding_gradient_gpu
        )
        
        return GPU_matrix(embedding_gradient_gpu, from_gpu=True)
    def randn(self, rows, cols):
        data = np.random.randn(rows, cols).astype(np.float32)
        return GPU_matrix(data)
    def zeros(self, rows, cols):
        data = np.zeros((rows, cols), dtype=np.float32)
        return GPU_matrix(data)
    def zero_matrices(self, matrix):
        rows = matrix.shape[0]
        cols = matrix.shape[1]
        data = np.zeros((rows, cols), dtype=np.float32)
        return GPU_matrix(data)
    def clip(self, matrix, minimum, maximum):
        matrix_gpu = matrix.to_gpu()
        result_gpu = cuda.device_array_like(matrix_gpu)

        threads_per_block = (16, 16)
        blocks_per_grid = (
            (matrix.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
            (matrix.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
        )

        clip_kernel[blocks_per_grid, threads_per_block](
            matrix_gpu,
            np.float32(minimum),
            np.float32(maximum),
            result_gpu
        )

        return GPU_matrix(result_gpu, from_gpu=True)