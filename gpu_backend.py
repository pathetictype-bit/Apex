import math
import numpy as np
from numba import cuda
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
def rnn_hidden_step_kernel(wxh, whh, bias, x, previous_hidden, output):
    hidden_index = cuda.grid(1)
    if hidden_index < output.shape[0]:
        total = bias[hidden_index][0]
        for input_index in range(wxh.shape[1]):
            total += wxh[hidden_index][input_index] * x[input_index][0]
        for previous_index in range(whh.shape[1]):
            total += (
                whh[hidden_index][previous_index]
                * previous_hidden[previous_index][0]
            )
        output[hidden_index][0] = math.tanh(total)


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
            # RNN.backward supplies the already activated hidden state, as
            # does tanh.derivative in the CPU backend.
            result[row][col] = 1.0 - value ** 2
        elif activation_kind == 4:
            result[row][col] = math.exp(value) if value < 0.0 else 1.0
        elif activation_kind == 5:
            result[row][col] = 0.0
        else:
            result[row][col] = 1.0

@cuda.jit
def softmax_kernel(matrix, result):
    thread_id = cuda.threadIdx.x
    stride = cuda.blockDim.x
    shared_values = cuda.shared.array(1024, dtype=np.float32)
    element_count = matrix.shape[0] * matrix.shape[1]

    if thread_id == 0:
        maximum = -math.inf
        for index in range(element_count):
            row = index // matrix.shape[1]
            col = index % matrix.shape[1]
            if matrix[row][col] > maximum:
                maximum = matrix[row][col]

        total = 0.0
        for index in range(element_count):
            row = index // matrix.shape[1]
            col = index % matrix.shape[1]
            total += math.exp(matrix[row][col] - maximum)
        shared_values[0] = maximum
        shared_values[1] = total
    cuda.syncthreads()

    for index in range(thread_id, element_count, stride):
        row = index // matrix.shape[1]
        col = index % matrix.shape[1]
        result[row][col] = (
            math.exp(matrix[row][col] - shared_values[0])
            / shared_values[1]
        )


@cuda.jit
def loss_and_gradient_kernel(
    prediction,
    target,
    gradient_scale,
    gradient,
    loss,
):
    index = cuda.grid(1)
    element_count = prediction.shape[0] * prediction.shape[1]

    if index == 0:
        target_row = target // prediction.shape[1]
        target_col = target % prediction.shape[1]
        probability = prediction[target_row][target_col]
        loss_value = -math.log(probability + 1e-9)
        for value_index in range(element_count):
            row = value_index // prediction.shape[1]
            col = value_index % prediction.shape[1]
            if not math.isfinite(prediction[row][col]):
                loss_value = math.nan
                break
        loss[0] = loss_value

    if index < element_count:
        row = index // prediction.shape[1]
        col = index % prediction.shape[1]
        gradient[row][col] = (
            prediction[row][col] - (1.0 if index == target else 0.0)
        ) * gradient_scale


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
    name = "GPU (Numba CUDA)"
    loss_validates_gradient = True

    @staticmethod
    def is_available():
        return cuda.is_available()

    def __init__(self):
        self.available = self.is_available()

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

    def rnn_forward_step(self, wxh, whh, bias, x, previous_hidden):
        wxh_gpu = wxh.to_gpu()
        whh_gpu = whh.to_gpu()
        bias_gpu = bias.to_gpu()
        x_gpu = x.to_gpu()
        previous_hidden_gpu = previous_hidden.to_gpu()
        output_gpu = cuda.device_array(
            (wxh.shape[0], 1),
            dtype=np.float32,
        )
        threads_per_block = 128
        blocks_per_grid = (
            wxh.shape[0] + threads_per_block - 1
        ) // threads_per_block
        rnn_hidden_step_kernel[blocks_per_grid, threads_per_block](
            wxh_gpu,
            whh_gpu,
            bias_gpu,
            x_gpu,
            previous_hidden_gpu,
            output_gpu,
        )
        return GPU_matrix(output_gpu, from_gpu=True)

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
        softmax_kernel[1, 512](matrix_gpu, result_gpu)

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
    def loss(self, prediction, target, gradient_scale=1.0):
        if (
            isinstance(target, bool)
            or not isinstance(target, int)
            or target < 0
            or target >= prediction.shape[0] * prediction.shape[1]
        ):
            raise ValueError("target class index is outside the prediction.")
        if (
            isinstance(gradient_scale, bool)
            or not isinstance(gradient_scale, (int, float))
            or not math.isfinite(gradient_scale)
            or gradient_scale <= 0.0
        ):
            raise ValueError("gradient_scale must be a finite positive number.")
        prediction_gpu = prediction.to_gpu()
        gradient_gpu = cuda.device_array_like(prediction_gpu)
        loss_gpu = cuda.device_array(1, dtype=np.float32)
        threads_per_block = 256
        element_count = prediction.shape[0] * prediction.shape[1]
        blocks_per_grid = (element_count + threads_per_block - 1) // threads_per_block
        loss_and_gradient_kernel[blocks_per_grid, threads_per_block](
            prediction_gpu,
            target,
            np.float32(gradient_scale),
            gradient_gpu,
            loss_gpu
        )
        loss_value = loss_gpu.copy_to_host()[0]
        if not math.isfinite(float(loss_value)):
            raise FloatingPointError(
                "GPU cross-entropy received a non-finite prediction."
            )
        return loss_value, GPU_matrix(gradient_gpu, from_gpu=True)
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
