from acitvation_functions import leaky_relu
from layer import Layer

class Network:
    def __init__(self, layers_sizes, backend):
        self.layers = []
        self.backend = backend
        for i in range(len(layers_sizes) - 1):
            input_size = layers_sizes[i]
            output_size = layers_sizes[i + 1]
            if i == len(layers_sizes) - 2:
                activation = backend.apply_softmax
            else:
                activation = leaky_relu()
            layer = Layer(input_size, output_size, activation, backend)
            self.layers.append(layer)
    def forward(self, inputs):
        output = inputs
        caches = []
        for layer in self.layers:
            output, cache = layer.forward(output)
            caches.append(cache)
        return output, caches
    def backward(self, output_gradient, caches):
        weight_gradients = [None] * len(self.layers)
        bias_gradients = [None] * len(self.layers)
        gradient = output_gradient
        for layer_index in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[layer_index]
            cache = caches[layer_index]
            input_gradient, weight_gradient, bias_gradient = layer.backward(gradient, cache)
            weight_gradients[layer_index] = weight_gradient
            bias_gradients[layer_index] = bias_gradient
            gradient = input_gradient
        return gradient, weight_gradients, bias_gradients

    def update_parameters(self, weight_gradients, bias_gradients, learning_rate):
        for layer, weight_gradient, bias_gradient in zip(self.layers, weight_gradients, bias_gradients):
            layer.update_parameters(weight_gradient, bias_gradient, learning_rate)

