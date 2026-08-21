import math
from layer import Layer
from acitvation_functions import softmax, leaky_relu_activation, leaky_relu_derivative
class Network:
    def __init__(self, layer_sizes):
        self.layers = []
        self.weight_gradients = None
        self.bias_gradients = None
        for i in range (len(layer_sizes)-1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i+1]
            if i == len(layer_sizes) -2:
                activation_derivative = None
                activation = softmax
            else:
                activation_derivative = leaky_relu_derivative
                activation = leaky_relu_activation
            self.layers.append(Layer(layer_size= output_size, input_size=input_size, activation = activation, activation_derivative = activation_derivative, is_output=(i == len(layer_sizes) - 2)))
    def forward(self, inputs):
        current_input = inputs
        
        for layer in self.layers:
            current_output = layer.forward(current_input)
            current_input = current_output
        return current_output
    
    def backward(self, gradient):
        current_weight_gradients = []
        current_bias_gradients = []

        for layer in reversed(self.layers):
            input_gradients, weight_gradients, bias_gradients = layer.backward(gradient)

            current_weight_gradients.append(weight_gradients)
            current_bias_gradients.append(bias_gradients)

            gradient = input_gradients

        current_weight_gradients.reverse()
        current_bias_gradients.reverse()

        if self.weight_gradients is None:
            self.weight_gradients = current_weight_gradients
            self.bias_gradients = current_bias_gradients

        else:
            for l in range(len(self.layers)):
                for n in range(len(self.layers[l].neurons)):
                    for w in range(len(self.layers[l].neurons[n].weights)):
                        self.weight_gradients[l][n][w] += current_weight_gradients[l][n][w]

                    self.bias_gradients[l][n] += current_bias_gradients[l][n]

        return gradient

    def update_parameters(self, learning_rate):
        for layer, weight_gradient, bias_gradient in zip(self.layers, self.weight_gradients, self.bias_gradients):
            layer.update_parameters(weight_gradient, bias_gradient, learning_rate)

    def get_weights(self):
        data = []

        for layer in self.layers:
            layer_data = []
            for neuron in layer.neurons:
                layer_data.append ({"weights" : neuron.weights,
                                    "bias" : neuron.bias})
            data.append({"neurons" : layer_data})

        return data

    def set_weights(self, data):
        for layer, layer_data in zip(self.layers, data):
            for neuron, neuron_data in zip(layer.neurons, layer_data["neurons"]):
                neuron.weights = neuron_data["weights"]
                neuron.bias = neuron_data["bias"]
    def reset_gradients(self):
        self.weight_gradients = None
        self.bias_gradients = None

