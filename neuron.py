from acitvation_functions import relu_activation, relu_derivative, softmax
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        self.last_inputs = inputs
        total = 0
        for input_value, weight in zip(inputs, self.weights):
            total += input_value * weight
        self.last_z = total + self.bias
        return self.last_z

    def backward(self, dL_dz):
        dL_dweights = [dL_dz * input_value for input_value in self.last_inputs]
        dL_db = dL_dz
        dL_dinputs = [dL_dz * weight for weight in self.weights]
        return dL_dweights, dL_db, dL_dinputs
    def update_parameters(self, weight_gradients, bias_gradient, learning_rate):
        for i in range(len(self.weights)):
            self.weights [i] -= learning_rate * weight_gradients[i]

        self.bias -= learning_rate * bias_gradient




