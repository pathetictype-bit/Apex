import random
from acitvation_functions import tanh_activation, tanh_derivative

class rnn:
    def __init__(self, input_size, hidden_size):
        self.history = []
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_weights = [[random.uniform(-1.0, 1.0) for _ in  range(input_size)] for _ in range(hidden_size)]
        self.hidden_weights = [[random.uniform(-1.0, 1.0) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.biases = [random.uniform(-1.0, 1.0) for _ in range(hidden_size)]
        self.hidden_state = [0] * hidden_size

    def forward(self, inputs, store_history = True):
        new_hidden_state = []
        z = []
        for i in range(self.hidden_size):
            total = 0
            for j in range(self.input_size):
                total += inputs[j] * self.input_weights[i][j]

            for j in range(self.hidden_size):
                total += self.hidden_state[j] * self.hidden_weights[i][j]

            total += self.biases[i]
            z.append(total)
            new_hidden_state.append(tanh_activation(total))
        h_prev = self.hidden_state.copy()

        if store_history:
            self.history.append({"inputs" : inputs.copy(), "h_prev" : h_prev.copy(), "z" : z.copy(), "h" : new_hidden_state.copy()})

        self.hidden_state = new_hidden_state
        return self.hidden_state

    def backward(self, gradients):
        input_weight_gradients = [
            [0 for _ in range(self.input_size)]
            for _ in range(self.hidden_size)
        ]
        hidden_weight_gradients = [
            [0 for _ in range(self.hidden_size)]
            for _ in range(self.hidden_size)
        ]
        bias_gradients = [0 for _ in range(self.hidden_size)]

        dh_next = [0 for _ in range(self.hidden_size)]

        for time_step in reversed(range(len(self.history))):
            step = self.history[time_step]

            inputs = step["inputs"]
            h_prev = step["h_prev"]
            h = step["h"]

            dh = gradients[time_step]

            dh_total = [
                dh[i] + dh_next[i]
                for i in range(self.hidden_size)
            ]

            dL_dz = [
                dh_total[i] * (1 - h[i] ** 2)
                for i in range(self.hidden_size)
            ]

            for i in range(self.hidden_size):
                for j in range(self.input_size):
                    input_weight_gradients[i][j] += dL_dz[i] * inputs[j]

            for i in range(self.hidden_size):
                for j in range(self.hidden_size):
                    hidden_weight_gradients[i][j] += dL_dz[i] * h_prev[j]

            for i in range(self.hidden_size):
                bias_gradients[i] += dL_dz[i]


            new_dh_next = [0 for _ in range(self.hidden_size)]

            for i in range(self.hidden_size):
                for j in range(self.hidden_size):
                    new_dh_next[j] += self.hidden_weights[i][j] * dL_dz[i]

            dh_next = new_dh_next

        return {
            "input_weight_gradients": input_weight_gradients,
            "hidden_weight_gradients": hidden_weight_gradients,
            "bias_gradients": bias_gradients,
        }

    def update_parameter(self, gradient, learning_rate):
        input_weight_gradients = gradient["input_weight_gradients"]
        hidden_weight_gradients = gradient["hidden_weight_gradients"]
        bias_gradients = gradient["bias_gradients"]

        for i in range(self.hidden_size):
            for j in range(self.input_size):
                self.input_weights[i][j] -= learning_rate * input_weight_gradients[i][j]
        for i in range(self.hidden_size):
            for j in range(self.hidden_size):
                self.hidden_weights[i][j] -= learning_rate * hidden_weight_gradients[i][j]
        for i in range(self.hidden_size):
            self.biases[i] -= learning_rate * bias_gradients[i]

    def get_weights(self):
        return {"input_weights" : self.input_weights,  
                "hidden_weights" : self.hidden_weights,
                "biases" : self.biases}

    def set_weights(self, data):
        self.input_weights = data["input_weights"]
        self.hidden_weights = data["hidden_weights"]
        self.biases = data["biases"]
    
        