import math
from acitvation_functions import tanh

class RNN:
    def __init__(self, input_size, hidden_size, output_size, backend):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.Wxh = backend.randn(hidden_size, input_size)

        self.Whh = backend.scalar_multiplication(
            backend.randn(hidden_size, hidden_size),
            0.5
        )

        self.Why = backend.randn(output_size, hidden_size)

        self.bh = backend.zeros(hidden_size, 1)
        self.by = backend.zeros(output_size, 1)

        self.backend = backend
        self.activation_function = tanh()
    def forward(self, inputs, h_prev):
        previous_hidden_state = []
        hidden_states = []
        outputs = []
        h = h_prev
        for x in inputs:
            previous_hidden_state.append(h)
            hidden_inputs = self.backend.add(
                    self.backend.matmul(self.Wxh, x),
                    self.backend.matmul(self.Whh, h),
                )
            
            hidden_inputs = self.backend.add(hidden_inputs, self.bh)
            
            h = self.backend.apply_activation(hidden_inputs, function=self.activation_function)
            y = self.backend.add(
                    self.backend.matmul(self.Why, h),
                    self.by
                )
            hidden_states.append(h)
            outputs.append(y)
        return hidden_states, outputs, previous_hidden_state
    def backward(self, inputs, hidden_states, previous_hidden_state, doutputs):
        dinputs = [None] * len(inputs)
        dWxh = self.backend.zeros(self.hidden_size, self.input_size)
        dWhh = self.backend.zeros(self.hidden_size, self.hidden_size)
        dbh = self.backend.zeros(self.hidden_size, 1)
        dh_next = self.backend.zeros(self.hidden_size, 1)

        for t in reversed(range(len(inputs))):
            dh = doutputs[t]
            if dh is None:
                dh = self.backend.zeros(self.hidden_size, 1)

            dh = self.backend.add(dh, dh_next)
            dh_raw = self.backend.apply_activation_derivative(hidden_states[t], function=self.activation_function)
            dh_raw = self.backend.element_wise_multiply(dh_raw, dh)
            dinputs[t] = self.backend.matmul(self.Wxh.transpose(), dh_raw)
            dbh = self.backend.add(dbh, dh_raw)
            dWxh = self.backend.add(dWxh, self.backend.matmul(dh_raw, inputs[t].transpose()))
            dWhh = self.backend.add(dWhh, self.backend.matmul(dh_raw, previous_hidden_state[t].transpose()))
            dh_next = self.backend.matmul(self.Whh.transpose(), dh_raw)

        return dWxh, dWhh, dbh, dinputs
