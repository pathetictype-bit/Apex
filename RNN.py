from acitvation_functions import tanh


BPTT_GRADIENT_LIMIT = 5.0


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
    def forward(self, inputs, h_prev, compute_outputs=True):
        previous_hidden_state = []
        hidden_states = []
        outputs = []
        h = h_prev
        for x in inputs:
            previous_hidden_state.append(h)
            if hasattr(self.backend, "rnn_forward_step"):
                h = self.backend.rnn_forward_step(
                    self.Wxh,
                    self.Whh,
                    self.bh,
                    x,
                    h,
                )
            else:
                hidden_inputs = self.backend.add(
                    self.backend.matmul(self.Wxh, x),
                    self.backend.matmul(self.Whh, h),
                )
                hidden_inputs = self.backend.add(hidden_inputs, self.bh)
                h = self.backend.apply_activation(
                    hidden_inputs,
                    function=self.activation_function,
                )

            if compute_outputs:
                y = self.backend.add(
                    self.backend.matmul(self.Why, h),
                    self.by
                )
            hidden_states.append(h)
            if compute_outputs:
                outputs.append(y)
        return hidden_states, outputs, previous_hidden_state
    def backward(self, inputs, hidden_states, previous_hidden_state, doutputs):
        """Backpropagate through time and return one ``dL/dx_t`` per input.

        Each input gradient has shape ``(input_size, 1)``. When ``x_t`` is an
        embedding lookup row, the caller can accumulate these gradients by
        token ID and pass them to the shared embedding table's backward path.
        """
        dinputs = [None] * len(inputs)
        dWxh = self.backend.zeros(self.hidden_size, self.input_size)
        dWhh = self.backend.zeros(self.hidden_size, self.hidden_size)
        dbh = self.backend.zeros(self.hidden_size, 1)
        dh_next = self.backend.zeros(self.hidden_size, 1)

        for t in reversed(range(len(inputs))):
            dh = doutputs[t]
            if dh is None:
                dh = self.backend.zeros(self.hidden_size, 1)

            dh = self.backend.clip(
                self.backend.add(dh, dh_next),
                -BPTT_GRADIENT_LIMIT,
                BPTT_GRADIENT_LIMIT,
            )
            dh_raw = self.backend.apply_activation_derivative(hidden_states[t], function=self.activation_function)
            dh_raw = self.backend.element_wise_multiply(dh_raw, dh)
            dinputs[t] = self.backend.matmul(self.Wxh.transpose(), dh_raw)
            dbh = self.backend.add(dbh, dh_raw)
            dWxh = self.backend.add(dWxh, self.backend.matmul(dh_raw, inputs[t].transpose()))
            dWhh = self.backend.add(dWhh, self.backend.matmul(dh_raw, previous_hidden_state[t].transpose()))
            dh_next = self.backend.clip(
                self.backend.matmul(self.Whh.transpose(), dh_raw),
                -BPTT_GRADIENT_LIMIT,
                BPTT_GRADIENT_LIMIT,
            )

        return dWxh, dWhh, dbh, dinputs
