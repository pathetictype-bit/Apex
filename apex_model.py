class Apex_model:
    def __init__(self, rnn, network, embedding):
        self.rnn = rnn
        self.network = network
        self.backend = rnn.backend
        self.embedding = embedding
        self._last_rnn_inputs = []

    def _as_rnn_inputs(self, embeddings):
        rnn_inputs = []
        rows = self._matrix_rows(embeddings)
        for row in rows:
            rnn_inputs.append(self.backend.create_matrix([[value] for value in row]))
        return rnn_inputs

    @staticmethod
    def _matrix_rows(matrix):
        rows = matrix.to_cpu() if hasattr(matrix, "to_cpu") else matrix.data
        return rows.tolist() if hasattr(rows, "tolist") else rows

    def forward(self, inputs):
        h_prev = self.backend.zeros(self.rnn.hidden_size, 1)
        embeddings = self.embedding.forward(inputs)
        self._last_rnn_inputs = self._as_rnn_inputs(embeddings)
        hidden_states, _, previous_hidden_state = self.rnn.forward(
            self._last_rnn_inputs,
            h_prev,
            compute_outputs=False,
        )
        network_outputs = []
        network_caches = []
        for hidden in hidden_states:
            network_output, cache = self.network.forward(hidden)
            network_outputs.append(network_output)
            network_caches.append(cache)
        self._last_network_caches = network_caches
        return hidden_states, previous_hidden_state, network_outputs, embeddings

    def predict_next_token(self, token_ids, h_prev=None):
        if not token_ids:
            return 0

        if h_prev is None:
            h_prev = self.backend.zeros(self.rnn.hidden_size, 1)

        embeddings = self.embedding.forward(token_ids)
        rnn_inputs = self._as_rnn_inputs(embeddings)
        hidden_states, _, _ = self.rnn.forward(
            rnn_inputs,
            h_prev,
            compute_outputs=False,
        )
        logits, _ = self.network.forward(hidden_states[-1])

        logits_rows = self._matrix_rows(logits)
        if logits_rows:
            flat_logits = [value for row in logits_rows for value in row]
            return max(range(len(flat_logits)), key=lambda idx: flat_logits[idx])
        return 0

    def generate(self, prompt, tokenizer, max_length=20, end_token_id=None):
        if isinstance(prompt, str):
            prompt_ids = tokenizer.encode(prompt)
        else:
            prompt_ids = list(prompt)

        if not prompt_ids:
            prompt_ids = [0]

        generated_ids = []
        h_prev = self.backend.zeros(self.rnn.hidden_size, 1)

        for token_id in prompt_ids:
            input_vector = self.embedding.forward([token_id])
            rnn_inputs = self._as_rnn_inputs(input_vector)
            hidden_states, _, _ = self.rnn.forward(
                rnn_inputs,
                h_prev,
                compute_outputs=False,
            )
            h_prev = hidden_states[-1]

        for _ in range(max_length):
            logits, _ = self.network.forward(h_prev)
            flat_logits = [value for row in self._matrix_rows(logits) for value in row]
            next_token = max(range(len(flat_logits)), key=lambda idx: flat_logits[idx])
            if next_token == end_token_id:
                break
            generated_ids.append(next_token)

            input_vector = self.embedding.forward([next_token])
            rnn_inputs = self._as_rnn_inputs(input_vector)
            hidden_states, _, _ = self.rnn.forward(
                rnn_inputs,
                h_prev,
                compute_outputs=False,
            )
            h_prev = hidden_states[-1]

        return tokenizer.decode(generated_ids)

    def backward(self, inputs, hidden_states, network_outputs, previous_hidden_state, doutputs):
        dhidden_states = []
        network_bias_gradients = None
        network_weight_gradients = None

        for doutput, cache in zip(doutputs, self._last_network_caches):
            dnetwork_input, dweight, dbias = self.network.backward(doutput, cache)
            dhidden_states.append(dnetwork_input)

            if network_weight_gradients is None:
                network_weight_gradients = [None] * len(dweight)
                network_bias_gradients = [None] * len(dbias)

            for i in range(len(dweight)):
                if network_weight_gradients[i] is None:
                    network_weight_gradients[i] = dweight[i]
                    network_bias_gradients[i] = dbias[i]
                else:
                    network_weight_gradients[i] = self.backend.add(network_weight_gradients[i], dweight[i])
                    network_bias_gradients[i] = self.backend.add(network_bias_gradients[i], dbias[i])

        dWxh, dWhh, dbh, dinputs = self.rnn.backward(
            self._last_rnn_inputs,
            hidden_states,
            previous_hidden_state,
            dhidden_states,
        )

        dinput_matrix = []
        for grad in dinputs:
            if grad is None:
                continue
            grad_rows = self._matrix_rows(grad)
            dinput_matrix.append([row[0] for row in grad_rows])

        if dinput_matrix:
            dinput_matrix = self.backend.create_matrix(dinput_matrix)
            embedding_gradients = self.embedding.backward(inputs, dinput_matrix)
        else:
            embedding_gradients = self.backend.create_matrix([
                [0.0 for _ in range(self.embedding.embedding_size)]
                for _ in range(self.embedding.vocab_size)
            ])

        return dWxh, dWhh, dbh, network_weight_gradients, network_bias_gradients, embedding_gradients

    def update_parameters(self, learning_rate, dWxh, dWhh, dbh, network_weight_gradients, network_bias_gradients, embedding_gradients):
        gradient_minimum = -5.0
        gradient_maximum = 5.0
        dWxh = self.backend.clip(dWxh, -5.0, 5.0)
        dWhh = self.backend.clip(dWhh, -5.0, 5.0)
        dbh = self.backend.clip(dbh, -5.0, 5.0)

        self.rnn.Wxh = self.backend.subtract(
            self.rnn.Wxh,
            self.backend.scalar_multiplication(dWxh, learning_rate)
        )

        self.rnn.Whh = self.backend.subtract(
            self.rnn.Whh,
            self.backend.scalar_multiplication(dWhh, learning_rate)
        )

        self.rnn.bh = self.backend.subtract(
            self.rnn.bh,
            self.backend.scalar_multiplication(dbh, learning_rate)
        )

        if network_weight_gradients is not None and network_bias_gradients is not None:
            clipped_weight_gradients = [
                self.backend.clip(gradient, gradient_minimum, gradient_maximum)
                for gradient in network_weight_gradients
            ]
            clipped_bias_gradients = [
                self.backend.clip(gradient, gradient_minimum, gradient_maximum)
                for gradient in network_bias_gradients
            ]
            self.network.update_parameters(clipped_weight_gradients, clipped_bias_gradients, learning_rate)

        if (
            embedding_gradients is not None
            and getattr(self.embedding, "trainable", True)
            and hasattr(self.embedding, "embedding_matrix")
        ):
            embedding_gradients = self.backend.clip(
                embedding_gradients,
                gradient_minimum,
                gradient_maximum
            )
            self.embedding.embedding_matrix = self.backend.subtract(
                self.embedding.embedding_matrix,
                self.backend.scalar_multiplication(embedding_gradients, learning_rate)
            )
