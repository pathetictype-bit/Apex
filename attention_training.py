"""Trainable self-attention kept outside the PNSN model classes."""

import math

from attention import Attention
from CPU_backend import CPUBackend


def _rows(matrix):
    data = matrix.to_cpu() if hasattr(matrix, "to_cpu") else matrix.data
    return data.tolist() if hasattr(data, "tolist") else data


def _transpose(matrix):
    return [list(column) for column in zip(*matrix)]


def _matmul(left, right):
    return [
        [
            math.fsum(left[row][index] * right[index][column]
                      for index in range(len(right)))
            for column in range(len(right[0]))
        ]
        for row in range(len(left))
    ]


def _add(left, right):
    return [
        [a + b for a, b in zip(left_row, right_row)]
        for left_row, right_row in zip(left, right)
    ]


class TrainableAttentionModel:
    """Wrap an existing APEX model with causal learned Q/K/V projections.

    The wrapped RNN, network, and embedding objects are unchanged. Attention
    parameters are trained separately and must be checkpointed separately.
    """

    def __init__(self, model, state=None):
        self.model = model
        self.rnn = model.rnn
        self.network = model.network
        self.embedding = model.embedding
        self.backend = model.backend
        self.dimension = self.embedding.embedding_size
        scale = 1.0 / math.sqrt(self.dimension)
        random_weights = _rows(
            self.backend.randn(self.dimension, self.dimension)
        )
        self.Wq = [[value * scale for value in row] for row in random_weights]
        random_weights = _rows(
            self.backend.randn(self.dimension, self.dimension)
        )
        self.Wk = [[value * scale for value in row] for row in random_weights]
        self.Wv = [
            [float(row == column) for column in range(self.dimension)]
            for row in range(self.dimension)
        ]
        self.attention = Attention(CPUBackend())
        self._forward_cache = None
        self._attention_gradients = None
        if state is not None:
            self.load_state(state)

    def state_dict(self):
        return {
            "dimension": self.dimension,
            "Wq": [row[:] for row in self.Wq],
            "Wk": [row[:] for row in self.Wk],
            "Wv": [row[:] for row in self.Wv],
        }

    def load_state(self, state):
        if not isinstance(state, dict) or state.get("dimension") != self.dimension:
            raise ValueError("Attention checkpoint dimension is incompatible.")
        matrices = {}
        for name in ("Wq", "Wk", "Wv"):
            matrix = state.get(name)
            if (
                not isinstance(matrix, list)
                or len(matrix) != self.dimension
                or any(
                    not isinstance(row, list) or len(row) != self.dimension
                    for row in matrix
                )
            ):
                raise ValueError(f"Attention checkpoint matrix {name} is invalid.")
            converted = []
            for row in matrix:
                values = []
                for value in row:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise ValueError(
                            f"Attention checkpoint matrix {name} is invalid."
                        )
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValueError(
                            f"Attention checkpoint matrix {name} is non-finite."
                        )
                    values.append(value)
                converted.append(values)
            matrices[name] = converted
        self.Wq, self.Wk, self.Wv = (
            matrices["Wq"], matrices["Wk"], matrices["Wv"]
        )

    def _project(self, inputs):
        query = _matmul(inputs, self.Wq)
        key = _matmul(inputs, self.Wk)
        value = _matmul(inputs, self.Wv)
        mask = [
            [key_index <= query_index for key_index in range(len(inputs))]
            for query_index in range(len(inputs))
        ]
        context, weights = self.attention.forward(
            query, key, value, mask=mask
        )
        context_rows = _rows(context)
        attended = _add(inputs, context_rows)
        return attended, {
            "inputs": inputs,
            "query": query,
            "key": key,
            "value": value,
            "weights": _rows(weights),
        }

    def forward(self, token_ids):
        embedding_matrix = self.embedding.forward(token_ids)
        inputs = _rows(embedding_matrix)
        attended, attention_cache = self._project(inputs)
        rnn_inputs = [
            self.backend.create_matrix([[value] for value in row])
            for row in attended
        ]
        h_prev = self.backend.zeros(self.rnn.hidden_size, 1)
        hidden_states, _, previous_hidden_state = self.rnn.forward(
            rnn_inputs,
            h_prev,
            compute_outputs=False,
        )
        network_outputs = []
        network_caches = []
        for hidden in hidden_states:
            output, cache = self.network.forward(hidden)
            network_outputs.append(output)
            network_caches.append(cache)
        self._forward_cache = {
            "rnn_inputs": rnn_inputs,
            "attention": attention_cache,
            "network": network_caches,
        }
        return hidden_states, previous_hidden_state, network_outputs, embedding_matrix

    def _attention_backward(self, output_gradient, cache):
        inputs = cache["inputs"]
        query = cache["query"]
        key = cache["key"]
        value = cache["value"]
        weights = cache["weights"]
        query_count = len(inputs)
        dimension = self.dimension
        scale = 1.0 / math.sqrt(dimension)

        dvalue = _matmul(_transpose(weights), output_gradient)
        dweights = _matmul(output_gradient, _transpose(value))
        dscores = [[0.0] * query_count for _ in range(query_count)]
        for query_index in range(query_count):
            weighted_sum = math.fsum(
                weights[query_index][key_index]
                * dweights[query_index][key_index]
                for key_index in range(query_count)
            )
            for key_index in range(query_index + 1):
                probability = weights[query_index][key_index]
                dscores[query_index][key_index] = (
                    probability
                    * (dweights[query_index][key_index] - weighted_sum)
                )

        dquery = _matmul(dscores, key)
        dquery = [[value * scale for value in row] for row in dquery]
        dkey = _matmul(_transpose(dscores), query)
        dkey = [[value * scale for value in row] for row in dkey]

        input_transpose = _transpose(inputs)
        gradients = (
            _matmul(input_transpose, dquery),
            _matmul(input_transpose, dkey),
            _matmul(input_transpose, dvalue),
        )
        return gradients

    def backward(
        self,
        token_ids,
        hidden_states,
        network_outputs,
        previous_hidden_state,
        output_gradients,
    ):
        _ = token_ids, network_outputs
        if self._forward_cache is None:
            raise RuntimeError("Attention model backward called before forward.")
        network_weight_gradients = None
        network_bias_gradients = None
        hidden_gradients = []
        for output_gradient, cache in zip(
            output_gradients, self._forward_cache["network"]
        ):
            dnetwork_input, dweights, dbiases = self.network.backward(
                output_gradient, cache
            )
            hidden_gradients.append(dnetwork_input)
            if network_weight_gradients is None:
                network_weight_gradients = dweights
                network_bias_gradients = dbiases
            else:
                network_weight_gradients = [
                    self.backend.add(total, gradient)
                    for total, gradient in zip(
                        network_weight_gradients, dweights
                    )
                ]
                network_bias_gradients = [
                    self.backend.add(total, gradient)
                    for total, gradient in zip(
                        network_bias_gradients, dbiases
                    )
                ]

        dWxh, dWhh, dbh, input_gradients = self.rnn.backward(
            self._forward_cache["rnn_inputs"],
            hidden_states,
            previous_hidden_state,
            hidden_gradients,
        )
        dcontext = [
            [row[0] for row in _rows(gradient)]
            for gradient in input_gradients
        ]
        self._attention_gradients = self._attention_backward(
            dcontext, self._forward_cache["attention"]
        )
        zero_embedding_gradients = self.backend.zeros(
            self.embedding.vocab_size, self.dimension
        )
        return (
            dWxh,
            dWhh,
            dbh,
            network_weight_gradients,
            network_bias_gradients,
            zero_embedding_gradients,
        )

    def update_parameters(
        self,
        learning_rate,
        dWxh,
        dWhh,
        dbh,
        network_weight_gradients,
        network_bias_gradients,
        embedding_gradients,
    ):
        self.model.update_parameters(
            learning_rate,
            dWxh,
            dWhh,
            dbh,
            network_weight_gradients,
            network_bias_gradients,
            embedding_gradients,
        )
        if self._attention_gradients is None:
            raise RuntimeError("Attention gradients are missing.")
        for name, gradient in zip(
            ("Wq", "Wk", "Wv"), self._attention_gradients
        ):
            parameters = getattr(self, name)
            for row_index, row in enumerate(gradient):
                for column_index, value in enumerate(row):
                    clipped = max(-5.0, min(5.0, value))
                    parameters[row_index][column_index] -= (
                        learning_rate * clipped
                    )

    def _attend_incremental(self, token_ids):
        embeddings = _rows(self.embedding.forward(token_ids))
        queries = _matmul(embeddings, self.Wq)
        keys = _matmul(embeddings, self.Wk)
        values = _matmul(embeddings, self.Wv)
        latest_query = [queries[-1]]
        context, _ = self.attention.forward(
            latest_query,
            keys,
            values,
            mask=[True] * len(keys),
        )
        attended = _add(_rows(context), [embeddings[-1]])[0]
        return self.backend.create_matrix(
            [[value] for value in attended]
        )

    def generate(self, prompt, tokenizer, max_length=20, end_token_id=None):
        prompt_ids = tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
        if not prompt_ids:
            prompt_ids = [0]

        embeddings = _rows(self.embedding.forward(prompt_ids))
        attended, _ = self._project(embeddings)
        rnn_inputs = [
            self.backend.create_matrix([[value] for value in row])
            for row in attended
        ]
        hidden_states, _, _ = self.rnn.forward(
            rnn_inputs,
            self.backend.zeros(self.rnn.hidden_size, 1),
            compute_outputs=False,
        )
        hidden = hidden_states[-1]
        generated_ids = []
        consumed_ids = list(prompt_ids)

        for _ in range(max_length):
            logits, _ = self.network.forward(hidden)
            logits_rows = _rows(logits)
            flat_logits = [value for row in logits_rows for value in row]
            next_token = max(range(len(flat_logits)), key=flat_logits.__getitem__)
            if next_token == end_token_id:
                break
            generated_ids.append(next_token)
            consumed_ids.append(next_token)
            attended_token = self._attend_incremental(consumed_ids)
            next_hidden, _, _ = self.rnn.forward(
                [attended_token],
                hidden,
                compute_outputs=False,
            )
            hidden = next_hidden[-1]
        return tokenizer.decode(generated_ids)
