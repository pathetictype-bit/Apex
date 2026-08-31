class Apex_model:
    def __init__(self, rnn, network, embedding):
        self.rnn = rnn
        self.network = network
        self.backend = rnn.backend
        self.embedding = embedding
    def forward(self, inputs):
        h_prev = self.backend.zeros(self.rnn.hidden_size, 1)
        embedding = self.embedding.forward(inputs)
        hidden_states, outputs, previous_hidden_state = self.rnn.forward(embedding, h_prev)
        network_outputs = []
        for hidden in hidden_states:
            network_output = self.network.forward(hidden)
            network_outputs.append(network_output)
        return hidden_states, outputs, previous_hidden_state, network_outputs, embedding