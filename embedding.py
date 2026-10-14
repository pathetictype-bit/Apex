import random
class Embedding:
    def __init__(self, vocab_size, embedding_size, backend):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.backend = backend
        data = [[random.uniform(-0.1, 0.1) for _ in range(embedding_size)] for _ in range(vocab_size)]
        self.embedding_matrix = backend.create_matrix(data)
    def forward(self, token_ids):
        return self.backend.embedding_lookup(self.embedding_matrix, token_ids)
    def backward(self, token_ids, output_gradient):
        return self.backend.embedding_backward(token_ids, output_gradient, self.vocab_size)