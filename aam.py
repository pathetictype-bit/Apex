
class AAM:
    """Apparent Attention Mechanism: mean-pool query token embeddings."""

    @staticmethod
    def mean_pool(token_vectors):
        if not token_vectors:
            raise ValueError("AAM requires at least one token vector.")

        dimensions = len(token_vectors[0])

        if dimensions == 0:
            raise ValueError("Token vectors cannot be empty.")

        for vector in token_vectors:
            if len(vector) != dimensions:
                raise ValueError(
                    "All token vectors must have the same dimensions."
                )

        query_vector = [0.0] * dimensions

        for vector in token_vectors:
            for i, value in enumerate(vector):
                query_vector[i] += value

        token_count = len(token_vectors)

        for i in range(dimensions):
            query_vector[i] /= token_count

        return query_vector
