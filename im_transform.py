"""Isolated linear-query transformation for the PSNS retrieval experiment.

This module trains only a 64 by 64 matrix. It does not read or write the
APEX checkpoint, shared embeddings, manual indices, or collection parameters.
"""

import math
from collections.abc import Mapping, Sequence
from numbers import Real

from CPU_backend import CPUBackend
from im import IM


class IMTransform:
    """Apply ``u = Wq`` and train W with retrieval cross-entropy."""

    DIMENSION = 64

    def __init__(self, backend=None, matrix=None):
        self.backend = backend if backend is not None else CPUBackend()
        if matrix is None:
            matrix = [
                [1.0 if row == column else 0.0 for column in range(self.DIMENSION)]
                for row in range(self.DIMENSION)
            ]
        self.W = self._matrix(matrix, "W", (self.DIMENSION, self.DIMENSION))

    @staticmethod
    def _matrix_data(matrix, name):
        data = getattr(matrix, "data", matrix)
        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)) or not data:
            raise TypeError(f"{name} must be a non-empty matrix.")
        if any(
            not isinstance(row, Sequence) or isinstance(row, (str, bytes))
            for row in data
        ):
            raise TypeError(f"{name} must be a two-dimensional matrix.")
        width = len(data[0])
        if width == 0 or any(len(row) != width for row in data):
            raise ValueError(f"{name} must be rectangular and non-empty.")
        values = []
        for row in data:
            converted = []
            for value in row:
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise TypeError(f"{name} must contain only real numbers.")
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f"{name} must contain only finite values.")
                converted.append(value)
            values.append(converted)
        return values

    def _matrix(self, matrix, name, shape):
        data = self._matrix_data(matrix, name)
        actual_shape = (len(data), len(data[0]))
        if actual_shape != shape:
            raise ValueError(f"{name} must have shape {shape}; got {actual_shape}.")
        return self.backend.create_matrix(data)

    @staticmethod
    def _vector(values, name, expected_size=None):
        vector = IM._vector(values, name)
        if expected_size is not None and len(vector) != expected_size:
            raise ValueError(
                f"{name} must have {expected_size} values; got {len(vector)}."
            )
        return vector

    @staticmethod
    def _cosine_and_gradient(query, query_norm, index, index_norm):
        query_unit = [value / query_norm for value in query]
        index_unit = [value / index_norm for value in index]
        cosine = math.fsum(a * b for a, b in zip(query_unit, index_unit))
        cosine = max(-1.0, min(1.0, cosine))
        # d cos(u,v)/du = (v_hat - cos(u,v) * u_hat) / ||u||.
        gradient = [
            (v_hat - cosine * u_hat) / query_norm
            for u_hat, v_hat in zip(query_unit, index_unit)
        ]
        return cosine, gradient

    def forward(self, query_vector):
        """Return the transformed query as a backend Matrix of shape (64, 1)."""
        query = self._vector(query_vector, "query vector", self.DIMENSION)
        query_column = self.backend.create_matrix([[value] for value in query])
        transformed = self.backend.matmul(self.W, query_column)
        return self._matrix(transformed, "transformed query", (self.DIMENSION, 1))

    def transformed_vector(self, query_vector):
        return [row[0] for row in self.forward(query_vector).data]

    def rank(self, query_vector, collection_indices):
        """Rank candidates using the existing IM cosine/ranking behavior."""
        return IM().rank(self.transformed_vector(query_vector), collection_indices)

    def loss_and_gradient(self, query_vector, correct_collection_ids, collection_indices):
        """Return single- or multi-positive softmax loss and gradient for W."""
        if not isinstance(collection_indices, Mapping) or not collection_indices:
            raise ValueError("collection_indices must be a non-empty mapping.")
        if isinstance(correct_collection_ids, str):
            positive_ids = {correct_collection_ids}
        elif (
            isinstance(correct_collection_ids, Sequence)
            and not isinstance(correct_collection_ids, (str, bytes))
            and correct_collection_ids
        ):
            if any(
                not isinstance(collection_id, str) or not collection_id
                for collection_id in correct_collection_ids
            ):
                raise ValueError("Positive collection IDs must be non-empty strings.")
            if len(set(correct_collection_ids)) != len(correct_collection_ids):
                raise ValueError("Positive collection IDs must be unique.")
            positive_ids = set(correct_collection_ids)
        else:
            raise ValueError(
                "Provide one correct collection ID or a non-empty sequence "
                "of positive collection IDs."
            )
        missing_ids = positive_ids - set(collection_indices)
        if missing_ids:
            raise ValueError(
                f"Labels {sorted(missing_ids)!r} are absent from collection_indices."
            )

        query = self._vector(query_vector, "query vector", self.DIMENSION)
        transformed = self.transformed_vector(query)
        transformed_norm = math.hypot(*transformed)
        if not math.isfinite(transformed_norm) or transformed_norm == 0.0:
            raise ValueError("transformed query must have finite, nonzero norm.")

        candidate_ids = []
        scores = []
        cosine_gradients = []
        for collection_id, raw_index in collection_indices.items():
            if not isinstance(collection_id, str) or not collection_id:
                raise ValueError("collection IDs must be non-empty strings.")
            index = self._vector(
                raw_index, f"index for {collection_id!r}", self.DIMENSION
            )
            index_norm = math.hypot(*index)
            if not math.isfinite(index_norm) or index_norm == 0.0:
                raise ValueError(
                    f"index for {collection_id!r} must have finite, nonzero norm."
                )
            score, cosine_gradient = self._cosine_and_gradient(
                transformed, transformed_norm, index, index_norm
            )
            candidate_ids.append(collection_id)
            scores.append(score)
            cosine_gradients.append(cosine_gradient)

        maximum = max(scores)
        exp_scores = [math.exp(score - maximum) for score in scores]
        exp_total = math.fsum(exp_scores)
        probabilities = [value / exp_total for value in exp_scores]
        positive_positions = [
            index
            for index, collection_id in enumerate(candidate_ids)
            if collection_id in positive_ids
        ]
        positive_maximum = max(scores[index] for index in positive_positions)
        positive_exp_scores = [
            math.exp(scores[index] - positive_maximum)
            for index in positive_positions
        ]
        positive_exp_total = math.fsum(positive_exp_scores)
        positive_probabilities = {
            position: probability / positive_exp_total
            for position, probability in zip(
                positive_positions, positive_exp_scores
            )
        }
        loss = (
            math.log(exp_total)
            + maximum
            - math.log(positive_exp_total)
            - positive_maximum
        )

        # The positive target is a distribution over the labeled set, so the
        # loss maximizes its total probability without preferring one positive.
        gradient_u = [
            math.fsum(
                (
                    probabilities[candidate]
                    - positive_probabilities.get(candidate, 0.0)
                )
                * cosine_gradients[candidate][dimension]
                for candidate in range(len(candidate_ids))
            )
            for dimension in range(self.DIMENSION)
        ]
        if not all(math.isfinite(value) for value in gradient_u):
            raise ValueError("retrieval gradient contains non-finite values.")

        # Since u = Wq, dL/dW = (dL/du) q^T.
        query_row = self.backend.create_matrix([query])
        gradient_column = self.backend.create_matrix(
            [[value] for value in gradient_u]
        )
        gradient_w = self._matrix(
            self.backend.matmul(gradient_column, query_row),
            "gradient_W",
            (self.DIMENSION, self.DIMENSION),
        )
        return {
            "loss": loss,
            "scores": dict(zip(candidate_ids, scores)),
            "probabilities": dict(zip(candidate_ids, probabilities)),
            "positive_probability_mass": math.fsum(
                probabilities[index] for index in positive_positions
            ),
            "transformed_query": transformed,
            "gradient_u": gradient_u,
            "gradient_W": gradient_w,
        }

    def train_step(self, query_vector, correct_collection_ids, collection_indices,
                   learning_rate):
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, Real):
            raise TypeError("learning_rate must be a finite positive real number.")
        learning_rate = float(learning_rate)
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be a finite positive real number.")

        result = self.loss_and_gradient(
            query_vector, correct_collection_ids, collection_indices
        )
        scaled_gradient = self.backend.scalar_multiplication(
            result["gradient_W"], learning_rate
        )
        updated_w = self.backend.subtract(self.W, scaled_gradient)
        # Validate before replacing the current matrix so overflow cannot leave
        # the experiment in a corrupted state.
        self.W = self._matrix(
            updated_w, "updated W", (self.DIMENSION, self.DIMENSION)
        )
        return result["loss"]
