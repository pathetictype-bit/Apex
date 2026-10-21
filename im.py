import math
from collections.abc import Mapping, Sequence
from numbers import Real


class IM:
    """Score and rank collection index vectors against an AAM query vector."""

    def rank(self, query_vector, collection_indices):
        """Return ranked ``collection_id``, cosine ``score``, and ``rank`` rows.

        ``collection_indices`` maps non-empty string IDs to one-dimensional
        numeric vectors. Invalid candidates are rejected instead of silently
        being omitted. Tied scores are ordered by collection ID.
        """
        if not isinstance(collection_indices, Mapping):
            raise TypeError("collection_indices must map collection IDs to vectors.")
        if not collection_indices:
            return []

        query = self._vector(query_vector, "query vector")
        query_norm = math.hypot(*query)
        if query_norm == 0.0:
            raise ValueError("query vector must have nonzero norm.")
        normalized_query = [value / query_norm for value in query]

        scored = []
        for collection_id, index_vector in collection_indices.items():
            if not isinstance(collection_id, str) or not collection_id:
                raise ValueError("collection IDs must be non-empty strings.")

            index = self._vector(index_vector, f"index for {collection_id!r}")
            if len(index) != len(query):
                raise ValueError(
                    f"Dimension mismatch for {collection_id!r}: "
                    f"query has {len(query)} values, index has {len(index)}."
                )

            index_norm = math.hypot(*index)
            if index_norm == 0.0:
                raise ValueError(f"index for {collection_id!r} must have nonzero norm.")

            score = math.fsum(
                (query_value / query_norm) * (index_value / index_norm)
                for query_value, index_value in zip(query, index)
            )
            # Floating-point rounding can move a cosine a few ulps outside
            # its mathematical range.
            score = max(-1.0, min(1.0, score))
            scored.append((collection_id, score))

        scored.sort(key=lambda result: (-result[1], result[0]))
        return [
            {"collection_id": collection_id, "score": score, "rank": rank}
            for rank, (collection_id, score) in enumerate(scored, start=1)
        ]

    @staticmethod
    def _vector(values, name):
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"{name} must be a one-dimensional numeric sequence.")
        if not values:
            raise ValueError(f"{name} cannot be empty.")

        vector = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must contain only real numbers.")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"{name} must contain only finite values.")
            vector.append(numeric_value)
        return vector
