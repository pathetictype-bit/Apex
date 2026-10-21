"""Parameter-free scaled dot-product attention using APEX backends.

The component consumes explicit query, key, and value matrices. It does not
own trainable parameters and is not part of AAM or the PNSN retrieval path.
"""

import math
from collections.abc import Sequence
from numbers import Real

from CPU_backend import CPUBackend


class Attention:
    """Compute ``softmax(Q K^T / sqrt(d_k)) V`` with optional key masks.

    Inputs are row-major matrices: queries ``(nq, dk)``, keys ``(nk, dk)``,
    and values ``(nk, dv)``. A mask entry of ``True`` means that key is
    available to that query. A one-dimensional mask is broadcast to every
    query; a two-dimensional mask can mask each query/key pair independently.
    Rows with no available keys produce zero weights and a zero context.
    """

    def __init__(self, backend=None):
        self.backend = backend if backend is not None else CPUBackend()

    @staticmethod
    def _rows(values, name):
        if hasattr(values, "to_cpu"):
            values = values.to_cpu()
        elif hasattr(values, "data"):
            values = values.data
        if hasattr(values, "tolist"):
            values = values.tolist()
        if (not isinstance(values, Sequence)
                or isinstance(values, (str, bytes))
                or not values):
            raise ValueError(f"{name} must be a non-empty matrix.")

        rows = []
        width = None
        for row in values:
            if (not isinstance(row, Sequence)
                    or isinstance(row, (str, bytes))
                    or not row):
                raise ValueError(f"{name} must be a rectangular non-empty matrix.")
            if width is None:
                width = len(row)
            if len(row) != width:
                raise ValueError(f"{name} must be rectangular.")
            converted = []
            for value in row:
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise TypeError(f"{name} must contain only real numbers.")
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f"{name} must contain only finite values.")
                converted.append(value)
            rows.append(converted)
        return rows

    @staticmethod
    def _mask(mask, query_count, key_count):
        if mask is None:
            return [[True] * key_count for _ in range(query_count)]
        if hasattr(mask, "tolist"):
            mask = mask.tolist()
        if not isinstance(mask, Sequence) or isinstance(mask, (str, bytes)):
            raise TypeError("mask must be a boolean sequence or matrix.")
        if len(mask) == key_count and all(isinstance(item, bool) for item in mask):
            return [list(mask) for _ in range(query_count)]
        if len(mask) != query_count:
            raise ValueError(
                f"mask must have {key_count} key values or {query_count} rows."
            )
        result = []
        for row in mask:
            if (not isinstance(row, Sequence)
                    or isinstance(row, (str, bytes))
                    or len(row) != key_count
                    or any(not isinstance(item, bool) for item in row)):
                raise ValueError(
                    f"each mask row must contain {key_count} boolean values."
                )
            result.append(list(row))
        return result

    @staticmethod
    def _cpu_rows(matrix):
        data = matrix.to_cpu() if hasattr(matrix, "to_cpu") else matrix.data
        return data.tolist() if hasattr(data, "tolist") else data

    def forward(self, queries, keys, values, mask=None):
        query_rows = self._rows(queries, "queries")
        key_rows = self._rows(keys, "keys")
        value_rows = self._rows(values, "values")
        query_count, key_count = len(query_rows), len(key_rows)
        key_width = len(key_rows[0])
        value_width = len(value_rows[0])
        if len(query_rows[0]) != key_width:
            raise ValueError("queries and keys must have the same feature dimension.")
        if len(value_rows) != key_count:
            raise ValueError("keys and values must have the same sequence length.")

        allowed = self._mask(mask, query_count, key_count)
        query_matrix = self.backend.create_matrix(query_rows)
        key_matrix = self.backend.create_matrix(key_rows)
        value_matrix = self.backend.create_matrix(value_rows)

        score_matrix = self.backend.matmul(query_matrix, key_matrix.transpose())
        raw_scores = self._cpu_rows(score_matrix)
        scale = 1.0 / math.sqrt(key_width)
        weights = []
        for query_index, row in enumerate(raw_scores):
            scaled = [float(score) * scale for score in row]
            if any(not math.isfinite(score) for score in scaled):
                raise ValueError("attention scores contain non-finite values.")
            valid_scores = [
                score for score, is_allowed in zip(scaled, allowed[query_index])
                if is_allowed
            ]
            if not valid_scores:
                weights.append([0.0] * key_count)
                continue
            maximum = max(valid_scores)
            exp_values = [
                math.exp(score - maximum) if is_allowed else 0.0
                for score, is_allowed in zip(scaled, allowed[query_index])
            ]
            total = math.fsum(exp_values)
            weights.append([value / total for value in exp_values])

        weight_matrix = self.backend.create_matrix(weights)
        context = self.backend.matmul(weight_matrix, value_matrix)
        # Check the backend preserved the contract before returning its native
        # matrix type (CPU Matrix or GPU_matrix).
        context_rows = self._cpu_rows(context)
        if (len(context_rows) != query_count
                or any(len(row) != value_width for row in context_rows)):
            raise ValueError("attention backend returned an unexpected context shape.")
        return context, weight_matrix
