"""Helpers for explicit single- and multi-positive query supervision."""

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence

from AI import create_query_vector


def prepare_labeled_queries(examples, tokenizer, embedding, collection_indices):
    """Encode one or more explicit target collection IDs against fixed indices."""
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise TypeError("examples must be a sequence of query-label mappings.")
    if not isinstance(collection_indices, Mapping) or not collection_indices:
        raise ValueError("collection_indices must be a non-empty mapping.")

    prepared = []
    for position, example in enumerate(examples):
        if not isinstance(example, Mapping):
            raise TypeError(f"Example {position} must be a mapping.")
        query = example.get("query")
        collection_id = example.get("collection_id")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Example {position} must have a non-empty query.")
        if "collection_id" in example and "collection_ids" in example:
            raise ValueError(
                f"Example {position} must use either collection_id or "
                "collection_ids, not both."
            )
        if "collection_ids" in example:
            collection_ids = example.get("collection_ids")
            if (
                not isinstance(collection_ids, Sequence)
                or isinstance(collection_ids, (str, bytes))
                or not collection_ids
            ):
                raise ValueError(
                    f"Example {position} must have a non-empty collection_ids list."
                )
            if any(
                not isinstance(collection_id, str) or not collection_id
                for collection_id in collection_ids
            ):
                raise ValueError(
                    f"Example {position} collection_ids must contain "
                    "non-empty strings."
                )
            if len(set(collection_ids)) != len(collection_ids):
                raise ValueError(
                    f"Example {position} collection_ids must be unique."
                )
            label = tuple(collection_ids)
        else:
            collection_id = example.get("collection_id")
            if not isinstance(collection_id, str) or not collection_id:
                raise ValueError(
                    f"Example {position} must have exactly one collection_id "
                    "or a non-empty collection_ids list."
                )
            label = collection_id

        unknown_ids = set(label) - set(collection_indices) if isinstance(
            label, tuple
        ) else {label} - set(collection_indices)
        if unknown_ids:
            raise ValueError(
                f"Labels {sorted(unknown_ids)!r} are not among the fixed "
                "index candidates."
            )
        prepared.append(
            (create_query_vector(query, tokenizer, embedding), label)
        )
    if not prepared:
        raise ValueError("At least one relevance-labeled query is required.")
    return prepared


def train_relevance_epoch(transform, examples, collection_indices, learning_rate):
    """Update only W once per explicitly labeled query; return mean step loss."""
    if not examples:
        raise ValueError("At least one prepared query is required.")
    losses = [
        transform.train_step(query_vector, collection_id, collection_indices, learning_rate)
        for query_vector, collection_id in examples
    ]
    return math.fsum(losses) / len(losses)


def shuffled_label_control(examples, seed):
    """Return a seeded label derangement while keeping queries fixed."""
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise TypeError("examples must be a sequence of query-label mappings.")
    if any(not isinstance(example, Mapping) for example in examples):
        raise TypeError("Every example must be a mapping.")
    labels = [example.get("collection_id") for example in examples]
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("Every example must have one non-empty collection_id label.")
    if len(labels) < 2 or len(set(labels)) < 2:
        raise ValueError("A shuffled-label control requires at least two distinct labels.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if max(Counter(labels).values()) * 2 > len(labels):
        raise ValueError(
            "A label derangement is impossible when one label occurs in more "
            "than half of the examples."
        )

    generator = random.Random(seed)
    for _ in range(10000):
        shuffled = list(labels)
        generator.shuffle(shuffled)
        if all(original != replacement
               for original, replacement in zip(labels, shuffled)):
            break
    else:
        raise RuntimeError("Could not produce a shuffled-label derangement.")

    return [
        {**example, "collection_id": label}
        for example, label in zip(examples, shuffled)
    ]
