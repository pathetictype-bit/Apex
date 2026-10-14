
import json
import math
import os
import tempfile
from pathlib import Path

FILE_NAME = Path(__file__).resolve().with_name("knowledge.json")
MAX_CHECKPOINT_VALUE = 100.0

RNN_KEYS = ("Wxh", "Whh", "Why", "bh", "by")


def _is_finite_matrix(matrix):
    if not isinstance(matrix, list) or not matrix:
        return False

    if not all(isinstance(row, list) and row for row in matrix):
        return False

    width = len(matrix[0])

    if any(len(row) != width for row in matrix):
        return False

    return all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for row in matrix
        for value in row
    )


def _is_usable_matrix(matrix):
    return (
        _is_finite_matrix(matrix)
        and all(
            abs(value) <= MAX_CHECKPOINT_VALUE
            for row in matrix
            for value in row
        )
    )


def _matrix_shape(matrix):
    if not _is_usable_matrix(matrix):
        return None

    return len(matrix), len(matrix[0])


def _read_file():
    if not FILE_NAME.exists():
        return {}

    with FILE_NAME.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Checkpoint root must be a JSON object.")

    if "collections" in data:
        collections = data["collections"]

        if not isinstance(collections, dict):
            raise ValueError("'collections' must be a JSON object.")

        return data

    # Support the old single-model checkpoint format.
    return {"collections": {"collection_1": data}}


def _valid_collection(collection):
    if not isinstance(collection, dict):
        return False

    rnn = collection.get("rnn")
    network = collection.get("network")
    embedding = collection.get("embedding")

    if not isinstance(rnn, dict):
        return False

    if not isinstance(network, list) or not network:
        return False

    if not isinstance(embedding, dict):
        return False

    if any(
        not _is_usable_matrix(rnn.get(key))
        for key in RNN_KEYS
    ):
        return False

    for layer in network:
        if not isinstance(layer, dict):
            return False

        if not _is_usable_matrix(layer.get("weights")):
            return False

        if not _is_usable_matrix(layer.get("biases")):
            return False

    if not _is_usable_matrix(embedding.get("matrix")):
        return False

    epoch = collection.get("epoch")

    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        return False

    tokenizer = collection.get("tokenizer")

    if tokenizer is not None:
        if not isinstance(tokenizer, dict):
            return False

        if not isinstance(tokenizer.get("vocab"), list):
            return False

        if not isinstance(tokenizer.get("merges"), list):
            return False

    return True


def model_exists(collection_id):
    try:
        data = _read_file()
        collections = data.get("collections", {})
        collection = collections.get(collection_id)

        return _valid_collection(collection)

    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        return False


def _atomic_write(data):
    # Write beside the destination so replacement stays on the same filesystem.
    fd, temp_name = tempfile.mkstemp(
        prefix="knowledge_",
        suffix=".tmp",
        dir=str(FILE_NAME.parent),
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temp_name, FILE_NAME)

    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)


def save_model(
    rnn_model,
    network,
    embedding,
    current_epoch,
    collection_id,
    tokenizer=None,
    allow_update=False,
):
    if not isinstance(collection_id, str) or not collection_id.strip():
        raise ValueError("collection_id must be a non-empty string.")

    collection = {
        "epoch": current_epoch,
        "rnn": {
            key: getattr(rnn_model, key).data
            for key in RNN_KEYS
        },
        "network": [
            {
                "weights": layer.weight_matrix.data,
                "biases": layer.bias_matrix.data,
            }
            for layer in network.layers
        ],
        "embedding": {
            "matrix": embedding.embedding_matrix.data,
        },
    }

    if tokenizer is not None:
        collection["tokenizer"] = {
            "vocab": tokenizer.vocab,
            "merges": [list(pair) for pair in tokenizer.merges],
        }

    if not _valid_collection(collection):
        raise ValueError(
            f"Refusing to save invalid checkpoint for {collection_id!r}."
        )

    data = _read_file()
    collections = data.setdefault("collections", {})

    # Never silently replace an existing collection with a different
    # checkpoint. Training code must explicitly authorize replacement.
    if collection_id in collections:
        if not allow_update:
            raise FileExistsError(
                f"Collection {collection_id!r} already exists. "
                "Refusing to overwrite it without permission."
            )

        if not _valid_collection(collections[collection_id]):
            raise ValueError(
                f"Existing collection {collection_id!r} is invalid."
            )

    collections[collection_id] = collection
    _atomic_write({"collections": collections})


def _compatible(saved, current):
    return (
        _matrix_shape(saved) is not None
        and _matrix_shape(saved) == _matrix_shape(current.data)
    )


def load_model(
    rnn_model,
    network,
    embedding,
    collection_id,
    tokenizer=None,
):
    if not model_exists(collection_id):
        raise ValueError(
            f"Collection {collection_id!r} is missing or invalid."
        )

    data = _read_file()
    collection = data["collections"][collection_id]

    rnn_data = collection["rnn"]
    network_data = collection["network"]
    embedding_data = collection["embedding"]

    # Validate every matrix before modifying the model.
    for key in RNN_KEYS:
        if not _compatible(rnn_data[key], getattr(rnn_model, key)):
            raise ValueError(
                f"Incompatible RNN matrix: {key} in {collection_id!r}."
            )

    if len(network_data) != len(network.layers):
        raise ValueError(
            f"Network layer count mismatch for {collection_id!r}: "
            f"saved={len(network_data)}, current={len(network.layers)}."
        )

    for index, (saved_layer, layer) in enumerate(
        zip(network_data, network.layers)
    ):
        if not _compatible(saved_layer["weights"], layer.weight_matrix):
            raise ValueError(
                f"Weight dimensions mismatch at network layer {index}."
            )

        if not _compatible(saved_layer["biases"], layer.bias_matrix):
            raise ValueError(
                f"Bias dimensions mismatch at network layer {index}."
            )

    if not _compatible(
        embedding_data["matrix"],
        embedding.embedding_matrix,
    ):
        raise ValueError(
            f"Embedding dimensions mismatch for {collection_id!r}."
        )

    tokenizer_data = collection.get("tokenizer")

    if tokenizer is not None:
        if not isinstance(tokenizer_data, dict):
            raise ValueError(
                f"Tokenizer state missing for {collection_id!r}."
            )

        # Restore tokenizer state before changing model parameters.
        tokenizer.load_state(tokenizer_data)

    backend = rnn_model.backend

    for key in RNN_KEYS:
        setattr(
            rnn_model,
            key,
            backend.create_matrix(rnn_data[key]),
        )

    for saved_layer, layer in zip(network_data, network.layers):
        layer.weight_matrix = backend.create_matrix(
            saved_layer["weights"]
        )
        layer.bias_matrix = backend.create_matrix(
            saved_layer["biases"]
        )

    embedding.embedding_matrix = backend.create_matrix(
        embedding_data["matrix"]
    )

    return collection["epoch"]


def load_tokenizer_state(collection_id):
    try:
        data = _read_file()
        collections = data.get("collections", {})
        collection = collections.get(collection_id)

        if not _valid_collection(collection):
            return None

        tokenizer = collection.get("tokenizer")

        if not isinstance(tokenizer, dict):
            return None

        return tokenizer

    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        return None
