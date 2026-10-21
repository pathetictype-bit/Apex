import json
import math
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

FILE_NAME = Path(__file__).resolve().with_name("knowledge.json")
MAX_CHECKPOINT_VALUE = 100.0
EMBEDDING_SIZE = 64
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
        return {"shared": None, "collections": {}}
    with FILE_NAME.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("Checkpoint root must be a JSON object.")
    collections = data.get("collections", {})
    if not isinstance(collections, dict):
        raise ValueError("'collections' must be a JSON object.")
    return {"shared": data.get("shared"), "collections": collections}


def _valid_shared(shared):
    if not isinstance(shared, dict):
        return False
    tokenizer = shared.get("tokenizer")
    embedding = shared.get("embedding")
    if not isinstance(tokenizer, dict) or not isinstance(embedding, dict):
        return False
    vocab = tokenizer.get("vocab")
    merges = tokenizer.get("merges")
    encoding = tokenizer.get("encoding", "character_bpe_v1")
    matrix = embedding.get("matrix")
    if not isinstance(vocab, list) or not vocab or not all(
        isinstance(token, str) for token in vocab
    ):
        return False
    if len(set(vocab)) != len(vocab) or "<UNK>" not in vocab:
        return False
    if not isinstance(merges, list):
        return False
    if encoding != "character_bpe_v1":
        return False
    shape = _matrix_shape(matrix)
    return shape == (len(vocab), EMBEDDING_SIZE)


def _valid_collection(collection):
    if not isinstance(collection, dict):
        return False
    # Shared model state must never be duplicated in a collection record.
    if "embedding" in collection or "tokenizer" in collection:
        return False
    rnn = collection.get("rnn")
    network = collection.get("network")
    epoch = collection.get("epoch")
    if not isinstance(rnn, dict) or not isinstance(network, list) or not network:
        return False
    if any(not _is_usable_matrix(rnn.get(key)) for key in RNN_KEYS):
        return False
    for layer in network:
        if not isinstance(layer, dict):
            return False
        if not _is_usable_matrix(layer.get("weights")):
            return False
        if not _is_usable_matrix(layer.get("biases")):
            return False
    return isinstance(epoch, int) and not isinstance(epoch, bool) and epoch >= 0


def _valid_checkpoint(data):
    return (
        isinstance(data, dict)
        and _valid_shared(data.get("shared"))
        and isinstance(data.get("collections"), dict)
        and all(_valid_collection(c) for c in data["collections"].values())
    )


def _atomic_write(data):
    fd, temp_name = tempfile.mkstemp(
        prefix="knowledge_", suffix=".tmp", dir=str(FILE_NAME.parent)
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


def _matrix_data(matrix):
    data = matrix.to_cpu() if hasattr(matrix, "to_cpu") else matrix.data
    return data.tolist() if hasattr(data, "tolist") else data


def _shared_record(tokenizer, embedding):
    return {
        "tokenizer": {
            "encoding": tokenizer.encoding,
            "vocab": list(tokenizer.vocab),
            "merges": [list(pair) for pair in tokenizer.merges],
        },
        "embedding": {"matrix": _matrix_data(embedding.embedding_matrix)},
    }


def shared_state_exists():
    try:
        return _valid_shared(_read_file().get("shared"))
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        return False


def initialize_shared_state(tokenizer, embedding):
    """Persist the one model-level tokenizer and embedding table exactly once."""
    data = _read_file()
    if data.get("shared") is not None:
        raise FileExistsError("Shared model state already exists.")
    if data.get("collections"):
        raise ValueError(
            "Cannot initialize shared state while incompatible collection "
            "records exist. Back up and explicitly reset the checkpoint first."
        )
    data["shared"] = _shared_record(tokenizer, embedding)
    data["collections"] = {}
    if not _valid_checkpoint(data):
        raise ValueError("Refusing to initialize invalid shared model state.")
    _atomic_write(data)


def archive_checkpoint():
    """Copy the current checkpoint to a unique sibling backup, if present."""
    if not FILE_NAME.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = FILE_NAME.with_name(
        f"{FILE_NAME.stem}.pre-vocab-rebuild-{timestamp}{FILE_NAME.suffix}.bak"
    )
    backup = base
    suffix = 1
    while backup.exists():
        backup = base.with_name(f"{base.stem}-{suffix}{base.suffix}")
        suffix += 1
    shutil.copy2(FILE_NAME, backup)
    return backup


def reinitialize_shared_state(tokenizer, embedding):
    """Replace shared state and clear incompatible collection parameters."""
    data = {
        "shared": _shared_record(tokenizer, embedding),
        "collections": {},
    }
    if not _valid_checkpoint(data):
        raise ValueError("Refusing to initialize invalid shared model state.")
    _atomic_write(data)


def load_shared_state(tokenizer, embedding):
    """Restore the shared tokenizer and exact embedding values in place."""
    data = _read_file()
    shared = data.get("shared")
    if not _valid_shared(shared):
        raise ValueError("Shared model state is missing or invalid.")
    matrix = shared["embedding"]["matrix"]
    if _matrix_shape(matrix) != (
        embedding.vocab_size,
        embedding.embedding_size,
    ):
        raise ValueError("Shared embedding dimensions do not match the model.")
    tokenizer.load_state(shared["tokenizer"])
    embedding.embedding_matrix = embedding.backend.create_matrix(matrix)
    return tokenizer, embedding


def model_exists(collection_id):
    try:
        data = _read_file()
        collection = data["collections"].get(collection_id)
        return _valid_shared(data.get("shared")) and _valid_collection(collection)
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        return False


def model_collection_ids():
    """Return IDs with valid saved model states under the current shared state."""
    data = _read_file()
    if not _valid_shared(data.get("shared")):
        raise ValueError("Shared model state is missing or invalid.")
    return {
        collection_id
        for collection_id, collection in data["collections"].items()
        if _valid_collection(collection)
    }


def next_collection_id():
    """Return the next unused numeric collection ID without altering state."""
    collections = _read_file()["collections"]
    existing_numbers = [
        int(collection_id[len("collection_"):])
        for collection_id in collections
        if isinstance(collection_id, str)
        and collection_id.startswith("collection_")
        and collection_id[len("collection_"):].isdecimal()
    ]
    return f"collection_{max(existing_numbers, default=0) + 1}"


def save_model(
    rnn_model,
    network,
    embedding,
    current_epoch,
    collection_id,
    allow_update=False,
):
    if not isinstance(collection_id, str) or not collection_id.strip():
        raise ValueError("collection_id must be a non-empty string.")
    data = _read_file()
    if not _valid_shared(data.get("shared")):
        raise ValueError("Initialize shared model state before saving a collection.")
    if _matrix_shape(_matrix_data(embedding.embedding_matrix)) != (
        len(data["shared"]["tokenizer"]["vocab"]),
        EMBEDDING_SIZE,
    ):
        raise ValueError("Shared embedding dimensions do not match shared vocabulary.")
    # Keep the one model-level matrix current. It is never copied into a
    # collection record, even when a later explicit phase trains it.
    data["shared"]["embedding"]["matrix"] = _matrix_data(
        embedding.embedding_matrix
    )
    collection = {
        "epoch": current_epoch,
        "rnn": {
            key: _matrix_data(getattr(rnn_model, key))
            for key in RNN_KEYS
        },
        "network": [
            {
                "weights": _matrix_data(layer.weight_matrix),
                "biases": _matrix_data(layer.bias_matrix),
            }
            for layer in network.layers
        ],
    }
    if not _valid_collection(collection):
        raise ValueError(f"Refusing to save invalid checkpoint for {collection_id!r}.")
    collections = data["collections"]
    if collection_id in collections:
        if not allow_update:
            raise FileExistsError(
                f"Collection {collection_id!r} already exists. "
                "Refusing to overwrite it without permission."
            )
        if not _valid_collection(collections[collection_id]):
            raise ValueError(f"Existing collection {collection_id!r} is invalid.")
    collections[collection_id] = collection
    if not _valid_checkpoint(data):
        raise ValueError("Refusing to save invalid shared checkpoint.")
    _atomic_write(data)


def _compatible(saved, current):
    return (
        _matrix_shape(saved) is not None
        and _matrix_shape(saved) == _matrix_shape(_matrix_data(current))
    )


def load_model(rnn_model, network, collection_id):
    if not model_exists(collection_id):
        raise ValueError(f"Collection {collection_id!r} is missing or invalid.")
    collection = _read_file()["collections"][collection_id]
    rnn_data = collection["rnn"]
    network_data = collection["network"]
    for key in RNN_KEYS:
        if not _compatible(rnn_data[key], getattr(rnn_model, key)):
            raise ValueError(f"Incompatible RNN matrix: {key} in {collection_id!r}.")
    if len(network_data) != len(network.layers):
        raise ValueError(
            f"Network layer count mismatch for {collection_id!r}: "
            f"saved={len(network_data)}, current={len(network.layers)}."
        )
    for index, (saved_layer, layer) in enumerate(zip(network_data, network.layers)):
        if not _compatible(saved_layer["weights"], layer.weight_matrix):
            raise ValueError(f"Weight dimensions mismatch at network layer {index}.")
        if not _compatible(saved_layer["biases"], layer.bias_matrix):
            raise ValueError(f"Bias dimensions mismatch at network layer {index}.")
    backend = rnn_model.backend
    for key in RNN_KEYS:
        setattr(rnn_model, key, backend.create_matrix(rnn_data[key]))
    for saved_layer, layer in zip(network_data, network.layers):
        layer.weight_matrix = backend.create_matrix(saved_layer["weights"])
        layer.bias_matrix = backend.create_matrix(saved_layer["biases"])
    return collection["epoch"]


def load_tokenizer_state():
    try:
        shared = _read_file().get("shared")
        if _valid_shared(shared):
            return shared["tokenizer"]
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        pass
    return None
