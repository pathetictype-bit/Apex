"""Build and persist versioned content-derived collection indices.

Collection IDs are caller-supplied identities. Content and representation
hashes identify versions only; they never generate or replace an ID.
"""

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path

from aam import AAM


INDEX_FILE = Path(__file__).resolve().with_name("collection_indices.json")
STORE_SCHEMA_VERSION = 1
SHARED_REPRESENTATION_VERSION = 1
TOKENIZER_VERSION = "character_bpe_v1"
AAM_VERSION = "mean_pool_v1"
TOKEN_CENTROID = "token_centroid"
CHUNK_CENTROID = "chunk_centroid"


def _canonical_hash(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _collection_id(value):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("collection_id must be a non-empty, trimmed string.")
    return value


def _embedding_rows(embedding):
    matrix = embedding.embedding_matrix
    values = matrix.to_cpu() if hasattr(matrix, "to_cpu") else matrix.data
    values = values.tolist() if hasattr(values, "tolist") else values
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError("The shared embedding matrix must be non-empty.")

    rows = []
    width = None
    for row in values:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or not row:
            raise ValueError("The shared embedding matrix must be rectangular.")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("The shared embedding matrix must be rectangular.")
        converted = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("Embedding values must be real numbers.")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("Embedding values must be finite.")
            converted.append(value)
        rows.append(converted)
    return rows


def shared_representation(tokenizer, embedding):
    """Describe the exact tokenizer/E representation used to make an index."""
    rows = _embedding_rows(embedding)
    tokenizer_state = {
        "vocab": list(tokenizer.vocab),
        "merges": [list(pair) for pair in tokenizer.merges],
    }
    return {
        "version": SHARED_REPRESENTATION_VERSION,
        "tokenizer": getattr(tokenizer, "encoding", TOKENIZER_VERSION),
        "tokenizer_sha256": _canonical_hash(tokenizer_state),
        "embedding_sha256": _canonical_hash(rows),
        "aam": AAM_VERSION,
        "dimension": len(rows[0]),
    }


def _normalise(vector):
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("Cannot normalize a zero or non-finite collection index.")
    return [value / norm for value in vector]


def _mean_vectors(vectors, dimension):
    return [
        math.fsum(vector[index] for vector in vectors) / len(vectors)
        for index in range(dimension)
    ]


def _index_vector(token_ids, embeddings, method, chunk_size):
    if not token_ids:
        raise ValueError("Source content must produce at least one token.")
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        or token_id >= len(embeddings)
        for token_id in token_ids
    ):
        raise ValueError("Tokenizer produced an ID outside the shared vocabulary.")

    token_vectors = [embeddings[token_id] for token_id in token_ids]
    dimension = len(embeddings[0])
    if method == TOKEN_CENTROID:
        # Uniform token weights reduce to AAM's token mean before L2 normalization.
        return _normalise(_mean_vectors(token_vectors, dimension))

    if method == CHUNK_CENTROID:
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        chunk_vectors = [
            AAM.mean_pool(token_vectors[start:start + chunk_size])
            for start in range(0, len(token_vectors), chunk_size)
        ]
        # Average raw chunk means equally, then normalize once. Normalizing each
        # chunk first would give short and long chunks different token weighting.
        return _normalise(_mean_vectors(chunk_vectors, dimension))

    raise ValueError(
        f"Unsupported index method {method!r}; expected "
        f"{TOKEN_CENTROID!r} or {CHUNK_CENTROID!r}."
    )


def build_collection_index(
    collection_id,
    source_content,
    tokenizer,
    embedding,
    method=TOKEN_CENTROID,
    chunk_size=128,
    version=1,
):
    """Build one immutable index-version record from recorded source text."""
    collection_id = _collection_id(collection_id)
    if not isinstance(source_content, str) or not source_content.strip():
        raise ValueError("source_content must be a non-empty string.")
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise ValueError("version must be a positive integer.")

    embeddings = _embedding_rows(embedding)
    token_ids = tokenizer.encode(source_content)
    vector = _index_vector(token_ids, embeddings, method, chunk_size)
    if method == TOKEN_CENTROID:
        config = {
            "weighting": "uniform",
            "aggregation": "mean_token_embeddings",
            "final_normalization": "l2",
        }
        method_version = "token_centroid_v1"
    elif method == CHUNK_CENTROID:
        config = {
            "chunk_size": chunk_size,
            "overlap": 0,
            "aggregation": "mean_unnormalized_chunk_vectors",
            "final_normalization": "l2",
        }
        method_version = "chunk_centroid_v1"
    else:
        raise ValueError(
            f"Unsupported index method {method!r}; expected "
            f"{TOKEN_CENTROID!r} or {CHUNK_CENTROID!r}."
        )

    return {
        "record_schema_version": STORE_SCHEMA_VERSION,
        "collection_id": collection_id,
        "index_version": version,
        "source_content": source_content,
        "source_sha256": hashlib.sha256(source_content.encode("utf-8")).hexdigest(),
        "shared_representation": shared_representation(tokenizer, embedding),
        "method": method,
        "method_version": method_version,
        "config": config,
        "vector": vector,
        "index_sha256": _canonical_hash(vector),
    }


def require_compatible_index(record, tokenizer, embedding):
    """Reject indices built with a different tokenizer or shared embedding."""
    if not isinstance(record, Mapping):
        raise TypeError("index record must be a mapping.")
    if record.get("shared_representation") != shared_representation(tokenizer, embedding):
        raise ValueError(
            "Collection index was built with a different shared representation."
        )


class CollectionIndexStore:
    """Append-only index versions stored separately from model checkpoints."""

    def __init__(self, path=INDEX_FILE):
        self.path = Path(path)

    def _read(self):
        if not self.path.exists():
            return {"schema_version": STORE_SCHEMA_VERSION, "collections": {}}
        with self.path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict) or data.get("schema_version") != STORE_SCHEMA_VERSION:
            raise ValueError("Collection index store has an unsupported schema.")
        collections = data.get("collections")
        if not isinstance(collections, dict):
            raise ValueError("Collection index store 'collections' must be an object.")
        for collection_id, versions in collections.items():
            _collection_id(collection_id)
            if not isinstance(versions, list) or not versions:
                raise ValueError(
                    f"Index versions for {collection_id!r} must be a non-empty list."
                )
            for index_version, record in enumerate(versions, start=1):
                if (
                    not isinstance(record, dict)
                    or record.get("collection_id") != collection_id
                    or record.get("index_version") != index_version
                ):
                    raise ValueError(
                        f"Invalid index version history for {collection_id!r}."
                    )
        return data

    def _write(self, data):
        fd, temp_name = tempfile.mkstemp(
            prefix="collection_indices_", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2, ensure_ascii=False, allow_nan=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.remove(temp_name)

    def register(
        self,
        collection_id,
        source_content,
        tokenizer,
        embedding,
        method=TOKEN_CENTROID,
        chunk_size=128,
        allow_new_version=False,
    ):
        """Append an index version; duplicate IDs require explicit versioning."""
        collection_id = _collection_id(collection_id)
        if not isinstance(allow_new_version, bool):
            raise TypeError("allow_new_version must be a boolean.")
        data = self._read()
        versions = data["collections"].get(collection_id, [])
        if versions and not allow_new_version:
            raise FileExistsError(
                f"Collection {collection_id!r} already has an index; "
                "pass allow_new_version=True to append a version."
            )
        record = build_collection_index(
            collection_id,
            source_content,
            tokenizer,
            embedding,
            method=method,
            chunk_size=chunk_size,
            version=len(versions) + 1,
        )
        if versions and all(
            versions[-1].get(key) == record.get(key)
            for key in (
                "source_content",
                "source_sha256",
                "shared_representation",
                "method",
                "method_version",
                "config",
                "vector",
                "index_sha256",
            )
        ):
            return versions[-1]
        data["collections"][collection_id] = versions + [record]
        self._write(data)
        return record

    def latest(self, collection_id, tokenizer=None, embedding=None):
        collection_id = _collection_id(collection_id)
        versions = self._read()["collections"].get(collection_id)
        if not versions:
            raise KeyError(f"No index is registered for {collection_id!r}.")
        if (tokenizer is None) != (embedding is None):
            raise ValueError("Provide both tokenizer and embedding to check compatibility.")
        record = versions[-1]
        if tokenizer is not None:
            require_compatible_index(record, tokenizer, embedding)
        return record

    def latest_vectors(self, tokenizer, embedding):
        """Return compatible latest ID→vector candidates for the current E/tokenizer."""
        data = self._read()
        result = {}
        for collection_id, versions in data["collections"].items():
            record = versions[-1]
            require_compatible_index(record, tokenizer, embedding)
            result[collection_id] = list(record["vector"])
        return result

    def latest_records(self):
        """Return the latest stored record for each collection ID."""
        data = self._read()
        return {
            collection_id: versions[-1]
            for collection_id, versions in data["collections"].items()
        }
