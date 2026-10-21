"""Persistence for IRM transformations tied to shared and collection indexes."""

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from collection_index import shared_representation
from im_transform import IMTransform


IRM_PARAMETER_FILE = Path(__file__).resolve().with_name("irm_parameters.json")
SCHEMA_VERSION = 1


def _canonical_hash(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _matrix_data(matrix):
    values = matrix.to_cpu() if hasattr(matrix, "to_cpu") else matrix.data
    return values.tolist() if hasattr(values, "tolist") else values


class IRMParameterStore:
    def __init__(self, path=IRM_PARAMETER_FILE):
        self.path = Path(path)

    @staticmethod
    def _context(tokenizer, embedding, candidate_records):
        if not isinstance(candidate_records, dict) or not candidate_records:
            raise ValueError("IRM requires at least one candidate collection index.")
        indexes = {}
        for collection_id, record in candidate_records.items():
            if not isinstance(collection_id, str) or not collection_id:
                raise ValueError("Candidate collection IDs must be non-empty strings.")
            if not isinstance(record, dict) or not isinstance(
                record.get("index_sha256"), str
            ):
                raise ValueError(
                    f"Candidate index metadata for {collection_id!r} is invalid."
                )
            indexes[collection_id] = record["index_sha256"]
        return {
            "shared_representation_sha256": _canonical_hash(
                shared_representation(tokenizer, embedding)
            ),
            "candidate_indexes": dict(sorted(indexes.items())),
        }

    def save(
        self,
        transform,
        tokenizer,
        embedding,
        candidate_records,
        training_metadata,
    ):
        context = self._context(tokenizer, embedding, candidate_records)
        validated = IMTransform(matrix=_matrix_data(transform.W))
        data = {
            "schema_version": SCHEMA_VERSION,
            **context,
            "training": training_metadata,
            "matrix": _matrix_data(validated.W),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix="irm_parameters_",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2, allow_nan=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.remove(temp_name)

    def load(self, transform, tokenizer, embedding, candidate_records):
        if not self.path.exists():
            return False
        with self.path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("IRM parameter file has an unsupported schema.")

        expected_context = self._context(tokenizer, embedding, candidate_records)
        actual_context = {
            "shared_representation_sha256": data.get(
                "shared_representation_sha256"
            ),
            "candidate_indexes": data.get("candidate_indexes"),
        }
        if actual_context != expected_context:
            raise ValueError(
                "Saved IRM parameters do not match the current shared "
                "representation and candidate indexes. Retrain with --train-irm."
            )
        loaded = IMTransform(matrix=data.get("matrix"))
        transform.W = transform.backend.create_matrix(_matrix_data(loaded.W))
        return True

    def archive_and_clear(self):
        if not self.path.exists():
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = self.path.with_name(
            f"{self.path.stem}.pre-shared-rebuild-{timestamp}{self.path.suffix}.bak"
        )
        backup = base
        suffix = 1
        while backup.exists():
            backup = base.with_name(f"{base.stem}-{suffix}{base.suffix}")
            suffix += 1
        os.replace(self.path, backup)
        return backup
