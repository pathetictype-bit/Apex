"""Atomic persistence for attention weights kept outside PNSN checkpoints."""

import json
import os
import tempfile
from pathlib import Path


ATTENTION_FILE = Path(__file__).resolve().with_name("collection_attention.json")
SCHEMA_VERSION = 1


def _json_default(value):
    """Convert NumPy-like values into JSON-serializable Python values."""
    if hasattr(value, "tolist"):
        return value.tolist()

    if hasattr(value, "item"):
        return value.item()

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable."
    )


class AttentionParameterStore:
    def __init__(self, path=ATTENTION_FILE):
        self.path = Path(path)

    def _read(self):
        if not self.path.exists():
            return {"schema_version": SCHEMA_VERSION, "collections": {}}
        with self.path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if (
            not isinstance(data, dict)
            or data.get("schema_version") != SCHEMA_VERSION
            or not isinstance(data.get("collections"), dict)
        ):
            raise ValueError("Attention parameter store has an unsupported schema.")
        return data

    def _write(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix="collection_attention_",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(
                    data,
                    file,
                    indent=2,
                    allow_nan=False,
                    default=_json_default,
                )
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.remove(temp_name)

    def save(self, collection_id, model):
        if not isinstance(collection_id, str) or not collection_id.strip():
            raise ValueError("collection_id must be a non-empty string.")
        data = self._read()
        data["collections"][collection_id] = model.state_dict()
        self._write(data)

    def load(self, collection_id, model):
        if not isinstance(collection_id, str) or not collection_id.strip():
            raise ValueError("collection_id must be a non-empty string.")
        state = self._read()["collections"].get(collection_id)
        if state is None:
            raise KeyError(
                f"No trained attention parameters are saved for {collection_id!r}."
            )
        model.load_state(state)

    def collection_ids(self):
        return set(self._read()["collections"])

    def archive_and_clear(self):
        if not self.path.exists():
            return None
        backup = self.path.with_name(
            f"{self.path.stem}.pre-shared-rebuild{self.path.suffix}.bak"
        )
        index = 1
        while backup.exists():
            backup = self.path.with_name(
                f"{self.path.stem}.pre-shared-rebuild-{index}"
                f"{self.path.suffix}.bak"
            )
            index += 1
        os.replace(self.path, backup)
        return backup
