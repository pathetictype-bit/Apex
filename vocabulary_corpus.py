"""Persistent text sources used to prepare the shared tokenizer vocabulary."""

import json
import os
import tempfile
from pathlib import Path


VOCABULARY_CORPUS_FILE = Path(__file__).resolve().with_name(
    "vocabulary_corpus.json"
)
SCHEMA_VERSION = 1


class VocabularyCorpusStore:
    def __init__(self, path=VOCABULARY_CORPUS_FILE):
        self.path = Path(path)

    def _read_pairs(self):
        if not self.path.exists():
            return []

        with self.path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if (
            not isinstance(data, dict)
            or data.get("schema_version") != SCHEMA_VERSION
            or not isinstance(data.get("pairs"), list)
        ):
            raise ValueError("Vocabulary corpus has an unsupported schema.")

        pairs = []
        for pair in data["pairs"]:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(not isinstance(text, str) or not text.strip() for text in pair)
            ):
                raise ValueError("Vocabulary corpus contains an invalid text pair.")
            pairs.append(tuple(pair))
        return pairs

    def _write_pairs(self, pairs):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix="vocabulary_corpus_",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "pairs": [list(pair) for pair in pairs],
                    },
                    file,
                    indent=2,
                    ensure_ascii=False,
                )
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.remove(temp_name)

    def add_pairs(self, new_pairs):
        pairs = self._read_pairs()
        original_pairs = list(pairs)
        seen = set(pairs)
        for pair in new_pairs:
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or any(not isinstance(text, str) or not text.strip() for text in pair)
            ):
                raise ValueError("Vocabulary sources must be non-empty text pairs.")
            normalized_pair = tuple(pair)
            if normalized_pair not in seen:
                pairs.append(normalized_pair)
                seen.add(normalized_pair)

        if pairs != original_pairs:
            self._write_pairs(pairs)
        return pairs
