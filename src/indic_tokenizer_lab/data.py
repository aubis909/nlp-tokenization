"""Input readers and deterministic corpus preparation."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

from .text import normalize


def read_text(path: str | Path, text_column: str = "text", limit: int | None = None,
              random_sample: bool = False, seed: int = 17) -> list[str]:
    """Read text rows, optionally reservoir-sampling a fixed-size corpus in bounded memory."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer")
    if random_sample and limit is None:
        raise ValueError("random_sample requires a positive limit")
    path = Path(path)
    suffix = path.suffix.casefold()
    def source_rows():
        if suffix in {".txt", ".text"}:
            with path.open(encoding="utf-8-sig") as stream:
                yield from (line.strip() for line in stream if line.strip())
        elif suffix in {".csv", ".tsv"}:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream, delimiter="\t" if suffix == ".tsv" else ",")
                if not reader.fieldnames or text_column not in reader.fieldnames:
                    raise ValueError(f"{path} must have a '{text_column}' column; found {reader.fieldnames}")
                yield from (row[text_column] for row in reader if row.get(text_column))
        elif suffix in {".jsonl", ".ndjson"}:
            with path.open(encoding="utf-8-sig") as stream:
                yield from (json.loads(line)[text_column] for line in stream if line.strip())
        else:
            raise ValueError("Input must be .txt, .csv, .tsv, or .jsonl")

    rows = []
    rng = random.Random(seed)
    rows_seen = 0
    for row in source_rows():
        if str(row).strip():
            normalized = normalize(str(row))
            if random_sample:
                rows_seen += 1
                if len(rows) < limit:
                    rows.append(normalized)
                else:
                    replacement = rng.randrange(rows_seen)
                    if replacement < limit:
                        rows[replacement] = normalized
            else:
                rows.append(normalized)
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def split_corpus(rows: list[str], test_fraction: float = 0.2, seed: int = 17):
    """Stable seeded split. Keep near-duplicate exact strings in only one partition."""
    unique = list(dict.fromkeys(rows))
    random.Random(seed).shuffle(unique)
    cutoff = max(1, round(len(unique) * (1 - test_fraction)))
    train, test = unique[:cutoff], unique[cutoff:]
    if not test:
        raise ValueError("Need at least two distinct lines to make a train/test split")
    return train, test


def write_lines(rows: Iterable[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.writelines(row.replace("\n", " ") + "\n" for row in rows)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
