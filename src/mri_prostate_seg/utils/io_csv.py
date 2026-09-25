"""CSV read/write helpers used by experiment runners and CLI scripts.

Mirrors the conventions used inside ``experiments/fgsm_adversarial_evaluation.py``
so that adversarial / robustness CSVs continue to be byte-comparable across
runs even after the migration into ``src/``.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Iterable, Mapping, Sequence


def write_rows(
    path: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    """Write ``rows`` (mappings) as CSV at ``path``.

    Field order is taken from ``fieldnames`` when provided, otherwise from the
    keys of the first row. Parent directories are created as needed.
    """
    rows_list = list(rows)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if fieldnames is None:
        if not rows_list:
            with open(path, "w", newline="") as fh:
                fh.write("")
            return
        fieldnames = list(rows_list[0].keys())

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)


def write_table(
    path: str,
    header: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> None:
    """Write a positional CSV (no field names) at ``path``."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def read_rows(path: str) -> list[dict[str, str]]:
    """Read a CSV with a header row into a list of dicts."""
    with open(path, "r", newline="") as fh:
        return list(csv.DictReader(fh))
