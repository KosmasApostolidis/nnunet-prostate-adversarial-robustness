"""Unit tests for ``mri_prostate_seg.utils.io_json``."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mri_prostate_seg.utils.io_json import read_json, update_json, write_json


def test_write_json_creates_parent_dir(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deeper" / "out.json"
    write_json(str(target), {"a": 1})
    assert target.exists()
    assert json.loads(target.read_text()) == {"a": 1}


def test_write_json_round_trips_lists_and_floats(tmp_path: Path) -> None:
    payload = {"epsilons": [0.0, 0.5, 1.0], "name": "fgsm"}
    target = tmp_path / "out.json"
    write_json(str(target), payload)
    assert read_json(str(target)) == payload


def test_write_json_indent_default_is_4(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    write_json(str(target), {"a": 1, "b": 2})
    raw = target.read_text()
    # Default indent=4 means key starts with 4 spaces on its own line
    assert '\n    "a"' in raw


def test_write_json_indent_override(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    write_json(str(target), {"a": 1}, indent=2)
    raw = target.read_text()
    assert '\n  "a"' in raw


def test_write_json_no_parent_dir_in_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_json("flat.json", {"a": 1})
    assert (tmp_path / "flat.json").exists()


def test_read_json_returns_parsed_object(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    target.write_text('{"k": [1, 2, 3]}')
    assert read_json(str(target)) == {"k": [1, 2, 3]}


def test_read_json_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_json(str(tmp_path / "missing.json"))


def test_update_json_creates_when_missing(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    merged = update_json(str(target), {"a": 1})
    assert merged == {"a": 1}
    assert read_json(str(target)) == {"a": 1}


def test_update_json_merges_into_existing_dict(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    write_json(str(target), {"a": 1, "b": 2})
    merged = update_json(str(target), {"b": 99, "c": 3})
    assert merged == {"a": 1, "b": 99, "c": 3}
    assert read_json(str(target)) == merged


def test_update_json_raises_on_non_dict_top_level(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    target.write_text("[1, 2, 3]")
    with pytest.raises(TypeError, match="dict at the top level"):
        update_json(str(target), {"x": 1})


def test_update_json_returns_full_merged_dict(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    write_json(str(target), {"a": 1})
    merged = update_json(str(target), {"b": 2})
    assert "a" in merged and "b" in merged
