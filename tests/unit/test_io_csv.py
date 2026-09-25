"""Unit tests for ``mri_prostate_seg.utils.io_csv``."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from mri_prostate_seg.utils.io_csv import read_rows, write_rows, write_table


def test_write_rows_creates_parent_dir(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "out.csv"
    write_rows(str(target), [{"a": 1, "b": 2}])
    assert target.exists()


def test_write_rows_emits_header_and_rows(tmp_path: Path) -> None:
    target = tmp_path / "out.csv"
    write_rows(str(target), [{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    rows = read_rows(str(target))
    assert rows == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


def test_write_rows_respects_explicit_field_order(tmp_path: Path) -> None:
    target = tmp_path / "out.csv"
    write_rows(
        str(target),
        [{"a": 1, "b": 2}],
        fieldnames=["b", "a"],
    )
    raw = target.read_text().splitlines()
    assert raw[0] == "b,a"


def test_write_rows_empty_iterable_writes_empty_file(tmp_path: Path) -> None:
    target = tmp_path / "out.csv"
    write_rows(str(target), [])
    assert target.exists()
    assert target.read_text() == ""


def test_write_rows_empty_iterable_with_explicit_fields_writes_header(
    tmp_path: Path,
) -> None:
    target = tmp_path / "out.csv"
    write_rows(str(target), [], fieldnames=["epsilon", "dice"])
    raw = target.read_text().splitlines()
    assert raw == ["epsilon,dice"]


def test_write_rows_field_order_taken_from_first_row(tmp_path: Path) -> None:
    target = tmp_path / "out.csv"
    write_rows(str(target), [{"x": 1, "y": 2, "z": 3}])
    raw = target.read_text().splitlines()
    assert raw[0] == "x,y,z"


def test_write_table_emits_header_then_positional_rows(tmp_path: Path) -> None:
    target = tmp_path / "out.csv"
    write_table(str(target), ["epsilon", "dice"], [(0.0, 0.91), (0.5, 0.42)])
    with target.open() as fh:
        rows = list(csv.reader(fh))
    assert rows == [["epsilon", "dice"], ["0.0", "0.91"], ["0.5", "0.42"]]


def test_write_table_creates_parent_dir(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "out.csv"
    write_table(str(target), ["a"], [[1], [2]])
    assert target.exists()


def test_write_table_no_parent_dir_in_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    write_table("flat.csv", ["a"], [[1]])
    assert (tmp_path / "flat.csv").exists()


def test_read_rows_round_trips_dict_writer_output(tmp_path: Path) -> None:
    target = tmp_path / "out.csv"
    write_rows(str(target), [{"k": "v1"}, {"k": "v2"}])
    assert read_rows(str(target)) == [{"k": "v1"}, {"k": "v2"}]


def test_read_rows_handles_empty_data_with_header_only(tmp_path: Path) -> None:
    target = tmp_path / "out.csv"
    target.write_text("a,b\n")
    assert read_rows(str(target)) == []


def test_write_rows_no_parent_dir_in_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    write_rows("flat.csv", [{"a": 1}])
    assert (tmp_path / "flat.csv").exists()
