"""Unit tests for ``mri_prostate_seg.utils.seeding``."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

from mri_prostate_seg.utils import seeding


def test_set_seed_none_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel_state = random.getstate()
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    seeding.set_seed(None)
    # PYTHONHASHSEED should not have been set
    assert "PYTHONHASHSEED" not in os.environ
    # random state should be untouched
    assert random.getstate() == sentinel_state


def test_set_seed_makes_python_random_deterministic() -> None:
    seeding.set_seed(42)
    a = [random.random() for _ in range(5)]
    seeding.set_seed(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_seed_makes_numpy_legacy_random_deterministic() -> None:
    seeding.set_seed(7)
    a = np.random.rand(4)
    seeding.set_seed(7)
    b = np.random.rand(4)
    np.testing.assert_array_equal(a, b)


def test_set_seed_sets_pythonhashseed_env() -> None:
    seeding.set_seed(123)
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_set_seed_handles_torch_when_available() -> None:
    # If torch is importable, set_seed must not raise.
    seeding.set_seed(2026)
    try:
        import torch
    except ModuleNotFoundError:
        pytest.skip("torch not available")
    assert torch.initial_seed() == 2026


def test_seeded_rng_returns_generator_with_independent_streams() -> None:
    rng_a = seeding.seeded_rng(11)
    rng_b = seeding.seeded_rng(11)
    np.testing.assert_array_equal(rng_a.random(5), rng_b.random(5))


def test_seeded_rng_with_none_returns_entropy_seeded_generator() -> None:
    rng = seeding.seeded_rng(None)
    # Two calls produce different streams (with overwhelming probability)
    a = rng.random(3)
    b = rng.random(3)
    assert not np.array_equal(a, b)


def test_torch_seeded_generator_returns_generator_when_torch_available() -> None:
    g = seeding.torch_seeded_generator(99)
    if g is None:
        pytest.skip("torch not available")
    import torch

    assert isinstance(g, torch.Generator)
    assert g.initial_seed() == 99


def test_torch_seeded_generator_none_seed_does_not_set_seed() -> None:
    g = seeding.torch_seeded_generator(None)
    if g is None:
        pytest.skip("torch not available")
    # initial_seed() returns whatever entropy assigned — just check it doesn't crash
    g.initial_seed()


def test_seeded_rng_two_different_seeds_produce_different_streams() -> None:
    a = seeding.seeded_rng(1).random(5)
    b = seeding.seeded_rng(2).random(5)
    assert not np.array_equal(a, b)


def test_set_seed_unaffected_by_repeated_call_with_different_seed() -> None:
    seeding.set_seed(1)
    a = random.random()
    seeding.set_seed(2)
    b = random.random()
    seeding.set_seed(1)
    c = random.random()
    assert a == c
    assert b != a
