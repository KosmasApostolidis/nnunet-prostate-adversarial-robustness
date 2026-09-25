from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "experiments"))

from analyze_asd_base_effect import (  # noqa: E402
    prepare_asd_model_frame,
    regional_means,
)


def _row(
    *,
    epsilon: float,
    asd_clean: float,
    asd_adv: float,
    asd_change: float,
    region: str = "Base",
    patient: str = "wg|0|case_a",
    class_name: str = "WG",
) -> dict:
    return {
        "epsilon": epsilon,
        "class": class_name,
        "anatomical_region": region,
        "patient_id": patient,
        "row_spacing_mm": 0.5,
        "column_spacing_mm": 0.5,
        "asd_clean": asd_clean,
        "asd_adv": asd_adv,
        "asd_change": asd_change,
    }


def test_converts_pixel_asd_to_millimetres_using_in_plane_spacing():
    # Arrange
    frame = pd.DataFrame(
        [_row(epsilon=0.1, asd_clean=2.0, asd_adv=6.0, asd_change=4.0)]
    )

    # Act
    model_frame, _ = prepare_asd_model_frame(frame)

    # Assert
    assert model_frame["asd_clean_mm"].iloc[0] == pytest.approx(1.0)
    assert model_frame["asd_adv_mm"].iloc[0] == pytest.approx(3.0)
    assert model_frame["asd_change_mm"].iloc[0] == pytest.approx(2.0)


def test_keeps_only_the_requested_endpoint_epsilon():
    # Arrange
    frame = pd.DataFrame(
        [
            _row(epsilon=0.02, asd_clean=2.0, asd_adv=3.0, asd_change=1.0),
            _row(epsilon=0.1, asd_clean=2.0, asd_adv=6.0, asd_change=4.0),
        ]
    )

    # Act
    model_frame, _ = prepare_asd_model_frame(frame)

    # Assert
    assert len(model_frame) == 1
    assert model_frame["asd_change_mm"].iloc[0] == pytest.approx(2.0)


def test_drops_undefined_asd_slices_and_reports_the_exclusion_count():
    # Arrange: the second slice has an empty prediction, so ASD is undefined
    frame = pd.DataFrame(
        [
            _row(epsilon=0.1, asd_clean=2.0, asd_adv=6.0, asd_change=4.0),
            _row(epsilon=0.1, asd_clean=2.0, asd_adv=np.nan, asd_change=np.nan),
        ]
    )

    # Act
    model_frame, exclusions = prepare_asd_model_frame(frame)

    # Assert
    assert len(model_frame) == 1
    assert exclusions.loc[0, "n_endpoint_slices"] == 2
    assert exclusions.loc[0, "n_dropped_undefined_asd"] == 1
    assert exclusions.loc[0, "dropped_fraction"] == pytest.approx(0.5)


def test_rejects_anisotropic_in_plane_spacing():
    # Arrange
    row = _row(epsilon=0.1, asd_clean=2.0, asd_adv=6.0, asd_change=4.0)
    row["column_spacing_mm"] = 0.75
    frame = pd.DataFrame([row])

    # Act / Assert
    with pytest.raises(RuntimeError, match="anisotropic"):
        prepare_asd_model_frame(frame)


def test_regional_means_average_within_patient_before_across_patients():
    # Arrange: patient_a contributes two basal slices, patient_b only one
    frame = pd.DataFrame(
        [
            _row(
                epsilon=0.1, asd_clean=2.0, asd_adv=4.0, asd_change=2.0, patient="p_a"
            ),
            _row(
                epsilon=0.1, asd_clean=2.0, asd_adv=6.0, asd_change=4.0, patient="p_a"
            ),
            _row(
                epsilon=0.1, asd_clean=2.0, asd_adv=8.0, asd_change=6.0, patient="p_b"
            ),
        ]
    )
    model_frame, _ = prepare_asd_model_frame(frame)

    # Act
    means = regional_means(model_frame)

    # Assert: patient_a mean is 1.5 mm, patient_b is 3.0 mm, so the region mean
    # is 2.25 mm rather than the slice-pooled 2.0 mm
    assert means["n_patients"].iloc[0] == 2
    assert means["mean_asd_change_mm"].iloc[0] == pytest.approx(2.25)
