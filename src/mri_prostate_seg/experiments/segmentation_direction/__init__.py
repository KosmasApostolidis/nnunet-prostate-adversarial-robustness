"""Direction of adversarial segmentation damage (Stage 1 of the spec).

Public API:
    analyze_binary_target, target_geometry, attack_success
    multiclass_transition_tables, transition_rows, TRANSITION_TYPES
    describe, bootstrap_median_ci, paired_median_difference
    SIGNED_BAND_EDGES_MM
"""

from .case import (
    SUCCESS_DELTA_DICE,
    TargetGeometry,
    analyze_binary_target,
    attack_success,
    target_geometry,
)
from .profile import SIGNED_BAND_EDGES_MM
from .statistics import bootstrap_median_ci, describe, paired_median_difference
from .zones import TRANSITION_TYPES, multiclass_transition_tables, transition_rows

__all__ = [
    "SIGNED_BAND_EDGES_MM",
    "SUCCESS_DELTA_DICE",
    "TRANSITION_TYPES",
    "TargetGeometry",
    "analyze_binary_target",
    "attack_success",
    "bootstrap_median_ci",
    "describe",
    "multiclass_transition_tables",
    "paired_median_difference",
    "target_geometry",
    "transition_rows",
]
