"""Edge-specific energy–utility attribution (ESEUA).

Pure-NumPy atlas, energy, ranking, feature, and control code lives in this
package; ``torch_ops`` is the only module that imports torch and is imported
explicitly by callers that need it.
"""

from .aggregate import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONCENTRATION_METRICS,
    CURVE_VALUES,
    REQUIRED_SUMMARY_COLUMNS,
    SchemaError,
    anatomical_category_summary,
    case_level,
    check_summary_schema,
    cohort_concentration,
    cohort_controls,
    cohort_curves,
    median_ci,
    rank_agreement_summary,
)
from .anatomical import (
    AtlasConfig,
    build_case_atlases,
    foreground_mask,
    outer_boundary_core,
    zone_interface_core,
)
from .controls import (
    BAND_EDGES_MM,
    distance_band,
    energy_matched_region,
    volume_matched_region,
)
from .damage import HIGHER_IS_BETTER, damage, fractional, primary_dice, surface_metrics
from .energy import (
    concentration_indices,
    conservation_error,
    energy_map,
    gini,
    patch_energy_table,
)
from .features import local_damage_direction, patch_static_features
from .gradients import multiscale_edge_strength, robust_normalize, roi_mask
from .patching import edge_cores, farthest_point_seeds, split_into_patches
from .sensitivity import (
    SweepPoint,
    choose_primary,
    k_stability,
    stability_table,
    top_set_jaccard,
)
from .ranking import (
    cumulative_curves,
    k_alpha,
    rank_agreement,
    rank_order,
    screen_patches,
    subcohort_case_ids,
    top_fraction_ids,
)
from .tubes import EdgeAtlas, atlas_flags, build_atlas, grow_tubes

__all__ = [
    "AtlasConfig",
    "BAND_EDGES_MM",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CONCENTRATION_METRICS",
    "CURVE_VALUES",
    "EdgeAtlas",
    "HIGHER_IS_BETTER",
    "REQUIRED_SUMMARY_COLUMNS",
    "SchemaError",
    "SweepPoint",
    "anatomical_category_summary",
    "atlas_flags",
    "build_atlas",
    "build_case_atlases",
    "case_level",
    "check_summary_schema",
    "choose_primary",
    "cohort_concentration",
    "cohort_controls",
    "cohort_curves",
    "concentration_indices",
    "conservation_error",
    "cumulative_curves",
    "damage",
    "distance_band",
    "edge_cores",
    "energy_map",
    "energy_matched_region",
    "farthest_point_seeds",
    "foreground_mask",
    "fractional",
    "gini",
    "grow_tubes",
    "k_alpha",
    "k_stability",
    "local_damage_direction",
    "median_ci",
    "multiscale_edge_strength",
    "outer_boundary_core",
    "patch_energy_table",
    "patch_static_features",
    "primary_dice",
    "rank_agreement",
    "rank_agreement_summary",
    "rank_order",
    "robust_normalize",
    "roi_mask",
    "screen_patches",
    "split_into_patches",
    "stability_table",
    "subcohort_case_ids",
    "surface_metrics",
    "top_fraction_ids",
    "top_set_jaccard",
    "volume_matched_region",
    "zone_interface_core",
]
