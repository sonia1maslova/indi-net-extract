"""Exact within-mask overlap measures for a single individual's binary maps."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import nibabel as nib
import numpy as np

from ._types import BinaryMapsResult, DiceResult
from ._validation import AFFINE_ATOL, AFFINE_RTOL, check_grid, load_image, load_mask
from .postprocessing import (
    _binary_reference,
    _component_ids,
    _image_fingerprint,
    _new_image,
    _read_4d,
    _reference_provenance,
)


def _validated_binary(
    binary: BinaryMapsResult,
) -> tuple[nib.spatialimages.SpatialImage, np.ndarray, np.ndarray, tuple[str, ...]]:
    image = load_image(binary.maps)
    mask_image, mask = load_mask(binary.mask)
    check_grid(image, mask_image, "binary.maps")
    values = _read_4d(image, "binary.maps")
    ids = _component_ids(binary, values.shape[3])
    inside = values[mask]
    if not np.all(np.isfinite(inside)) or not np.all(
        (inside == 0.0) | (inside == 1.0)
    ):
        raise ValueError("binary.maps must contain only finite 0/1 values inside mask")
    binary_values = np.zeros(values.shape, dtype=bool)
    binary_values[mask] = inside.astype(bool)
    return image, mask, binary_values, ids


def template_dice(binary: BinaryMapsResult, reference: Any) -> DiceResult:
    """Return the full exact Dice matrix, with individual rows and reference columns.

    The input volume orders are preserved. No component matching or spatial
    tolerance is applied. The same binary analysis mask is used for both sides.
    Two empty components have undefined Dice (NaN); one empty side has Dice zero.
    The matched-component scores are the matrix diagonal, including any NaNs.
    """
    image, mask, individual, ids = _validated_binary(binary)
    n_components = individual.shape[3]
    reference_image, reference_values = _binary_reference(
        reference, image, mask, n_components
    )
    individual_inside = individual[mask]
    reference_inside = reference_values[mask]
    individual_counts = np.count_nonzero(individual_inside, axis=0).astype(np.int64)
    reference_counts = np.count_nonzero(reference_inside, axis=0).astype(np.int64)
    intersections = np.einsum(
        "vi,vj->ij",
        individual_inside,
        reference_inside,
        dtype=np.int64,
        optimize=False,
    )
    denominators = individual_counts[:, np.newaxis] + reference_counts[np.newaxis, :]
    matrix = np.full((n_components, n_components), np.nan, dtype=np.float64)
    np.divide(
        2.0 * intersections,
        denominators,
        out=matrix,
        where=denominators != 0,
    )
    metadata = {
        "metric": "exact_dice",
        "formula": "2*intersection/(individual_count+reference_count)",
        "rows": "individual_input_volume_order",
        "columns": "reference_input_volume_order",
        "component_rematching": False,
        "spatial_tolerance_voxels": 0,
        "empty_both": "NaN",
        "empty_one": 0.0,
        "matched_component_rule": "matrix_diagonal_in_input_order",
        "invalid_matched_component_policy": "full_component_mean_is_NaN",
        "aggregation_scope": "single_individual",
        "analysis_mask_voxels": int(np.count_nonzero(mask)),
        "analysis_mask_sha256": _image_fingerprint(
            load_image(binary.mask), mask, dtype="|u1"
        ),
        "grid_validation": {
            "spatial_shape": "exact",
            "affine_rtol": AFFINE_RTOL,
            "affine_atol": AFFINE_ATOL,
        },
        "reference": _reference_provenance(
            reference, reference_image, reference_values
        ),
        "source_binary": {
            "subject_id": binary.subject_id,
            "component_ids": list(ids),
            "fingerprint_sha256": _image_fingerprint(image, individual, dtype="|u1"),
            "metadata": deepcopy(binary.metadata),
        },
    }
    return DiceResult(
        subject_id=binary.subject_id,
        matrix=matrix,
        component_ids=ids,
        individual_counts=individual_counts,
        reference_counts=reference_counts,
        metadata=metadata,
    )


def overlap_fraction(binary: BinaryMapsResult) -> nib.spatialimages.SpatialImage:
    """Return the fraction of K components selecting each voxel, inside the mask.

    The denominator is the full number of input components at every in-mask
    voxel. Output values outside the analysis mask are zero. This is a spatial
    participation fraction, not a statistical significance map.
    """
    image, mask, values, _ = _validated_binary(binary)
    fraction = np.mean(values, axis=3, dtype=np.float64)
    fraction[~mask] = 0.0
    return _new_image(
        fraction,
        image,
        dtype=np.float64,
        description="indinetextract network overlap fraction",
    )
