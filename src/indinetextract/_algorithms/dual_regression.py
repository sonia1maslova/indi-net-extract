"""Minimal classic dual regression, following the project's stock-FSL check.

The stage-one template normalization, accumulated cross-products, SVD-based
inverse Gram, stage-two intercept and normal-equation solver are adapted from
``dr_improvements_20261031/dr_core.py``. The selected route is exactly the
``official_compatible_fit`` convention in ``validate_fsl_dr.py``: one input,
``run_baselines=False``, ``temporal_ddof=1`` (FSL ``des_norm=1``).

Only continuous stage-two beta maps and normalized timecourses are calculated.
The module adds no spatial prior, frame weighting or statistical-Z output.
Caller-side validation must reject nonfinite or constant masked observations;
this implementation never discards a voxel or changes the analysis support.
"""
from __future__ import annotations

import time

import numpy as np


SOURCE_NAME = "dr_improvements_20261031/dr_core.py"
SOURCE_SHA256 = "24b366a5e8420b8ecc01f1dca59fe35c8d37f816c3ce53e204ed20e75ffd9052"
VALIDATOR_NAME = "rest_factorial_pilot/validate_fsl_dr.py"
VALIDATOR_SHA256 = "0da89b814179a215e085cc81648f77b688d37fa2d41028ea523dd3e2d2dbe15c"


def rank_info(matrix, name):
    """Use the source core's float64 SVD tolerance and full-column-rank rule."""
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    if rank != matrix.shape[1]:
        raise ValueError(
            "%s is rank deficient: rank %d, %d columns"
            % (name, rank, matrix.shape[1])
        )
    return {
        "rank": rank,
        "condition_number": float(singular[0] / singular[-1]),
        "singular_values": singular.tolist(),
        "rank_tolerance": float(tolerance),
    }


def fit_dual_regression(data, reference, *, chunk_size=8192):
    """Fit float64 classic beta maps from Y[time, voxel], S[voxel, component].

    There is one complete, already prepared fMRI input. Temporal demeaning
    here is an intrinsic regression operation; there is no input variance
    normalization, run partitioning or concatenation.
    """
    started = time.perf_counter()
    if np.iscomplexobj(data) or np.iscomplexobj(reference):
        raise ValueError("Data and templates must be real-valued")
    data = np.asarray(data)
    reference = np.asarray(reference, dtype=np.float64)
    if data.ndim != 2 or reference.ndim != 2:
        raise ValueError("Expected data [time, voxel] and reference [voxel, component]")
    n_timepoints, n_voxels = data.shape
    n_components = reference.shape[1]
    if reference.shape[0] != n_voxels or n_components < 1:
        raise ValueError("Data and reference have incompatible dimensions")
    if n_voxels <= n_components:
        raise ValueError("Insufficient masked voxels for the spatial template design")
    if n_timepoints <= n_components + 1:
        raise ValueError("Insufficient timepoints for the stage-two timecourse design")
    if (
        isinstance(chunk_size, (bool, np.bool_))
        or not isinstance(chunk_size, (int, np.integer))
        or chunk_size < 1
    ):
        raise ValueError("chunk_size must be a positive integer")
    if not np.isfinite(reference).all():
        raise ValueError("Templates contain nonfinite values inside the analysis mask")

    means = np.mean(reference, axis=0)
    spatial_scales = np.std(reference, axis=0, ddof=0)
    if np.any(spatial_scales <= 0) or not np.isfinite(spatial_scales).all():
        raise ValueError("A template is spatially constant inside the analysis mask")
    matrix = (reference - means) / spatial_scales
    template_rank = rank_info(matrix, "Spatial template design")
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    del left
    gram_inverse = (right.T / (singular * singular)) @ right

    cross = np.zeros((n_timepoints, n_components), dtype=np.float64)
    for start in range(0, n_voxels, chunk_size):
        stop = min(start + chunk_size, n_voxels)
        block = np.array(data[:, start:stop], dtype=np.float64, copy=True)
        if not np.isfinite(block).all():
            raise ValueError("BOLD values are nonfinite inside the analysis mask")
        block -= block.mean(axis=0, keepdims=True)
        cross += block @ matrix[start:stop]
    raw = cross @ gram_inverse
    centered = raw - raw.mean(axis=0, keepdims=True)
    temporal_scales = np.std(centered, axis=0, ddof=1)
    if np.any(temporal_scales <= 0) or not np.isfinite(temporal_scales).all():
        raise ValueError("A stage-one timecourse is constant or invalid")
    timecourses = centered / temporal_scales
    design = np.column_stack((np.ones(n_timepoints), timecourses))
    timecourse_rank = rank_info(design, "Stage-two timecourse design")
    linear = np.linalg.solve(design.T @ design, design.T)
    components = np.empty((n_voxels, n_components), dtype=np.float64)
    for start in range(0, n_voxels, chunk_size):
        stop = min(start + chunk_size, n_voxels)
        block = np.array(data[:, start:stop], dtype=np.float64, copy=True)
        block -= block.mean(axis=0, keepdims=True)
        components[start:stop] = (linear @ block)[1:].T
    if not np.isfinite(components).all() or not np.isfinite(timecourses).all():
        raise FloatingPointError("Classic dual regression produced nonfinite output")
    return {
        "components": components,
        "timecourses": timecourses,
        "metadata": {
            "implementation": "classic_dual_regression_FSL_des_norm_1_convention",
            "source_name": SOURCE_NAME,
            "source_sha256": SOURCE_SHA256,
            "validator_name": VALIDATOR_NAME,
            "validator_sha256": VALIDATOR_SHA256,
            "n_timepoints": n_timepoints,
            "n_voxels": n_voxels,
            "n_components": n_components,
            "spatial_template_centering": True,
            "spatial_template_standardization_ddof": 0,
            "spatial_template_means": means.tolist(),
            "spatial_template_sd": spatial_scales.tolist(),
            "temporal_centering_Y": "whole_single_input_per_voxel",
            "stage1_timecourse_centering": True,
            "temporal_standardization_ddof": 1,
            "stage1_timecourse_sd": temporal_scales.tolist(),
            "timecourses_returned": "centered_sample_SD_normalized_stage1",
            "stage2_intercept": True,
            "n_nuisance": 1,
            "run_baselines": False,
            "des_norm": 1,
            "template_rank_diagnostics": template_rank,
            "timecourse_rank_diagnostics": timecourse_rank,
            "residual_degrees_of_freedom": n_timepoints - n_components - 1,
            "map_definition": "stage2_continuous_beta",
            "output_dtype": "float64",
            "source_beta_storage_dtype": "float32",
            "spatial_prior_added": False,
            "frame_weighting_added": False,
            "fsl_invoked": False,
            "fsl_equivalence_tested_for_this_code": False,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
