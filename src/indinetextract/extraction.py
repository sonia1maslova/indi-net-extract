"""Extract ordered continuous component maps from one prepared fMRI image."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import nibabel as nib
import numpy as np
import scipy
import threadpoolctl

from ._algorithms.dual_regression import fit_dual_regression, rank_info
from ._algorithms.gig_ica import fit_gig_ica
from ._types import NetworkResult
from ._validation import AFFINE_ATOL, AFFINE_RTOL, check_grid, load_image, load_mask


def _ids(component_ids, n_components):
    if component_ids is None:
        return tuple(f"C{index + 1}" for index in range(n_components))
    if isinstance(component_ids, (str, bytes)):
        raise ValueError("component_ids must be an ordered sequence of strings")
    try:
        values = tuple(component_ids)
    except TypeError as exc:
        raise ValueError("component_ids must be an ordered sequence of strings") from exc
    if len(values) != n_components:
        raise ValueError("component_ids must contain exactly one ID per template component")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("Every component ID must be a nonempty string")
    if len(set(values)) != len(values):
        raise ValueError("Component IDs must be unique")
    return values


def _source_hashes(method):
    directory = Path(__file__).resolve().parent
    algorithm = "gig_ica.py" if method == "gig_ica" else "dual_regression.py"
    relative_paths = ("extraction.py", "_validation.py", "_types.py", f"_algorithms/{algorithm}")
    result = {}
    for relative in relative_paths:
        digest = hashlib.sha256()
        with (directory / relative).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result[relative] = digest.hexdigest()
    return result


def _read_masked(image, mask, destination, *, columns, name):
    """Read one 3D volume at a time, hashing the exact masked float64 values."""
    digest = hashlib.sha256()
    flat_mask = mask.ravel(order="F")
    for index in range(image.shape[3]):
        volume = np.asanyarray(image.dataobj[..., index])
        if np.iscomplexobj(volume):
            raise ValueError(f"{name} must be real-valued")
        selected = np.ascontiguousarray(volume.ravel(order="F")[flat_mask], dtype=np.float64)
        if not np.isfinite(selected).all():
            raise ValueError(
                f"{name} volume {index + 1} contains nonfinite values inside the analysis mask"
            )
        digest.update(memoryview(selected))
        if columns:
            destination[:, index] = selected
        else:
            destination[index] = selected
    return digest.hexdigest()


def _check_temporal_variation(data, chunk_size=8192):
    """Reject constant masked BOLD voxels; do not silently shrink the mask."""
    bad_count = 0
    first_indices = []
    for start in range(0, data.shape[1], chunk_size):
        block = np.asarray(data[:, start:start + chunk_size])
        amplitude = np.max(np.abs(block), axis=0)
        spread = np.max(block, axis=0) - np.min(block, axis=0)
        tolerance = 32.0 * np.finfo(np.float64).eps * np.maximum(amplitude, 1e-300)
        constant = spread <= tolerance
        bad_count += int(constant.sum())
        if len(first_indices) < 10:
            first_indices.extend((np.flatnonzero(constant) + start)[:10 - len(first_indices)].tolist())
    if bad_count:
        raise ValueError(
            f"BOLD contains {bad_count} temporally constant voxel(s) inside the analysis mask; "
            f"first masked voxel indices (NIfTI Fortran order, zero based): {first_indices}. "
            "Provide a prepared input and an explicit compatible mask; no voxel was removed."
        )


def _check_template_design(reference):
    amplitude = np.max(np.abs(reference), axis=0)
    spread = np.max(reference, axis=0) - np.min(reference, axis=0)
    tolerance = 32.0 * np.finfo(np.float64).eps * np.maximum(amplitude, 1e-300)
    if np.any(spread <= tolerance):
        raise ValueError("Each template must be nonempty and spatially nonconstant inside the mask")
    scales = reference.std(axis=0, ddof=0)
    if np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ValueError("Each template must be nonempty and spatially nonconstant inside the mask")
    centered = (reference - reference.mean(axis=0)) / scales
    return rank_info(centered, "Spatial template design")


def _output_rank(values):
    """Report rank without adding a joint-independence rule to GIG-ICA."""
    singular = np.linalg.svd(values, compute_uv=False)
    tolerance = np.finfo(np.float64).eps * max(values.shape) * singular[0]
    retained = singular > tolerance
    return {
        "rank": int(retained.sum()),
        "n_columns": int(values.shape[1]),
        "rank_tolerance": float(tolerance),
        "singular_values": singular.tolist(),
        "full_column_rank": bool(retained.sum() == values.shape[1]),
    }


def _make_maps(components, mask_image, mask):
    data = np.zeros(mask.shape + (components.shape[1],), dtype=np.float64, order="C")
    flat_mask = mask.ravel(order="F")
    flat_volume = np.zeros(mask.size, dtype=np.float64)
    for component in range(components.shape[1]):
        flat_volume[flat_mask] = components[:, component]
        data[..., component] = flat_volume.reshape(mask.shape, order="F")
    header = nib.Nifti1Header.from_header(mask_image.header)
    header.set_data_dtype(np.float64)
    header.set_intent("none")
    header["descrip"] = b"indinetextract continuous network components"
    image = nib.Nifti1Image(data, np.asarray(mask_image.affine).copy(), header=header)
    image.header.set_zooms(tuple(mask_image.header.get_zooms()[:3]) + (1.0,))
    return image


def _extract_one(
    fmri,
    templates,
    mask,
    *,
    subject_id,
    method,
    component_ids=None,
    n_threads=2,
    work_dir=None,
) -> NetworkResult:
    """Fit classic DR or GIG-ICA to one already prepared 4D fMRI image.

    Inputs share exactly one spatial grid and an explicit binary 3D mask.
    Only masked fMRI/template values participate in validation and fitting.
    Their zeros are observations, not missing data. No denoising, input
    variance normalization, run discovery, concatenation or resampling occurs.

    A temporary float64 [time, masked voxel] memmap bounds the input-loading
    copy. GIG-ICA's unchanged core still requires in-memory centering,
    whitening and SVD workspaces; this wrapper does not make it out-of-core.
    The temporary file is removed before the independent result is returned.
    """
    started = time.perf_counter()
    if not isinstance(subject_id, str) or not subject_id.strip():
        raise ValueError("subject_id must be a nonempty string")
    if method not in ("dual_regression", "gig_ica"):
        raise ValueError("method must be 'dual_regression' or 'gig_ica'")
    if (
        isinstance(n_threads, (bool, np.bool_))
        or not isinstance(n_threads, (int, np.integer))
        or n_threads < 1
    ):
        raise ValueError("n_threads must be a positive integer")
    if work_dir is not None and not isinstance(work_dir, (str, os.PathLike)):
        raise TypeError("work_dir must be a directory path or None")
    directory = None if work_dir is None else Path(work_dir).expanduser()

    # Limits are scoped to this fit and restored on exit, including failures.
    # In particular, importing indinetextract never sets global thread environment variables.
    with threadpoolctl.threadpool_limits(limits=int(n_threads)):
        fmri_image = load_image(fmri)
        template_image = load_image(templates)
        mask_image, mask_array = load_mask(mask)
        if fmri_image.ndim != 4 or fmri_image.shape[3] < 2:
            raise ValueError("fmri must be one 4D image with at least two timepoints")
        if template_image.ndim != 4:
            raise ValueError("templates must be a 4D image with an ordered component axis")
        check_grid(fmri_image, mask_image, "fMRI")
        check_grid(template_image, mask_image, "Templates")
        n_timepoints = fmri_image.shape[3]
        n_components = template_image.shape[3]
        n_voxels = int(mask_array.sum())
        if n_voxels <= n_components:
            raise ValueError("The mask must contain more voxels than template components")
        ids = _ids(component_ids, n_components)
        reference = np.empty((n_voxels, n_components), dtype=np.float64)
        template_digest = _read_masked(
            template_image, mask_array, reference, columns=True, name="Templates"
        )
        template_rank = _check_template_design(reference)
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="indinetextract_", dir=directory) as temporary:
            data = np.memmap(
                Path(temporary) / "masked_bold.float64.dat",
                mode="w+", dtype=np.float64, shape=(n_timepoints, n_voxels), order="C",
            )
            try:
                fmri_digest = _read_masked(
                    fmri_image, mask_array, data, columns=False, name="fMRI"
                )
                _check_temporal_variation(data)
                loaded = time.perf_counter()
                if method == "dual_regression":
                    fitted = fit_dual_regression(data, reference)
                else:
                    fitted = fit_gig_ica(data, reference.T, pinv_method="svd", eigen_driver="evd")
                fitted_at = time.perf_counter()
                components = np.asarray(fitted["components"], dtype=np.float64)
                timecourses = np.ascontiguousarray(fitted["timecourses"], dtype=np.float64)
                if components.shape != (n_voxels, n_components):
                    raise RuntimeError("The numerical core returned an unexpected map shape")
                if timecourses.shape != (n_timepoints, n_components):
                    raise RuntimeError("The numerical core returned an unexpected timecourse shape")
                if not np.isfinite(components).all() or not np.isfinite(timecourses).all():
                    raise FloatingPointError("The numerical core returned nonfinite values")
                if np.any(components.std(axis=0, ddof=0) <= 0):
                    raise FloatingPointError("The numerical core returned a spatially constant component")
                if np.any(timecourses.std(axis=0, ddof=0) <= 0):
                    raise FloatingPointError("The numerical core returned a temporally constant timecourse")
                maps_image = _make_maps(components, mask_image, mask_array)
                map_rank = _output_rank(components)
                timecourse_rank = _output_rank(timecourses)
            finally:
                # Close explicitly so Windows can remove the temporary directory.
                # Neither returned image nor returned timecourses references this mapping.
                data._mmap.close()

        mask_digest = hashlib.sha256(
            memoryview(np.ascontiguousarray(mask_array.ravel(order="F")))
        ).hexdigest()
        metadata = {
            "method": method,
            "subject_id": subject_id,
            "component_ids": list(ids),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "n_timepoints": n_timepoints,
            "n_mask_voxels": n_voxels,
            "n_components": n_components,
            "n_threads": int(n_threads),
            "spatial_shape": list(mask_array.shape),
            "affine": np.asarray(mask_image.affine).tolist(),
            "grid_tolerance": {"rtol": AFFINE_RTOL, "atol": AFFINE_ATOL},
            "masking": "explicit_binary_mask_unchanged; NIfTI_Fortran_order_masked_voxel_index",
            "finite_value_scope": "fMRI_and_templates_inside_mask; entire_binary_mask",
            "numerical_constant_check": "range <= 32 * eps(float64) * max(max_abs_value, 1e-300)",
            "input_count": 1,
            "input_is_caller_prepared": True,
            "additional_cleaning": False,
            "input_temporal_variance_normalization_added": False,
            "runs_detected_or_concatenated": False,
            "resampling_added": False,
            "component_reordering_added": False,
            "sign_matching_added": False,
            "intrinsic_algorithm_operations": (
                [
                    "spatial_template_mean_and_population_SD_normalization",
                    "single_input_voxel_temporal_demeaning",
                    "stage1_timecourse_temporal_mean_and_sample_SD_normalization",
                    "stage2_intercept_and_continuous_beta",
                ]
                if method == "dual_regression" else [
                    "spatial_demeaning_of_each_timepoint",
                    "source_eigenvalue_threshold_whitening_and_conditional_sample_SD_scaling",
                    "reference_spatial_mean_and_sample_SD_normalization",
                    "independent_template_guided_component_optimization",
                    "timecourses_from_original_input_projection_divided_by_mask_voxel_count",
                ]
            ),
            "output_dtype": "float64",
            "outside_mask_value": 0.0,
            "template_rank_diagnostics": template_rank,
            "output_map_rank_diagnostics": map_rank,
            "output_timecourse_rank_diagnostics": timecourse_rank,
            "algorithm": fitted["metadata"],
            "input_sha256": {
                "fmri_masked_float64_time_then_NIfTI_Fortran_voxel": fmri_digest,
                "templates_masked_float64_component_then_NIfTI_Fortran_voxel": template_digest,
                "mask_boolean_NIfTI_Fortran_order": mask_digest,
            },
            "source_sha256": _source_hashes(method),
            "environment": {
                "executable": sys.executable,
                "python": sys.version,
                "prefix": sys.prefix,
                "platform": platform.platform(),
                "versions": {
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "nibabel": nib.__version__,
                    "threadpoolctl": threadpoolctl.__version__,
                },
                "requested_threads": int(n_threads),
                "threadpools_during_fit": threadpoolctl.threadpool_info(),
            },
            "storage": {
                "input_loading": "float64_temporary_memmap",
                "work_dir": None if directory is None else str(directory),
                "temporary_file_removed": True,
                "GIG_core_uses_in_memory_whitening_and_SVD": method == "gig_ica",
            },
            "timing_seconds": {
                "input_validation_and_loading": loaded - started,
                "algorithm": fitted_at - loaded,
                "total": time.perf_counter() - started,
            },
        }
        return NetworkResult(
            subject_id=subject_id,
            maps=maps_image,
            timecourses=timecourses,
            mask=mask_image,
            component_ids=ids,
            metadata=metadata,
        )
