"""Mask-aware smoothing and deterministic finite signed Top-N selection.

Top-N is a ranking rule. It does not establish statistical significance.
Each call starts from ``fit.maps`` and never from an earlier smoothing result.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from numbers import Integral, Real
from os import PathLike
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

from ._types import BinaryMapsResult, NetworkResult
from ._validation import AFFINE_ATOL, AFFINE_RTOL, check_grid, load_image, load_mask


_GAUSSIAN_TRUNCATE = 4.0
_SHEAR_RTOL = 1e-5
_SHEAR_ATOL = 1e-6


def _component_ids(result: Any, n_components: int) -> tuple[str, ...]:
    """Validate component labels without changing their input order."""
    ids = tuple(result.component_ids)
    if len(ids) != n_components:
        raise ValueError("component_ids must contain one label per map volume")
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("component_ids must contain nonempty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("component_ids must be unique")
    return ids


def _read_4d(image: nib.spatialimages.SpatialImage, name: str) -> np.ndarray:
    if len(image.shape) != 4 or image.shape[3] < 1:
        raise ValueError(f"{name} must be a 4D image with at least one component")
    if any(size < 1 for size in image.shape[:3]):
        raise ValueError(f"{name} must have nonempty spatial dimensions")
    # Loading scaled data directly as float64 avoids a float32 ranking round-trip.
    return np.asarray(image.dataobj, dtype=np.float64)


def _new_image(
    data: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    *,
    dtype: Any,
    description: str,
) -> nib.spatialimages.SpatialImage:
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    header.set_slope_inter(1.0, 0.0)
    header.set_intent("none")
    header["cal_min"] = 0.0
    header["cal_max"] = 0.0
    header["descrip"] = description.encode("ascii")[:79]
    image_class = (
        nib.Nifti2Image if isinstance(reference, nib.Nifti2Image) else nib.Nifti1Image
    )
    image = image_class(data, np.array(reference.affine, copy=True), header=header)
    image.header.set_data_dtype(dtype)
    image.header.set_slope_inter(1.0, 0.0)
    return image


def _array_hash_update(digest: Any, values: np.ndarray, dtype: str) -> None:
    """Hash canonical numeric values, independent of memory or byte order."""
    digest.update(json.dumps(list(values.shape), separators=(",", ":")).encode())
    digest.update(dtype.encode("ascii"))
    # Work a component at a time so hashing does not duplicate the whole 4D map.
    if values.ndim == 4:
        for component in range(values.shape[3]):
            block = np.ascontiguousarray(values[..., component], dtype=dtype)
            digest.update(memoryview(block).cast("B"))
    else:
        block = np.ascontiguousarray(values, dtype=dtype)
        digest.update(memoryview(block).cast("B"))


def _image_fingerprint(
    image: nib.spatialimages.SpatialImage,
    values: np.ndarray,
    *,
    dtype: str,
) -> str:
    # Preserve the original hash domain across the package rename.
    digest = hashlib.sha256(b"templnet-image-values-v1\0")
    _array_hash_update(digest, np.asarray(image.affine), "<f8")
    _array_hash_update(digest, values, dtype)
    return digest.hexdigest()


def _reference_provenance(
    value: Any,
    image: nib.spatialimages.SpatialImage,
    binary_values: np.ndarray,
) -> dict[str, Any]:
    """Fingerprint the actual in-mask reference and, when available, its file."""
    provenance: dict[str, Any] = {
        "source": "in_memory_image",
        "fingerprint_sha256": _image_fingerprint(
            image, binary_values, dtype="|u1"
        ),
        "fingerprint_scope": "binary_reference_in_analysis_mask_with_affine",
    }
    if isinstance(value, (str, PathLike)):
        path = Path(value).expanduser()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        provenance["source"] = str(path)
        provenance["file_sha256"] = digest.hexdigest()
    return provenance


def _binary_reference(
    value: Any,
    grid: nib.spatialimages.SpatialImage,
    mask: np.ndarray,
    n_components: int,
) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    """Read fixed binary volumes; values outside the analysis mask are ignored."""
    image = load_image(value)
    check_grid(image, grid, "reference")
    values = _read_4d(image, "reference")
    if values.shape[3] != n_components:
        raise ValueError("reference must contain the same number of components")
    inside = values[mask]
    if not np.all(np.isfinite(inside)) or not np.all(
        (inside == 0.0) | (inside == 1.0)
    ):
        raise ValueError("reference must contain only finite 0/1 values inside mask")
    binary_values = np.zeros(values.shape, dtype=bool)
    binary_values[mask] = inside.astype(bool)
    return image, binary_values


def _validate_top_n(top_n: Any, n_components: int) -> list[int]:
    if isinstance(top_n, Integral) and not isinstance(top_n, (bool, np.bool_)):
        values = [top_n] * n_components
    else:
        if isinstance(top_n, (str, bytes)) or not isinstance(
            top_n, (Sequence, np.ndarray)
        ):
            raise ValueError("top_n must be a positive integer or a per-component list")
        if isinstance(top_n, np.ndarray) and top_n.ndim != 1:
            raise ValueError("top_n array must be one-dimensional")
        values = list(top_n)
        if len(values) != n_components:
            raise ValueError("top_n must contain one integer per component")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value <= 0
        for value in values
    ):
        raise ValueError("every top_n count must be a positive integer")
    return [int(value) for value in values]


def _smoothing_parameters(
    image: nib.spatialimages.SpatialImage, fwhm_mm: Any
) -> tuple[float, np.ndarray, dict[str, Any]]:
    if isinstance(fwhm_mm, (bool, np.bool_)) or not isinstance(fwhm_mm, Real):
        raise ValueError("fwhm_mm must be a finite nonnegative number")
    fwhm = float(fwhm_mm)
    if not np.isfinite(fwhm) or fwhm < 0.0:
        raise ValueError("fwhm_mm must be finite and nonnegative")
    voxel_sizes = np.asarray(nib.affines.voxel_sizes(image.affine), dtype=np.float64)
    if not np.all(np.isfinite(voxel_sizes)) or np.any(voxel_sizes <= 0.0):
        raise ValueError("map affine must define finite positive voxel sizes")
    unit = image.header.get_xyzt_units()[0]
    mm_factor = {"mm": 1.0, "meter": 1000.0, "micron": 0.001, "unknown": 1.0}
    if unit not in mm_factor:
        raise ValueError(f"unsupported spatial unit for smoothing: {unit!r}")
    voxel_sizes_mm = voxel_sizes * mm_factor[unit]
    sigma = (fwhm / np.sqrt(8.0 * np.log(2.0))) / voxel_sizes_mm
    if not np.all(np.isfinite(sigma)):
        raise ValueError("fwhm_mm and voxel sizes produce a nonfinite Gaussian sigma")
    if fwhm > 0.0:
        directions = np.asarray(image.affine[:3, :3]) / voxel_sizes[np.newaxis, :]
        if not np.allclose(
            directions.T @ directions,
            np.eye(3),
            rtol=_SHEAR_RTOL,
            atol=_SHEAR_ATOL,
        ):
            raise ValueError(
                "positive-FWHM smoothing requires orthogonal voxel axes; "
                "a sheared affine cannot be approximated by axis-wise Gaussian filters"
            )
    metadata = {
        "method": "mask_normalized_gaussian",
        "applied": fwhm > 0.0,
        "formula": "G(M*f)/G(M)",
        "fwhm_mm": fwhm,
        "sigma_voxels": sigma.tolist(),
        "voxel_sizes_mm": voxel_sizes_mm.tolist(),
        "input_spatial_unit": unit,
        "unknown_spatial_units_assumed_mm": unit == "unknown",
        "truncate_sigma": _GAUSSIAN_TRUNCATE,
        "mode": "constant",
        "cval": 0.0,
        "spatial_dimensions": 3,
        "source": "original_fit.maps",
        "outside_mask_value": 0.0,
        "positive_fwhm_requires_finite_in_mask": True,
        "shear_policy": "reject_for_positive_fwhm",
        "orthogonality_rtol": _SHEAR_RTOL,
        "orthogonality_atol": _SHEAR_ATOL,
    }
    return fwhm, sigma, metadata


def postprocess_maps(
    fit: NetworkResult,
    *,
    fwhm_mm: float = 0.0,
    top_n: int | Sequence[int] | None = None,
    count_reference: Any = None,
) -> BinaryMapsResult:
    """Smooth original maps within the mask, then select finite signed Top-N.

    Exactly one of ``top_n`` and ``count_reference`` is required. A scalar count
    applies to every component; a sequence has one positive integer per component.
    ``count_reference`` must be a fixed binary 4D image on the same grid. Its
    per-component counts are calculated after intersecting with ``fit.mask``.

    Values are ranked as float64 in descending signed order, including real zero
    and negative values. Ties use the ascending zero-based original voxel index
    in Fortran order. Fewer than N finite in-mask candidates is an error.

    Positive-FWHM smoothing requires finite values inside the mask. Unknown NIfTI
    spatial units are explicitly recorded as assumed millimetres. Nonorthogonal
    (sheared) voxel axes are rejected for positive FWHM.
    """
    if (top_n is None) == (count_reference is None):
        raise ValueError("provide exactly one of top_n and count_reference")
    image = load_image(fit.maps)
    mask_image, mask = load_mask(fit.mask)
    check_grid(image, mask_image, "fit.maps")
    raw = _read_4d(image, "fit.maps")
    n_components = raw.shape[3]
    ids = _component_ids(fit, n_components)
    fwhm, sigma, smoothing_metadata = _smoothing_parameters(image, fwhm_mm)

    reference_metadata = None
    if count_reference is not None:
        reference_image, reference_values = _binary_reference(
            count_reference, image, mask, n_components
        )
        counts = np.count_nonzero(reference_values, axis=(0, 1, 2))
        requested = _validate_top_n(counts, n_components)
        reference_metadata = _reference_provenance(
            count_reference, reference_image, reference_values
        )
        reference_metadata["counts_in_analysis_mask"] = requested
        count_source = "reference_mask_intersection"
    else:
        requested = _validate_top_n(top_n, n_components)
        count_source = "explicit_top_n"

    continuous = np.zeros(raw.shape, dtype=np.float64)
    if fwhm == 0.0:
        continuous[mask] = raw[mask]
    else:
        if not np.all(np.isfinite(raw[mask])):
            raise ValueError(
                "positive-FWHM smoothing requires finite original values inside mask"
            )
        denominator = gaussian_filter(
            mask.astype(np.float64),
            sigma=tuple(sigma),
            mode="constant",
            cval=0.0,
            truncate=_GAUSSIAN_TRUNCATE,
        )
        if np.any(denominator[mask] <= 0.0):
            raise ValueError("mask-normalized smoothing has zero in-mask support")
        for component in range(n_components):
            numerator = gaussian_filter(
                np.where(mask, raw[..., component], 0.0),
                sigma=tuple(sigma),
                mode="constant",
                cval=0.0,
                truncate=_GAUSSIAN_TRUNCATE,
            )
            np.divide(
                numerator,
                denominator,
                out=continuous[..., component],
                where=mask,
            )

    binary = np.zeros(raw.shape, dtype=np.uint8)
    mask_flat = mask.ravel(order="F")
    selection: list[dict[str, Any]] = []
    for component, n_requested in enumerate(requested):
        values = continuous[..., component].ravel(order="F")
        candidates = np.flatnonzero(mask_flat & np.isfinite(values))
        if candidates.size < n_requested:
            raise ValueError(
                f"component {ids[component]!r} has {candidates.size} finite in-mask "
                f"candidates, fewer than requested N={n_requested}"
            )
        # candidates are already in ascending original Fortran voxel order.
        order = np.argsort(-values[candidates], kind="stable")
        selected_indices = candidates[order[:n_requested]]
        selected_values = values[selected_indices]
        selected_flat = np.zeros(mask_flat.size, dtype=np.uint8)
        selected_flat[selected_indices] = 1
        binary[..., component] = selected_flat.reshape(mask.shape, order="F")
        cutoff = float(selected_values[-1])
        selection.append(
            {
                "component_id": ids[component],
                "n_requested": n_requested,
                "n_selected": int(selected_indices.size),
                "finite_candidates": int(candidates.size),
                "selected_positive": int(np.count_nonzero(selected_values > 0.0)),
                "selected_zero": int(np.count_nonzero(selected_values == 0.0)),
                "selected_negative": int(np.count_nonzero(selected_values < 0.0)),
                "cutoff_value": cutoff,
                "cutoff_tie_candidates": int(np.count_nonzero(values[candidates] == cutoff)),
                "cutoff_tie_selected": int(np.count_nonzero(selected_values == cutoff)),
                "ranking_dtype": "float64",
            }
        )

    maps_sha256 = _image_fingerprint(image, raw, dtype="<f8")
    mask_sha256 = _image_fingerprint(mask_image, mask, dtype="|u1")
    fit_identity = {
        "subject_id": fit.subject_id,
        "component_ids": list(ids),
        "maps_sha256": maps_sha256,
        "mask_sha256": mask_sha256,
    }
    source_fingerprint = hashlib.sha256(
        json.dumps(fit_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata: dict[str, Any] = {
        "operation": "finite_signed_top_n",
        "statistical_significance": False,
        "analysis_mask_voxels": int(np.count_nonzero(mask)),
        "grid_validation": {
            "spatial_shape": "exact",
            "affine_rtol": AFFINE_RTOL,
            "affine_atol": AFFINE_ATOL,
        },
        "smoothing": smoothing_metadata,
        "top_n": requested,
        "count_source": count_source,
        "count_reference": reference_metadata,
        "ranking": {
            "dtype": "float64",
            "finite_only": True,
            "signed_values": True,
            "order": "descending",
            "tie_breaker": "ascending_zero_based_original_voxel_index_Fortran_order",
        },
        "source_fit": {
            **fit_identity,
            "fingerprint_sha256": source_fingerprint,
            "metadata": deepcopy(fit.metadata),
        },
    }
    return BinaryMapsResult(
        subject_id=fit.subject_id,
        maps=_new_image(
            binary, image, dtype=np.uint8, description="indinetextract finite signed Top-N"
        ),
        mask=mask_image,
        component_ids=ids,
        selection=selection,
        metadata=metadata,
        smoothed_maps=_new_image(
            continuous,
            image,
            dtype=np.float64,
            description="indinetextract mask-normalized continuous maps",
        ),
    )
