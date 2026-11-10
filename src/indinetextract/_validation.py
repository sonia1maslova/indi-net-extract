"""Shared, explicit NIfTI and analysis-support checks.

These functions never resample an image, fill missing observations, change an
analysis mask, or infer a new support from an image's nonzero values.
"""
from __future__ import annotations

import os
from pathlib import Path

import nibabel as nib
import numpy as np


AFFINE_RTOL = 1e-5
AFFINE_ATOL = 1e-5


def load_image(value):
    """Load a real three/four-dimensional NIfTI path or accept a NIfTI image.

    Loading a path keeps nibabel's lazy data proxy. Intensity values are
    checked by the operation that knows the applicable analysis support.
    """
    if isinstance(value, (str, os.PathLike)):
        image = nib.load(str(Path(value).expanduser()), mmap=True, keep_file_open=True)
    else:
        image = value
    if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
        raise TypeError("Expected a NIfTI-1/NIfTI-2 image or a NIfTI file path")
    if image.ndim not in (3, 4) or any(size < 1 for size in image.shape):
        raise ValueError("NIfTI images must have three or four nonempty dimensions")
    if np.dtype(image.get_data_dtype()).kind not in "biuf":
        raise ValueError("NIfTI intensities must be real numeric values")
    affine = np.asarray(image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("The NIfTI spatial affine must be finite and 4 by 4")
    if not np.allclose(affine[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-8):
        raise ValueError("The NIfTI spatial affine has an invalid last row")
    if np.linalg.matrix_rank(affine[:3, :3]) != 3:
        raise ValueError("The NIfTI spatial affine is singular")
    return image


def check_grid(image, reference, name):
    """Require identical spatial dimensions and numerically matching affines.

    The affine comparison uses rtol=atol=1e-5. This tolerance permits ordinary
    NIfTI header roundoff; it does not authorize registration or resampling.
    """
    image = load_image(image)
    reference = load_image(reference)
    if tuple(image.shape[:3]) != tuple(reference.shape[:3]):
        raise ValueError(
            f"{name} spatial shape {image.shape[:3]} does not match "
            f"the reference {reference.shape[:3]}"
        )
    if not np.allclose(
        image.affine, reference.affine, rtol=AFFINE_RTOL, atol=AFFINE_ATOL
    ):
        raise ValueError(f"{name} affine does not match the reference spatial grid")


def load_mask(value):
    """Return (original image, Boolean array) for a finite, nonempty 0/1 mask."""
    image = load_image(value)
    if image.ndim != 3:
        raise ValueError("The analysis mask must be a three-dimensional NIfTI")
    values = np.asanyarray(image.dataobj)
    if np.iscomplexobj(values):
        raise ValueError("The analysis mask must be real-valued")
    if not np.isfinite(values).all():
        raise ValueError("The analysis mask contains nonfinite values")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("The analysis mask must contain only exact binary 0/1 values")
    mask = np.asarray(values == 1, dtype=bool)
    if not mask.any():
        raise ValueError("The analysis mask is empty")
    return image, mask


def load_binary_reference(value, mask_img, mask_array, expected_k):
    """Return a Boolean 4D reference, preserving its input component order.

    Reference values inside the mask must be finite and binary, and every
    component must have positive support there. Outside-mask values are ignored
    and returned as zero. The supplied analysis mask itself is never changed.
    """
    image = load_image(value)
    check_grid(image, mask_img, "Binary reference")
    if (
        isinstance(expected_k, (bool, np.bool_))
        or not isinstance(expected_k, (int, np.integer))
        or expected_k < 1
    ):
        raise ValueError("expected_k must be a positive integer")
    if image.ndim != 4 or image.shape[3] != expected_k:
        raise ValueError(f"Binary reference must be 4D with {expected_k} components")
    mask_img, checked_mask = load_mask(mask_img)
    supplied_mask = np.asarray(mask_array)
    if supplied_mask.dtype.kind != "b" or supplied_mask.shape != tuple(mask_img.shape):
        raise ValueError("mask_array must be a Boolean array matching mask_img")
    if not np.array_equal(supplied_mask, checked_mask):
        raise ValueError("mask_array does not match the binary values in mask_img")
    binary = np.zeros(image.shape, dtype=bool)
    for component in range(expected_k):
        volume = np.asanyarray(image.dataobj[..., component])
        if np.iscomplexobj(volume):
            raise ValueError("The binary reference must be real-valued")
        selected = volume[supplied_mask]
        if not np.isfinite(selected).all():
            raise ValueError(
                f"Binary reference component {component + 1} is nonfinite inside the mask"
            )
        if not np.all((selected == 0) | (selected == 1)):
            raise ValueError(
                f"Binary reference component {component + 1} must contain binary 0/1 values inside the mask"
            )
        binary[..., component][supplied_mask] = selected == 1
        if not binary[..., component][supplied_mask].any():
            raise ValueError(
                f"Binary reference component {component + 1} is empty inside the analysis mask"
            )
    return binary
