"""Deterministic synthetic inputs; no project data or external executables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from indinetextract import NetworkResult


AFFINE = np.array(
    [[2.0, 0.0, 0.0, -12.0], [0.0, 2.5, 0.0, -10.0],
     [0.0, 0.0, 3.0, -9.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)


@dataclass
class SyntheticDataset:
    subject_ids: list[str]
    fmri_paths: list[Path]
    templates: Path
    mask: Path
    template_data: np.ndarray
    mask_data: np.ndarray
    fmri_data: list[np.ndarray]
    component_ids: tuple[str, ...]


def make_synthetic_dataset(
    root: Path,
    *,
    n_subjects: int = 2,
    n_components: int = 3,
    n_timepoints: int = 48,
    seed: int = 8041,
) -> SyntheticDataset:
    """Mix non-Gaussian spatial sources into short, noisy prepared BOLD scans.

    There is one 4D image per subject. No run concatenation or run-wise
    normalization is performed. File names deliberately carry no subject ID:
    association follows the explicit order of ``subject_ids`` and ``fmri_paths``.
    """
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    shape = (5, 4, 3)
    mask_data = np.ones(shape, dtype=np.uint8)
    mask_data[0, 0, :] = 0
    mask_data[-1, -1, -1] = 0
    included = mask_data.astype(bool)
    n_voxels = int(included.sum())

    # Laplace sources have heavier tails than Gaussian noise. All columns
    # remain distinct and full rank for the small fixed seeds used here.
    sources = rng.laplace(size=(n_voxels, n_components))
    sources -= sources.mean(axis=0)
    sources /= sources.std(axis=0, ddof=0)
    template_data = np.zeros((*shape, n_components), dtype=np.float64)
    template_data[included, :] = sources
    template_path = root / "templates.nii.gz"
    mask_path = root / "mask.nii.gz"
    nib.save(nib.Nifti1Image(template_data, AFFINE), template_path)
    nib.save(nib.Nifti1Image(mask_data, AFFINE), mask_path)

    default_ids = ["participant-b", "participant-a"]
    subject_ids = [
        default_ids[i] if i < len(default_ids) else f"participant-{i + 1}"
        for i in range(n_subjects)
    ]
    fmri_paths: list[Path] = []
    fmri_data: list[np.ndarray] = []
    phase = np.linspace(0.0, 4.0 * np.pi, n_timepoints, endpoint=False)
    for i in range(n_subjects):
        timecourses = rng.normal(size=(n_timepoints, n_components))
        timecourses += np.column_stack(
            [np.sin((j + 1) * phase + 0.3 * i) for j in range(n_components)]
        )
        timecourses *= 1.0 + 0.4 * i
        bold = timecourses @ sources.T
        bold += 0.015 * rng.normal(size=bold.shape)
        # An arbitrary baseline tests whole-image temporal centering. It is
        # not a second run and must not introduce any run-wise operation.
        bold += 100.0 + 11.0 * i + np.linspace(-0.4, 0.4, n_voxels)
        data = np.zeros((*shape, n_timepoints), dtype=np.float64)
        data[included, :] = bold.T
        path = root / f"prepared_{i}.nii.gz"
        nib.save(nib.Nifti1Image(data, AFFINE), path)
        fmri_paths.append(path)
        fmri_data.append(data)

    return SyntheticDataset(
        subject_ids, fmri_paths, template_path, mask_path, template_data,
        mask_data, fmri_data, tuple(f"C{i + 1}" for i in range(n_components)),
    )


def make_fit(
    maps: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    affine: np.ndarray | None = None,
    component_ids: tuple[str, ...] | None = None,
    subject_id: str = "synthetic-person",
) -> NetworkResult:
    values = np.asarray(maps, dtype=np.float64)
    if mask is None:
        mask = np.ones(values.shape[:3], dtype=np.uint8)
    if affine is None:
        affine = AFFINE
    if component_ids is None:
        component_ids = tuple(f"C{i + 1}" for i in range(values.shape[-1]))
    n_timepoints = max(8, values.shape[-1] + 2)
    timecourses = np.arange(
        n_timepoints * values.shape[-1], dtype=np.float64
    ).reshape(n_timepoints, values.shape[-1])
    return NetworkResult(
        subject_id=subject_id,
        maps=nib.Nifti1Image(values, affine),
        timecourses=timecourses,
        mask=nib.Nifti1Image(np.asarray(mask, dtype=np.uint8), affine),
        component_ids=component_ids,
        metadata={"fixture": "synthetic", "n_threads": 1},
    )


def independent_dual_regression(
    data: np.ndarray, templates: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Direct two-stage OLS reference for the full small synthetic design.

    This uses two ``lstsq`` calls rather than the production algorithm's
    blockwise cross-products, inverse Gram and normal-equation calculations.
    The design and normalizations follow the documented mathematical contract.
    """
    included = mask.astype(bool)
    design = templates[included, :].astype(np.float64)
    design -= design.mean(axis=0)
    design /= design.std(axis=0, ddof=0)
    bold = data[included, :].T.astype(np.float64)
    bold -= bold.mean(axis=0)
    raw_timecourses = np.linalg.lstsq(design, bold.T, rcond=None)[0].T
    timecourses = raw_timecourses - raw_timecourses.mean(axis=0)
    timecourses /= timecourses.std(axis=0, ddof=1)
    temporal_design = np.column_stack((np.ones(len(timecourses)), timecourses))
    coefficients = np.linalg.lstsq(temporal_design, bold, rcond=None)[0][1:, :].T
    maps = np.zeros((*mask.shape, templates.shape[-1]), dtype=np.float64)
    maps[included, :] = coefficients
    return timecourses, maps
