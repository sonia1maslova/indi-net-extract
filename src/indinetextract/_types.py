"""Small result objects; accessing them never starts a new fit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


@dataclass
class NetworkResult:
    subject_id: str
    maps: nib.Nifti1Image
    timecourses: np.ndarray
    mask: nib.Nifti1Image
    component_ids: tuple[str, ...]
    metadata: dict[str, Any]

    def save(self, output_root, *, label="extraction", overwrite=False) -> Path:
        from ._io import save_network
        return save_network(self, output_root, label=label, overwrite=overwrite)

    @classmethod
    def load(cls, directory, *, verify=True) -> NetworkResult:
        from ._io import load_network
        return load_network(directory, verify=verify)


@dataclass
class BinaryMapsResult:
    subject_id: str
    maps: nib.Nifti1Image
    mask: nib.Nifti1Image
    component_ids: tuple[str, ...]
    selection: list[dict[str, Any]]
    metadata: dict[str, Any]
    smoothed_maps: nib.Nifti1Image | None = None

    def save(self, output_root, *, label="postprocessing", overwrite=False) -> Path:
        from ._io import save_binary
        return save_binary(self, output_root, label=label, overwrite=overwrite)

    @classmethod
    def load(cls, directory, *, verify=True) -> BinaryMapsResult:
        from ._io import load_binary
        return load_binary(directory, verify=verify)


@dataclass
class DiceResult:
    subject_id: str
    matrix: np.ndarray
    component_ids: tuple[str, ...]
    individual_counts: np.ndarray
    reference_counts: np.ndarray
    metadata: dict[str, Any]

    @property
    def per_component(self) -> np.ndarray:
        return np.diag(self.matrix).copy()

    @property
    def mean_dice(self) -> float:
        values = self.per_component
        # Never increase the mean by silently dropping an invalid component.
        return float(values.mean()) if values.size and np.isfinite(values).all() else float("nan")

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "n_components": len(self.component_ids),
            "n_valid_components": int(np.isfinite(self.per_component).sum()),
            "mean_dice": self.mean_dice,
        }

    def save(self, output_root, *, label="template_dice", overwrite=False) -> Path:
        from ._io import save_dice
        return save_dice(self, output_root, label=label, overwrite=overwrite)

    @classmethod
    def load(cls, directory, *, verify=True) -> DiceResult:
        from ._io import load_dice
        return load_dice(directory, verify=verify)


@dataclass
class BatchResult:
    batch_id: str
    records: list[dict[str, Any]]
    report_dir: Path
    metadata: dict[str, Any]

    @property
    def summary(self) -> list[dict[str, Any]]:
        return self.records

    @property
    def successful(self) -> list[dict[str, Any]]:
        return [row for row in self.records if row["status"] == "success"]

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [row for row in self.records if row["status"] == "failed"]


class BatchExecutionError(RuntimeError):
    """A fail-fast batch error, retaining its already written status report."""

    def __init__(self, message: str, result: BatchResult):
        super().__init__(message)
        self.result = result
