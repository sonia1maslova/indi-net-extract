"""Persistence and spatial-identity checks through the public result class."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np

from indinetextract import DiceResult, NetworkResult

from tests.helpers import AFFINE, make_fit


class NetworkResultIOTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="indinetextract-io-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.values = np.arange(3 * 4 * 2 * 3, dtype=float).reshape(3, 4, 2, 3)
        self.mask = np.ones((3, 4, 2), dtype=np.uint8)
        self.mask[0, 0, 0] = 0
        self.values[0, 0, 0, :] = 0.0
        self.fit = make_fit(
            self.values, mask=self.mask, component_ids=("visual", "motor", "other")
        )

    def test_round_trip_preserves_values_order_and_affine(self) -> None:
        self.fit.save(self.root, label="review")
        directory = self.root / self.fit.subject_id / "review"
        expected = {
            "continuous_components.nii.gz", "timecourses.tsv", "mask.nii.gz",
            "components.tsv", "metadata.json",
        }
        self.assertTrue(expected.issubset({p.name for p in directory.iterdir()}))
        loaded = NetworkResult.load(directory)
        self.assertEqual(loaded.subject_id, self.fit.subject_id)
        self.assertEqual(loaded.component_ids, self.fit.component_ids)
        np.testing.assert_allclose(loaded.maps.get_fdata(), self.values, atol=1e-10)
        np.testing.assert_allclose(loaded.timecourses, self.fit.timecourses, atol=1e-10)
        np.testing.assert_array_equal(loaded.mask.get_fdata(), self.mask)
        np.testing.assert_allclose(loaded.maps.affine, AFFINE, rtol=0, atol=1e-6)
        np.testing.assert_allclose(loaded.mask.affine, AFFINE, rtol=0, atol=1e-6)
        self.assertEqual(loaded.metadata["fixture"], "synthetic")
        with (directory / "metadata.json").open(encoding="utf-8") as stream:
            self.assertIsInstance(json.load(stream), dict)

    def test_default_save_label_is_extraction(self) -> None:
        self.fit.save(self.root)
        loaded = NetworkResult.load(self.root / self.fit.subject_id / "extraction")
        self.assertEqual(loaded.maps.shape[-1], 3)

    def test_existing_output_is_preserved_without_overwrite(self) -> None:
        self.fit.save(self.root)
        directory = self.root / self.fit.subject_id / "extraction"
        original = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.iterdir() if p.is_file()
        }
        changed = make_fit(self.values + 10.0, mask=self.mask)
        with self.assertRaises((FileExistsError, ValueError)):
            changed.save(self.root)
        after = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.iterdir() if p.is_file()
        }
        self.assertEqual(after, original)

    def test_explicit_overwrite_replaces_result(self) -> None:
        self.fit.save(self.root)
        replacement = self.values * -0.5
        changed = make_fit(replacement, mask=self.mask)
        changed.save(self.root, overwrite=True)
        loaded = NetworkResult.load(self.root / self.fit.subject_id / "extraction")
        np.testing.assert_allclose(loaded.maps.get_fdata(), replacement)

    def test_save_rejects_component_count_mismatch(self) -> None:
        inconsistent = make_fit(self.values, component_ids=("only-one",))
        with self.assertRaises(ValueError):
            inconsistent.save(self.root)

    def test_save_rejects_timecourse_component_mismatch(self) -> None:
        inconsistent = NetworkResult(
            subject_id="bad-timecourse", maps=self.fit.maps,
            timecourses=np.ones((12, 2)), mask=self.fit.mask,
            component_ids=self.fit.component_ids, metadata={},
        )
        with self.assertRaises(ValueError):
            inconsistent.save(self.root)

    def test_load_detects_changed_mask_affine(self) -> None:
        self.fit.save(self.root)
        directory = self.root / self.fit.subject_id / "extraction"
        shifted = AFFINE.copy()
        shifted[0, 3] += 0.5
        nib.save(nib.Nifti1Image(self.mask, shifted), directory / "mask.nii.gz")
        with self.assertRaises(ValueError):
            NetworkResult.load(directory)

    def test_load_incomplete_directory_fails(self) -> None:
        directory = self.root / "incomplete"
        directory.mkdir()
        with self.assertRaises((FileNotFoundError, ValueError)):
            NetworkResult.load(directory)

    def test_save_and_load_reject_inconsistent_map_mask_grids(self) -> None:
        shifted = AFFINE.copy()
        shifted[0, 3] += 0.5
        masks = {
            "affine": nib.Nifti1Image(self.mask, shifted),
            "shape": nib.Nifti1Image(self.mask[:-1, ...], AFFINE),
        }
        for name, mask in masks.items():
            with self.subTest(stage="save", mismatch=name), self.assertRaises(ValueError):
                replace(self.fit, mask=mask).save(self.root / f"invalid-save-{name}")
            destination = self.root / f"invalid-load-{name}"
            self.fit.save(destination)
            directory = destination / self.fit.subject_id / "extraction"
            nib.save(mask, directory / "mask.nii.gz")
            # Disabling hashes must not disable semantic spatial validation.
            with self.subTest(stage="load", mismatch=name), self.assertRaises(ValueError):
                NetworkResult.load(directory, verify=False)

    def test_save_rejects_nonbinary_or_nonfinite_analysis_mask(self) -> None:
        for invalid in (0.5, -1.0, np.nan):
            mask = self.mask.astype(float)
            mask[1, 1, 1] = invalid
            invalid_fit = replace(self.fit, mask=nib.Nifti1Image(mask, AFFINE))
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                invalid_fit.save(self.root / f"invalid-mask-{invalid}")

    def test_dice_round_trip_allows_component_id_equal_to_row_header(self) -> None:
        expected = np.array([[1.0, 0.2], [0.4, 0.7]], dtype=float)
        result = DiceResult(
            subject_id="dice-person", matrix=expected,
            component_ids=("individual_component", "C2"),
            individual_counts=np.array([2, 3]), reference_counts=np.array([2, 4]),
            metadata={"fixture": "reserved_header_regression"},
        )
        result.save(self.root)
        loaded = DiceResult.load(self.root / result.subject_id / "template_dice")
        self.assertEqual(loaded.component_ids, result.component_ids)
        np.testing.assert_allclose(loaded.matrix, expected)
        np.testing.assert_allclose(loaded.per_component, [1.0, 0.7])
        np.testing.assert_array_equal(loaded.individual_counts, [2, 3])
        np.testing.assert_array_equal(loaded.reference_counts, [2, 4])


if __name__ == "__main__":
    unittest.main()
