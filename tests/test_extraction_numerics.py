"""Mathematical contract checks; these are not external FSL/MATLAB comparisons."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np

from indinetextract import NetworkResult, extract_batch

from tests.helpers import AFFINE, independent_dual_regression, make_synthetic_dataset


class ExtractionNumericsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="indinetextract-numeric-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = make_synthetic_dataset(self.root / "inputs", n_subjects=1)

    def extract(self, output: str, **kwargs):
        options = {"method": "dual_regression", "n_jobs": 1, "n_threads": 1}
        options.update(kwargs)
        return extract_batch(
            self.dataset.subject_ids, self.dataset.fmri_paths,
            self.dataset.templates, self.dataset.mask, self.root / output,
            **options,
        )

    def assert_success(self, batch) -> NetworkResult:
        self.assertEqual(batch.records[0]["status"], "success", msg=str(batch.records[0]))
        return NetworkResult.load(batch.records[0]["output_dir"])

    def test_dual_regression_matches_independent_full_design_least_squares(self) -> None:
        fitted = self.assert_success(self.extract("dr-reference"))
        expected_tc, expected_maps = independent_dual_regression(
            self.dataset.fmri_data[0], self.dataset.template_data, self.dataset.mask_data
        )
        np.testing.assert_allclose(fitted.timecourses, expected_tc, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(fitted.maps.get_fdata(), expected_maps, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(fitted.timecourses.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(fitted.timecourses.std(axis=0, ddof=1), 1, atol=1e-12)
        self.assertEqual(fitted.component_ids, ("C1", "C2", "C3"))
        np.testing.assert_allclose(fitted.maps.affine, AFFINE, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(fitted.maps.get_fdata()[self.dataset.mask_data == 0], 0)

    def test_dual_regression_beta_preserves_prepared_input_scale(self) -> None:
        original = self.assert_success(self.extract("original-scale"))
        scaled_data = self.dataset.fmri_data[0] * 2.5
        nib.save(nib.Nifti1Image(scaled_data, AFFINE), self.dataset.fmri_paths[0])
        scaled = self.assert_success(self.extract("scaled"))
        # Input voxelwise z-scoring would incorrectly remove this scale change.
        np.testing.assert_allclose(scaled.maps.get_fdata(), original.maps.get_fdata() * 2.5, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(scaled.timecourses, original.timecourses, rtol=1e-9, atol=1e-9)

    def test_gig_ica_produces_finite_template_ordered_non_gaussian_maps(self) -> None:
        fitted = self.assert_success(self.extract("gig", method="gig_ica"))
        mask = self.dataset.mask_data.astype(bool)
        actual = fitted.maps.get_fdata()[mask, :]
        self.assertEqual(actual.shape[1], 3)
        self.assertEqual(fitted.timecourses.shape, (48, 3))
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(np.isfinite(fitted.timecourses).all())
        self.assertTrue((actual.std(axis=0, ddof=1) > 0).all())
        np.testing.assert_allclose(actual.std(axis=0, ddof=1), 1.0, rtol=0, atol=0.01)
        # References are the true Laplace sources. Positive diagonal alignment
        # detects lost signs or component reordering without asserting joint
        # independence or a particular optimum of the nonlinear objective.
        reference = self.dataset.template_data[mask, :]
        correlations = np.corrcoef(actual.T, reference.T)[:3, 3:]
        self.assertTrue((np.diag(correlations) > 0.5).all(), msg=str(correlations))
        self.assertEqual(fitted.component_ids, ("C1", "C2", "C3"))

    def test_gig_ica_same_small_input_is_reproducible(self) -> None:
        first = self.assert_success(self.extract("gig-first", method="gig_ica"))
        second = self.assert_success(self.extract("gig-second", method="gig_ica"))
        np.testing.assert_allclose(second.maps.get_fdata(), first.maps.get_fdata(), rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(second.timecourses, first.timecourses, rtol=1e-9, atol=1e-9)

    def test_fmri_affine_mismatch_becomes_explicit_subject_failure(self) -> None:
        shifted = AFFINE.copy()
        shifted[0, 3] += 0.5
        nib.save(nib.Nifti1Image(self.dataset.fmri_data[0], shifted), self.dataset.fmri_paths[0])
        result = self.extract("wrong-grid")
        self.assertEqual(result.records[0]["status"], "failed")
        self.assertEqual(result.records[0]["error_type"], "ValueError")

    def test_fmri_spatial_shape_mismatch_is_rejected(self) -> None:
        smaller = self.dataset.fmri_data[0][:-1, ...]
        nib.save(nib.Nifti1Image(smaller, AFFINE), self.dataset.fmri_paths[0])
        result = self.extract("wrong-shape")
        self.assertEqual(result.records[0]["status"], "failed")
        self.assertEqual(result.records[0]["error_type"], "ValueError")

    def test_nonbinary_mask_is_rejected_without_thresholding(self) -> None:
        mask = self.dataset.mask_data.astype(float)
        mask[1, 1, 1] = 0.5
        nib.save(nib.Nifti1Image(mask, AFFINE), self.dataset.mask)
        with self.assertRaises(ValueError):
            self.extract("nonbinary-mask")

    def test_three_dimensional_fmri_is_not_silently_interpreted_as_a_run(self) -> None:
        nib.save(nib.Nifti1Image(self.dataset.fmri_data[0][..., 0], AFFINE), self.dataset.fmri_paths[0])
        result = self.extract("3d")
        self.assertEqual(result.records[0]["status"], "failed")
        self.assertEqual(result.records[0]["error_type"], "ValueError")

    def test_nonfinite_value_inside_mask_fails_without_mask_shrinkage(self) -> None:
        data = self.dataset.fmri_data[0].copy()
        location = tuple(np.argwhere(self.dataset.mask_data)[0])
        data[(*location, 0)] = np.nan
        nib.save(nib.Nifti1Image(data, AFFINE), self.dataset.fmri_paths[0])
        result = self.extract("nonfinite")
        self.assertEqual(result.records[0]["status"], "failed")
        self.assertEqual(result.records[0]["error_type"], "ValueError")

    def test_nonfinite_values_outside_mask_do_not_change_the_fit(self) -> None:
        original = self.assert_success(self.extract("masked-original"))
        data = self.dataset.fmri_data[0].copy()
        data[self.dataset.mask_data == 0] = np.nan
        nib.save(nib.Nifti1Image(data, AFFINE), self.dataset.fmri_paths[0])
        fitted = self.assert_success(self.extract("masked-outside"))
        np.testing.assert_allclose(fitted.maps.get_fdata(), original.maps.get_fdata(), rtol=1e-9, atol=1e-9)

    def test_constant_masked_voxel_is_rejected_without_dropping_it(self) -> None:
        data = self.dataset.fmri_data[0].copy()
        location = tuple(np.argwhere(self.dataset.mask_data)[0])
        data[location] = 5.0
        nib.save(nib.Nifti1Image(data, AFFINE), self.dataset.fmri_paths[0])
        result = self.extract("constant-voxel")
        self.assertEqual(result.records[0]["status"], "failed")
        self.assertEqual(result.records[0]["error_type"], "ValueError")

    def test_dependent_templates_are_rejected(self) -> None:
        templates = self.dataset.template_data.copy()
        templates[..., 1] = templates[..., 0]
        nib.save(nib.Nifti1Image(templates, AFFINE), self.dataset.templates)
        with self.assertRaises(ValueError):
            self.extract("dependent-templates")


if __name__ == "__main__":
    unittest.main()
