"""Signed Top-N, mask-normalized smoothing, Dice and overlap checks."""

from __future__ import annotations

import unittest

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

from indinetextract import BinaryMapsResult, overlap_fraction, postprocess_maps, template_dice

from tests.helpers import AFFINE, make_fit


def binary_result(values: np.ndarray, mask: np.ndarray | None = None) -> BinaryMapsResult:
    values = np.asarray(values, dtype=np.uint8)
    if mask is None:
        mask = np.ones(values.shape[:3], dtype=np.uint8)
    return BinaryMapsResult(
        subject_id="synthetic-person", maps=nib.Nifti1Image(values, AFFINE),
        mask=nib.Nifti1Image(np.asarray(mask, dtype=np.uint8), AFFINE),
        component_ids=tuple(f"C{i + 1}" for i in range(values.shape[-1])),
        selection=[], metadata={},
    )


class TopNTests(unittest.TestCase):
    def test_signed_ranking_includes_zero_and_negative_values(self) -> None:
        raw = np.array([4.0, -1.0, 0.0, -2.0, 3.0, -9.0]).reshape(3, 2, 1, 1)
        result = postprocess_maps(make_fit(raw), top_n=4)
        actual = result.maps.get_fdata().reshape(-1)
        np.testing.assert_array_equal(actual, [1, 1, 1, 0, 1, 0])
        self.assertEqual(int(actual.sum()), 4)
        audit = result.selection[0]
        self.assertEqual(audit["n_requested"], 4)
        self.assertEqual(audit["n_selected"], 4)
        self.assertEqual(audit["finite_candidates"], 6)
        self.assertEqual(audit["selected_positive"], 2)
        self.assertEqual(audit["selected_zero"], 1)
        self.assertEqual(audit["selected_negative"], 1)
        self.assertEqual(audit["cutoff_value"], -1.0)

    def test_ties_follow_fortran_linear_order(self) -> None:
        raw = np.full((2, 3, 2, 1), 7.0)
        result = postprocess_maps(make_fit(raw), top_n=3)
        expected = np.zeros((2, 3, 2), dtype=np.uint8)
        expected[0, 0, 0] = 1
        expected[1, 0, 0] = 1
        expected[0, 1, 0] = 1
        np.testing.assert_array_equal(result.maps.get_fdata()[..., 0], expected)

    def test_float64_near_ties_are_not_rounded_before_ranking(self) -> None:
        raw = np.array([1.0, 1.0 + 2.0**-40, 0.0, -1.0]).reshape(2, 2, 1, 1)
        result = postprocess_maps(make_fit(raw), top_n=1)
        np.testing.assert_array_equal(result.maps.get_fdata().reshape(-1), [0, 1, 0, 0])

    def test_only_finite_candidates_are_ranked(self) -> None:
        raw = np.array([np.nan, np.inf, -np.inf, -2.0, 0.0, 1.0]).reshape(3, 2, 1, 1)
        result = postprocess_maps(make_fit(raw), top_n=3)
        np.testing.assert_array_equal(
            result.maps.get_fdata().reshape(-1), [0, 0, 0, 1, 1, 1]
        )

    def test_too_few_finite_candidates_raise_instead_of_short_selection(self) -> None:
        raw = np.array([np.nan, -3.0, -2.0, -1.0]).reshape(2, 2, 1, 1)
        with self.assertRaises(ValueError):
            postprocess_maps(make_fit(raw), top_n=4)

    def test_per_component_counts_with_k_not_eleven(self) -> None:
        raw = np.arange(6 * 3, dtype=float).reshape(3, 2, 1, 3)
        result = postprocess_maps(make_fit(raw), top_n=[1, 3, 5])
        np.testing.assert_array_equal(result.maps.get_fdata().sum(axis=(0, 1, 2)), [1, 3, 5])
        self.assertEqual(result.component_ids, ("C1", "C2", "C3"))

    def test_invalid_or_ambiguous_count_arguments_are_rejected(self) -> None:
        fit = make_fit(np.arange(12, dtype=float).reshape(3, 2, 1, 2))
        for invalid in (0, -1, 1.5, True, [1], [1, 8]):
            with self.subTest(top_n=invalid), self.assertRaises(ValueError):
                postprocess_maps(fit, top_n=invalid)
        reference = nib.Nifti1Image(np.ones(fit.maps.shape, dtype=np.uint8), AFFINE)
        with self.assertRaises(ValueError):
            postprocess_maps(fit, top_n=1, count_reference=reference)

    def test_reference_counts_use_only_the_shared_mask(self) -> None:
        raw = np.arange(12, dtype=float).reshape(3, 2, 1, 2)
        mask = np.ones((3, 2, 1), dtype=np.uint8)
        mask[0, 0, 0] = 0
        reference = np.zeros(raw.shape, dtype=np.uint8)
        reference[0, 0, 0, :] = 1  # Outside the mask and excluded from both N values.
        reference[1, 0, 0, 0] = 1
        reference[1:, :, :, 1] = 1
        fit = make_fit(raw, mask=mask)
        result = postprocess_maps(
            fit, count_reference=nib.Nifti1Image(reference, AFFINE)
        )
        np.testing.assert_array_equal(result.maps.get_fdata().sum(axis=(0, 1, 2)), [1, 4])
        np.testing.assert_array_equal(result.maps.get_fdata()[0, 0, 0, :], [0, 0])

    def test_count_reference_rejects_nonbinary_values_inside_mask(self) -> None:
        fit = make_fit(np.arange(8, dtype=float).reshape(2, 2, 1, 2))
        for invalid in (0.5, -1.0, np.nan):
            reference = np.ones(fit.maps.shape, dtype=float)
            reference[0, 0, 0, 0] = invalid
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                postprocess_maps(fit, count_reference=nib.Nifti1Image(reference, AFFINE))

    def test_count_reference_rejects_affine_mismatch(self) -> None:
        fit = make_fit(np.arange(8, dtype=float).reshape(2, 2, 1, 2))
        affine = AFFINE.copy()
        affine[2, 3] += 1.0
        reference = nib.Nifti1Image(np.ones(fit.maps.shape, dtype=np.uint8), affine)
        with self.assertRaises(ValueError):
            postprocess_maps(fit, count_reference=reference)

    def test_zero_reference_support_is_rejected_for_strict_top_n(self) -> None:
        fit = make_fit(np.arange(8, dtype=float).reshape(2, 2, 1, 2))
        reference = np.ones(fit.maps.shape, dtype=np.uint8)
        reference[..., 1] = 0
        with self.assertRaises(ValueError):
            postprocess_maps(fit, count_reference=nib.Nifti1Image(reference, AFFINE))


class SmoothingTests(unittest.TestCase):
    def test_mask_normalization_preserves_a_constant_at_mask_edges(self) -> None:
        mask = np.zeros((7, 7, 7), dtype=np.uint8)
        mask[1:6, 1:6, 1:6] = 1
        mask[3, 3, 3] = 0
        raw = np.full((*mask.shape, 1), 999.0)
        raw[mask.astype(bool), 0] = 4.0
        result = postprocess_maps(make_fit(raw, mask=mask), fwhm_mm=6.0, top_n=3)
        smoothed = result.smoothed_maps.get_fdata()[..., 0]
        np.testing.assert_allclose(smoothed[mask.astype(bool)], 4.0, rtol=0, atol=1e-10)
        np.testing.assert_array_equal(smoothed[~mask.astype(bool)], 0.0)

    def test_smoothing_is_computed_from_continuous_maps_before_top_n(self) -> None:
        mask = np.ones((5, 4, 3), dtype=np.uint8)
        mask[0, 0, 0] = 0
        raw = np.zeros((*mask.shape, 1), dtype=float)
        raw[1, 1, 1, 0] = 10.0
        raw[3, 2, 1, 0] = -6.0
        raw[4, 3, 2, 0] = 3.0
        fit = make_fit(raw, mask=mask)
        fwhm = 4.0
        sigma = fwhm / np.sqrt(8.0 * np.log(2.0)) / np.array([2.0, 2.5, 3.0])
        support = mask.astype(bool)
        denominator = gaussian_filter(mask.astype(float), sigma, mode="constant", cval=0.0, truncate=4.0)
        numerator = gaussian_filter(raw[..., 0] * mask, sigma, mode="constant", cval=0.0, truncate=4.0)
        expected = np.zeros(mask.shape, dtype=float)
        expected[support] = numerator[support] / denominator[support]
        first = postprocess_maps(fit, fwhm_mm=fwhm, top_n=2)
        second = postprocess_maps(fit, fwhm_mm=fwhm, top_n=7)
        np.testing.assert_allclose(first.smoothed_maps.get_fdata()[..., 0], expected, atol=1e-10)
        np.testing.assert_allclose(second.smoothed_maps.get_fdata()[..., 0], expected, atol=1e-10)
        np.testing.assert_array_equal(fit.maps.get_fdata(), raw)
        self.assertEqual(int(first.maps.get_fdata().sum()), 2)
        self.assertEqual(int(second.maps.get_fdata().sum()), 7)

    def test_zero_fwhm_retains_raw_signed_values_inside_mask(self) -> None:
        raw = np.array([8.0, -2.0, 0.0, -4.0]).reshape(2, 2, 1, 1)
        mask = np.ones((2, 2, 1), dtype=np.uint8)
        mask[0, 0, 0] = 0
        result = postprocess_maps(make_fit(raw, mask=mask), fwhm_mm=0, top_n=2)
        expected = raw.copy()
        expected[0, 0, 0, 0] = 0.0
        np.testing.assert_array_equal(result.smoothed_maps.get_fdata(), expected)

    def test_positive_fwhm_rejects_nonfinite_values_inside_mask(self) -> None:
        raw = np.arange(8, dtype=float).reshape(2, 2, 2, 1)
        raw[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            postprocess_maps(make_fit(raw), fwhm_mm=2.0, top_n=2)

    def test_smoothing_ignores_outside_nan_and_never_mixes_components(self) -> None:
        mask = np.zeros((5, 5, 5), dtype=np.uint8)
        mask[1:4, 1:4, 1:4] = 1
        raw = np.full((*mask.shape, 2), np.nan)
        raw[mask.astype(bool), :] = [2.0, -7.0]
        result = postprocess_maps(make_fit(raw, mask=mask), fwhm_mm=4.0, top_n=2)
        smoothed = result.smoothed_maps.get_fdata()
        np.testing.assert_allclose(smoothed[mask.astype(bool), 0], 2.0, atol=1e-10)
        np.testing.assert_allclose(smoothed[mask.astype(bool), 1], -7.0, atol=1e-10)
        np.testing.assert_array_equal(smoothed[~mask.astype(bool), :], 0.0)

    def test_positive_fwhm_rejects_sheared_voxel_axes(self) -> None:
        raw = np.arange(27, dtype=float).reshape(3, 3, 3, 1)
        affine = AFFINE.copy()
        affine[0, 1] = 0.3
        with self.assertRaises(ValueError):
            postprocess_maps(make_fit(raw, affine=affine), fwhm_mm=2.0, top_n=2)


class DiceAndOverlapTests(unittest.TestCase):
    def test_full_matrix_keeps_component_order_without_matching(self) -> None:
        individual = np.zeros((4, 1, 1, 2), dtype=np.uint8)
        individual[[0, 1], 0, 0, 0] = 1
        individual[2, 0, 0, 1] = 1
        reference = np.zeros_like(individual)
        reference[[1, 2], 0, 0, 0] = 1
        reference[[0, 1], 0, 0, 1] = 1
        result = template_dice(binary_result(individual), nib.Nifti1Image(reference, AFFINE))
        expected = np.array([[0.5, 1.0], [2.0 / 3.0, 0.0]])
        np.testing.assert_allclose(result.matrix, expected)
        np.testing.assert_allclose(result.per_component, [0.5, 0.0])
        self.assertAlmostEqual(result.mean_dice, 0.25)
        np.testing.assert_array_equal(result.individual_counts, [2, 1])
        np.testing.assert_array_equal(result.reference_counts, [2, 2])

    def test_empty_sets_have_explicit_nan_and_zero_policy(self) -> None:
        individual = np.zeros((3, 1, 1, 2), dtype=np.uint8)
        reference = np.zeros_like(individual)
        reference[1, 0, 0, 1] = 1
        result = template_dice(binary_result(individual), nib.Nifti1Image(reference, AFFINE))
        self.assertTrue(np.isnan(result.matrix[0, 0]))
        self.assertEqual(result.matrix[0, 1], 0.0)
        self.assertEqual(result.matrix[1, 1], 0.0)
        self.assertTrue(np.isnan(result.mean_dice))

    def test_dice_and_overlap_use_mask_and_all_k_components(self) -> None:
        mask = np.ones((3, 1, 1), dtype=np.uint8)
        mask[2, 0, 0] = 0
        individual = np.zeros((3, 1, 1, 3), dtype=np.uint8)
        individual[0, 0, 0, :] = [1, 1, 0]
        individual[1, 0, 0, :] = [0, 1, 0]
        individual[2, 0, 0, :] = [1, 1, 1]
        result = binary_result(individual, mask=mask)
        overlap = overlap_fraction(result)
        self.assertEqual(overlap.ndim, 3)
        np.testing.assert_allclose(overlap.get_fdata().reshape(-1), [2 / 3, 1 / 3, 0])
        np.testing.assert_allclose(overlap.affine, AFFINE, atol=1e-6)
        dice = template_dice(result, nib.Nifti1Image(individual, AFFINE))
        np.testing.assert_array_equal(dice.individual_counts, [1, 2, 0])
        np.testing.assert_array_equal(dice.reference_counts, [1, 2, 0])

    def test_dice_rejects_same_shape_with_different_affine(self) -> None:
        values = np.ones((3, 1, 1, 2), dtype=np.uint8)
        affine = AFFINE.copy()
        affine[0, 3] += 0.5
        with self.assertRaises(ValueError):
            template_dice(binary_result(values), nib.Nifti1Image(values, affine))


if __name__ == "__main__":
    unittest.main()
