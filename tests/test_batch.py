"""Subject association, process execution, failure isolation and output safety."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from indinetextract import BatchExecutionError, NetworkResult, extract_batch

from tests.helpers import independent_dual_regression, make_synthetic_dataset


class BatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="indinetextract-batch-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = make_synthetic_dataset(self.root / "inputs")

    def extract(self, output: str, **kwargs):
        options = {"method": "dual_regression", "n_jobs": 1, "n_threads": 1}
        options.update(kwargs)
        return extract_batch(
            self.dataset.subject_ids, self.dataset.fmri_paths,
            self.dataset.templates, self.dataset.mask, self.root / output,
            component_ids=self.dataset.component_ids, **options,
        )

    def test_input_order_and_id_path_association_survive_round_trip(self) -> None:
        batch = self.extract("ordered", label="audit")
        self.assertIs(batch.summary, batch.records)
        self.assertEqual([r["subject_id"] for r in batch.records], self.dataset.subject_ids)
        self.assertEqual([r["status"] for r in batch.records], ["success", "success"])
        self.assertTrue(Path(batch.report_dir).is_dir())
        self.assertTrue(batch.batch_id)
        for index, row in enumerate(batch.records):
            self.assertEqual(Path(row["fmri_path"]).resolve(), self.dataset.fmri_paths[index].resolve())
            expected_directory = self.root / "ordered" / self.dataset.subject_ids[index] / "audit"
            self.assertEqual(Path(row["output_dir"]).resolve(), expected_directory.resolve())
            result = NetworkResult.load(expected_directory)
            expected_tc, expected_maps = independent_dual_regression(
                self.dataset.fmri_data[index], self.dataset.template_data,
                self.dataset.mask_data,
            )
            np.testing.assert_allclose(result.timecourses, expected_tc, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(result.maps.get_fdata(), expected_maps, rtol=1e-9, atol=1e-9)
            self.assertEqual(result.subject_id, self.dataset.subject_ids[index])
            self.assertEqual(result.component_ids, self.dataset.component_ids)
            self.assertGreaterEqual(row["elapsed_seconds"], 0)

    def test_duplicate_subject_ids_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            extract_batch(
                ["same-person", "same-person"], self.dataset.fmri_paths,
                self.dataset.templates, self.dataset.mask, self.root / "duplicate",
                method="dual_regression", n_threads=1,
            )

    def test_id_and_path_length_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            extract_batch(
                self.dataset.subject_ids[:1], self.dataset.fmri_paths,
                self.dataset.templates, self.dataset.mask, self.root / "mismatch",
                method="dual_regression", n_threads=1,
            )

    def test_one_missing_file_does_not_abort_default_batch(self) -> None:
        ids = [self.dataset.subject_ids[0], "missing-person", self.dataset.subject_ids[1]]
        paths = [self.dataset.fmri_paths[0], self.root / "absent.nii.gz", self.dataset.fmri_paths[1]]
        batch = extract_batch(
            ids, paths, self.dataset.templates, self.dataset.mask,
            self.root / "continue", method="dual_regression", n_jobs=1, n_threads=1,
        )
        self.assertEqual([r["subject_id"] for r in batch.records], ids)
        self.assertEqual([r["status"] for r in batch.records], ["success", "failed", "success"])
        self.assertTrue(batch.records[1]["error_type"])
        self.assertTrue(batch.records[1]["error"])
        self.assertTrue(Path(batch.report_dir).is_dir())
        for index in (0, 2):
            self.assertTrue(Path(batch.records[index]["output_dir"]).is_dir())

    def test_raise_policy_retains_failure_report_and_cancelled_rows(self) -> None:
        ids = ["missing-first", *self.dataset.subject_ids]
        paths = [self.root / "absent.nii.gz", *self.dataset.fmri_paths]
        with self.assertRaises(BatchExecutionError) as caught:
            extract_batch(
                ids, paths, self.dataset.templates, self.dataset.mask,
                self.root / "raise", method="dual_regression", n_jobs=1,
                n_threads=1, on_error="raise",
            )
        batch = caught.exception.result
        self.assertEqual([r["subject_id"] for r in batch.records], ids)
        self.assertEqual([r["status"] for r in batch.records], ["failed", "cancelled", "cancelled"])
        self.assertTrue(Path(batch.report_dir).is_dir())

    def test_existing_extraction_is_not_overwritten_by_later_batch(self) -> None:
        first = self.extract("protected")
        directory = Path(first.records[0]["output_dir"])
        original = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.iterdir() if p.is_file()
        }
        second = self.extract("protected")
        self.assertEqual([r["status"] for r in second.records], ["failed", "failed"])
        current = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.iterdir() if p.is_file()
        }
        self.assertEqual(current, original)
        self.assertNotEqual(first.batch_id, second.batch_id)
        self.assertNotEqual(Path(first.report_dir), Path(second.report_dir))

    def test_two_processes_one_thread_match_serial_values_and_order(self) -> None:
        serial = self.extract("serial", n_jobs=1, n_threads=1)
        parallel = self.extract("parallel", n_jobs=2, n_threads=1)
        self.assertEqual(parallel.metadata["n_jobs"], 2)
        self.assertEqual(parallel.metadata["n_threads"], 1)
        self.assertEqual([r["subject_id"] for r in parallel.records], self.dataset.subject_ids)
        self.assertEqual([r["status"] for r in parallel.records], ["success", "success"])
        for before, after in zip(serial.records, parallel.records):
            self.assertNotEqual(after["worker_pid"], os.getpid())
            serial_fit = NetworkResult.load(before["output_dir"])
            parallel_fit = NetworkResult.load(after["output_dir"])
            np.testing.assert_allclose(parallel_fit.maps.get_fdata(), serial_fit.maps.get_fdata(), rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(parallel_fit.timecourses, serial_fit.timecourses, rtol=1e-9, atol=1e-9)
            self.assertEqual(parallel_fit.metadata["n_threads"], 1)

    def test_default_thread_request_is_recorded(self) -> None:
        batch = extract_batch(
            self.dataset.subject_ids[:1], self.dataset.fmri_paths[:1],
            self.dataset.templates, self.dataset.mask, self.root / "defaults",
            method="dual_regression",
        )
        self.assertEqual(batch.metadata["n_jobs"], 1)
        self.assertEqual(batch.metadata["n_threads"], 2)
        self.assertEqual(batch.records[0]["status"], "success")
        fitted = NetworkResult.load(batch.records[0]["output_dir"])
        self.assertEqual(fitted.metadata["n_threads"], 2)

    def test_nonpositive_or_noninteger_worker_configuration_is_rejected(self) -> None:
        for option in ("n_jobs", "n_threads"):
            for invalid in (0, -1, 1.5, True):
                with self.subTest(option=option, value=invalid), self.assertRaises(ValueError):
                    self.extract(f"invalid-{option}-{invalid}", **{option: invalid})

    def test_unknown_method_is_rejected_before_execution(self) -> None:
        with self.assertRaises(ValueError):
            self.extract("unsupported", method="drbeta")


if __name__ == "__main__":
    unittest.main()
