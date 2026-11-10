"""Optional development check of an unpacked wheel and stock FSL, using synthetic data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np

import indinetextract
from indinetextract import (BinaryMapsResult, DiceResult, NetworkResult, extract_batch,
                      postprocess_maps, template_dice)
from tests.helpers import AFFINE, make_synthetic_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fsl-dir", type=Path)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    dataset = make_synthetic_dataset(args.output_root / "inputs")
    batch = extract_batch(
        dataset.subject_ids, dataset.fmri_paths, dataset.templates, dataset.mask,
        args.output_root / "wheel_gig", method="gig_ica", n_jobs=2, n_threads=1,
    )
    if len(batch.successful) != 2:
        raise AssertionError(batch.records)
    fit = NetworkResult.load(batch.records[0]["output_dir"])
    reference = (dataset.template_data > 0).astype(np.uint8)
    reference_path = args.output_root / "reference_binary.nii.gz"
    nib.save(nib.Nifti1Image(reference, AFFINE), reference_path)
    binary = postprocess_maps(fit, fwhm_mm=2, count_reference=reference_path)
    binary_dir = binary.save(args.output_root / "wheel_gig", label="fwhm_2mm")
    loaded_binary = BinaryMapsResult.load(binary_dir)
    dice = template_dice(loaded_binary, reference_path)
    dice_dir = dice.save(args.output_root / "wheel_gig", label="fwhm_2mm/template_dice")
    restored = DiceResult.load(dice_dir)
    np.testing.assert_array_equal(restored.matrix, dice.matrix)
    report = {
        "executable": sys.executable, "indinetextract_version": indinetextract.__version__,
        "imported_module": indinetextract.__file__, "wheel_parallel_gig_passed": True,
        "binary_and_dice_round_trip_passed": True,
        "n_jobs": 2, "n_threads": 1, "scope": "synthetic inputs only",
        "fsl_cases": [],
    }
    if args.fsl_dir is not None:
        executable = args.fsl_dir / "bin/fsl_glm"
        if not executable.is_file():
            raise FileNotFoundError(executable)
        report["fsl_version"] = (args.fsl_dir / "etc/fslversion").read_text().strip()
        report["fsl_executable_sha256"] = hashlib.sha256(executable.read_bytes()).hexdigest()
        environment = dict(os.environ, FSLDIR=str(args.fsl_dir), FSLOUTPUTTYPE="NIFTI_GZ")
        environment["PATH"] = str(args.fsl_dir / "bin") + os.pathsep + environment.get("PATH", "")
        for kind in ("continuous", "binary"):
            directory = args.output_root / ("fsl_" + kind)
            directory.mkdir()
            template_path = dataset.templates if kind == "continuous" else reference_path
            first = directory / "stage1.txt"
            beta_path = directory / "stage2_beta.nii.gz"
            commands = [
                [str(executable), "-i", str(dataset.fmri_paths[0]), "-d", str(template_path),
                 "-o", str(first), "--demean", "-m", str(dataset.mask)],
                [str(executable), "-i", str(dataset.fmri_paths[0]), "-d", str(first),
                 "-o", str(beta_path), "--demean", "-m", str(dataset.mask), "--des_norm"],
            ]
            for index, command in enumerate(commands):
                completed = subprocess.run(command, env=environment, text=True, capture_output=True, check=True)
                (directory / f"stage{index + 1}.log").write_text(completed.stdout + completed.stderr)
            fitted = extract_batch(
                dataset.subject_ids[:1], dataset.fmri_paths[:1], template_path,
                dataset.mask, directory / "python", method="dual_regression", n_threads=1,
            )
            if len(fitted.successful) != 1:
                raise AssertionError(fitted.records)
            fitted = NetworkResult.load(fitted.records[0]["output_dir"])
            official_maps = nib.load(beta_path).get_fdata()
            official_tc = np.loadtxt(first, ndmin=2)
            official_tc = (official_tc - official_tc.mean(axis=0)) / official_tc.std(axis=0, ddof=1)
            np.testing.assert_allclose(fitted.maps.get_fdata(), official_maps, rtol=3e-4, atol=3e-4)
            np.testing.assert_allclose(fitted.timecourses, official_tc, rtol=1e-4, atol=1e-4)
            report["fsl_cases"].append({
                "template": kind, "passed": True,
                "max_abs_beta_error": float(np.max(np.abs(fitted.maps.get_fdata() - official_maps))),
                "max_abs_normalized_timecourse_error": float(np.max(np.abs(fitted.timecourses - official_tc))),
                "rtol_beta": 3e-4, "atol_beta": 3e-4,
                "rtol_timecourse": 1e-4, "atol_timecourse": 1e-4,
            })
    report["passed"] = True
    (args.output_root / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
