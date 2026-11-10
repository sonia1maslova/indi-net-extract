"""Explicit, checked result persistence with no analysis side effects."""
from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import sys
import tempfile
import uuid

import nibabel as nib
import numpy as np
import scipy

from ._version import __version__


def json_safe(value):
    """Use JSON null for missing numeric values, never nonstandard NaN tokens."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    # The temporary file is in the same filesystem, so replacement is atomic.
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(json_safe(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_table(path, rows, fieldnames):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint() -> str:
    base = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(base.rglob("*.py")):
        digest.update(path.relative_to(base).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime_record() -> dict:
    return {
        "indinetextract": __version__, "executable": sys.executable,
        "python": sys.version, "numpy": np.__version__,
        "scipy": scipy.__version__, "nibabel": nib.__version__,
        "platform": platform.platform(), "source_sha256": source_fingerprint(),
    }


def checked_subject_id(value) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("subject IDs must be nonempty strings without surrounding whitespace")
    if value in (".", "..", "_batches") or any(ch in value for ch in '<>:"/\\|?*\x00'):
        raise ValueError("subject ID must be a folder name, not a path: " + repr(value))
    return value


def subject_directory(output_root, subject_id, label) -> Path:
    checked_subject_id(subject_id)
    if not isinstance(label, str) or not label or "\\" in label:
        raise ValueError("label must be a nonempty relative directory using '/' separators")
    relative = PurePosixPath(label)
    if relative.is_absolute() or any(part in (".", "..") for part in label.split("/")):
        raise ValueError("label cannot be absolute or contain '.' or '..' path segments")
    if any(ch in label for ch in '<>:"\\|?*\x00'):
        raise ValueError("label contains invalid directory characters")
    root = Path(output_root).expanduser().resolve()
    target = (root / subject_id / Path(*relative.parts)).resolve()
    if not target.is_relative_to(root / subject_id):
        raise ValueError("result directory escapes the subject folder")
    return target


@contextmanager
def result_directory(output_root, subject_id, label, overwrite=False):
    target = subject_directory(output_root, subject_id, label)
    if target.exists() and not overwrite:
        raise FileExistsError("result directory already exists: " + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".indinetextract-", dir=target.parent))
    backup = None
    try:
        yield staging
        if target.exists():
            if not overwrite:
                raise FileExistsError("result directory appeared during computation: " + str(target))
            backup = target.with_name(target.name + ".backup-" + uuid.uuid4().hex)
            target.rename(backup)
        try:
            staging.rename(target)
        except BaseException:
            if backup is not None and not target.exists():
                backup.rename(target)
            raise
    finally:
        # Only remove the private temporary directory created by this call.
        if staging.exists():
            if not staging.resolve().is_relative_to(target.parent.resolve()) or not staging.name.startswith(".indinetextract-"):
                raise RuntimeError("refusing to remove an unexpected temporary directory")
            shutil.rmtree(staging)


def _components(directory, component_ids):
    write_table(directory / "components.tsv", [
        {"volume_index": index, "component_id": name}
        for index, name in enumerate(component_ids)
    ], ["volume_index", "component_id"])


def _finish(directory, kind, result, extra=None):
    files = {p.name: file_sha256(p) for p in directory.iterdir() if p.is_file()}
    payload = {
        "schema_version": 1, "kind": kind, "subject_id": result.subject_id,
        "component_ids": list(result.component_ids), "metadata": result.metadata,
        "runtime": runtime_record(), "files": files,
    }
    if extra:
        payload.update(extra)
    write_json(directory / "metadata.json", payload)


def _image(values, reference, dtype):
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    return nib.Nifti1Image(np.asarray(values, dtype=dtype), reference.affine, header)


def _check_map_bundle(maps, mask, component_ids, *, binary=False):
    from ._validation import check_grid, load_mask
    mask_image, inside = load_mask(mask)
    check_grid(maps, mask_image, "saved maps")
    if maps.ndim != 4 or maps.shape[3] != len(component_ids) or not component_ids:
        raise ValueError("map/component ID dimensions disagree")
    if (any(not isinstance(name, str) or not name.strip() for name in component_ids)
            or len(set(component_ids)) != len(component_ids)):
        raise ValueError("component IDs must be unique nonempty strings")
    if np.issubdtype(maps.get_data_dtype(), np.complexfloating):
        raise ValueError("complex-valued maps are not supported")
    if binary:
        values = np.asarray(maps.dataobj)[inside]
        if not np.isfinite(values).all() or not np.all((values == 0) | (values == 1)):
            raise ValueError("binary maps must contain finite 0/1 values inside the mask")


def save_network(result, output_root, *, label, overwrite=False):
    _check_map_bundle(result.maps, result.mask, result.component_ids)
    timecourses = np.asarray(result.timecourses, dtype=np.float64)
    if timecourses.ndim != 2 or timecourses.shape[1] != len(result.component_ids) or not np.isfinite(timecourses).all():
        raise ValueError("timecourses must be a finite [time, component] matrix")
    with result_directory(output_root, result.subject_id, label, overwrite) as directory:
        nib.save(_image(result.maps.dataobj, result.maps, np.float64), directory / "continuous_components.nii.gz")
        nib.save(_image(result.mask.dataobj, result.mask, np.uint8), directory / "mask.nii.gz")
        np.savetxt(directory / "timecourses.tsv", timecourses, delimiter="\t", fmt="%.17g")
        _components(directory, result.component_ids)
        _finish(directory, "network", result, {"timecourse_shape": list(timecourses.shape)})
    return subject_directory(output_root, result.subject_id, label)


def save_binary(result, output_root, *, label, overwrite=False):
    _check_map_bundle(result.maps, result.mask, result.component_ids, binary=True)
    if result.smoothed_maps is not None:
        _check_map_bundle(result.smoothed_maps, result.mask, result.component_ids)
    with result_directory(output_root, result.subject_id, label, overwrite) as directory:
        nib.save(_image(result.maps.dataobj, result.maps, np.uint8), directory / "binary_maps.nii.gz")
        nib.save(_image(result.mask.dataobj, result.mask, np.uint8), directory / "mask.nii.gz")
        _components(directory, result.component_ids)
        names = list(dict.fromkeys(key for row in result.selection for key in row))
        write_table(directory / "selection.tsv", result.selection, names or ["component_id"])
        if result.smoothed_maps is not None:
            nib.save(_image(result.smoothed_maps.dataobj, result.smoothed_maps, np.float64), directory / "smoothed_components.nii.gz")
        _finish(directory, "binary", result, {"selection": result.selection})
    return subject_directory(output_root, result.subject_id, label)


def save_dice(result, output_root, *, label, overwrite=False):
    k = len(result.component_ids)
    if np.shape(result.matrix) != (k, k):
        raise ValueError("Dice matrix must have shape [component, component]")
    with result_directory(output_root, result.subject_id, label, overwrite) as directory:
        _components(directory, result.component_ids)
        # Positional fields allow any component name, including a name equal to
        # the descriptive first-column heading.
        with (directory / "matrix.tsv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["individual_component"] + list(result.component_ids))
            for i, name in enumerate(result.component_ids):
                writer.writerow([name] + list(result.matrix[i]))
        rows = [{"component_id": name, "dice": float(result.per_component[i]),
                 "individual_count": int(result.individual_counts[i]),
                 "reference_count": int(result.reference_counts[i])}
                for i, name in enumerate(result.component_ids)]
        write_table(directory / "per_component.tsv", rows,
                    ["component_id", "dice", "individual_count", "reference_count"])
        write_json(directory / "summary.json", result.summary)
        _finish(directory, "dice", result)
    return subject_directory(output_root, result.subject_id, label)


def _read_bundle(directory, kind, verify):
    directory = Path(directory)
    with (directory / "metadata.json").open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema_version") != 1 or payload.get("kind") != kind:
        raise ValueError("not a supported indinetextract " + kind + " result")
    checked_subject_id(payload["subject_id"])
    with (directory / "components.tsv").open(encoding="utf-8", newline="") as stream:
        components = list(csv.DictReader(stream, delimiter="\t"))
    if ([row["component_id"] for row in components] != payload["component_ids"]
            or [int(row["volume_index"]) for row in components] != list(range(len(components)))):
        raise ValueError("saved component table differs from metadata")
    if verify:
        for name, expected in payload["files"].items():
            if Path(name).name != name or file_sha256(directory / name) != expected:
                raise ValueError("saved result is incomplete or has changed: " + name)
    return directory, payload


def load_network(directory, *, verify=True):
    from ._types import NetworkResult
    directory, record = _read_bundle(directory, "network", verify)
    maps = nib.load(directory / "continuous_components.nii.gz")
    mask = nib.load(directory / "mask.nii.gz")
    timecourses = np.loadtxt(directory / "timecourses.tsv", delimiter="\t", ndmin=2)
    _check_map_bundle(maps, mask, tuple(record["component_ids"]))
    if not np.isfinite(timecourses).all():
        raise ValueError("saved timecourses contain nonfinite values")
    if list(timecourses.shape) != record["timecourse_shape"] or maps.shape[3] != timecourses.shape[1]:
        raise ValueError("saved network dimensions do not match their metadata")
    return NetworkResult(record["subject_id"], maps, timecourses, mask,
                         tuple(record["component_ids"]), record["metadata"])


def load_binary(directory, *, verify=True):
    from ._types import BinaryMapsResult
    directory, record = _read_bundle(directory, "binary", verify)
    smooth = directory / "smoothed_components.nii.gz"
    result = BinaryMapsResult(record["subject_id"], nib.load(directory / "binary_maps.nii.gz"),
                              nib.load(directory / "mask.nii.gz"), tuple(record["component_ids"]),
                              record["selection"], record["metadata"],
                              nib.load(smooth) if smooth.exists() else None)
    _check_map_bundle(result.maps, result.mask, result.component_ids, binary=True)
    if result.smoothed_maps is not None:
        _check_map_bundle(result.smoothed_maps, result.mask, result.component_ids)
    return result


def load_dice(directory, *, verify=True):
    from ._types import DiceResult
    directory, record = _read_bundle(directory, "dice", verify)
    ids = tuple(record["component_ids"])
    with (directory / "matrix.tsv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))
    if rows[0][1:] != list(ids) or [row[0] for row in rows[1:]] != list(ids):
        raise ValueError("saved Dice matrix component order differs from metadata")
    matrix = np.asarray([[float(value) for value in row[1:]] for row in rows[1:]])
    with (directory / "per_component.tsv").open(encoding="utf-8", newline="") as stream:
        counts = list(csv.DictReader(stream, delimiter="\t"))
    return DiceResult(record["subject_id"], matrix, ids,
                      np.asarray([int(row["individual_count"]) for row in counts]),
                      np.asarray([int(row["reference_count"]) for row in counts]), record["metadata"])
