"""Batch orchestration: pass file paths to workers, save per person, keep order."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import multiprocessing
from numbers import Integral
import os
from pathlib import Path
import time
import uuid

from threadpoolctl import threadpool_limits

from ._io import (checked_subject_id, file_sha256, runtime_record,
                  subject_directory, write_json, write_table)
from ._types import BatchExecutionError, BatchResult


_COLUMNS = ["subject_id", "fmri_path", "status", "output_dir", "elapsed_seconds",
            "worker_pid", "error_type", "error"]
_WORKER_THREAD_LIMITER = None


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _init_worker(n_threads):
    global _WORKER_THREAD_LIMITER
    # Only worker processes are affected; importing indinetextract never changes
    # thread settings in the caller. Each extraction also has its own context.
    _WORKER_THREAD_LIMITER = threadpool_limits(limits=n_threads)


def _run_subject(job):
    from .extraction import _extract_one
    started = time.perf_counter()
    record = {
        "subject_id": job["subject_id"], "fmri_path": job["fmri"],
        "status": "failed", "output_dir": job["output_dir"],
        "elapsed_seconds": 0.0, "worker_pid": os.getpid(),
        "error_type": "", "error": "",
    }
    try:
        if Path(job["output_dir"]).exists():
            raise FileExistsError("result directory already exists: " + job["output_dir"])
        fit = _extract_one(
            job["fmri"], job["templates"], job["mask"],
            subject_id=job["subject_id"], method=job["method"],
            component_ids=job["component_ids"], n_threads=job["n_threads"],
            work_dir=job["work_dir"],
        )
        fit.save(job["output_root"], label=job["label"])
        record["status"] = "success"
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error))
    record["elapsed_seconds"] = time.perf_counter() - started
    return record


def _checkpoint(result, state):
    write_table(result.report_dir / "summary.tsv", result.records, _COLUMNS)
    record = dict(result.metadata)
    record.update({
        "state": state,
        "n_success": len(result.successful), "n_failed": len(result.failed),
        "n_cancelled": sum(row["status"] == "cancelled" for row in result.records),
    })
    write_json(result.report_dir / "metadata.json", record)


def extract_batch(
    subject_ids, fmri_paths, templates, mask, output_root, *,
    method, component_ids=None, label="extraction", n_jobs=1, n_threads=2,
    work_dir=None, on_error="continue",
) -> BatchResult:
    """Extract networks using shared settings, with one output folder per ID.

    ``n_jobs`` is the maximum number of simultaneous participants.
    ``n_threads`` limits numerical threads within each participant.
    Every fmri path denotes ONE already prepared 4D image, never a run list.
    Parallel execution uses fresh processes (spawn); use a main guard in scripts.
    """
    from ._validation import check_grid, load_image, load_mask
    from .extraction import _check_template_design

    if isinstance(subject_ids, (str, bytes)) or isinstance(fmri_paths, (str, bytes, os.PathLike)):
        raise ValueError("subject_ids and fmri_paths must be lists, not single strings")
    ids, paths = list(subject_ids), list(fmri_paths)
    if not ids or len(ids) != len(paths):
        raise ValueError("subject_ids and fmri_paths must be nonempty and have equal lengths")
    ids = [checked_subject_id(value) for value in ids]
    if len(set(ids)) != len(ids):
        raise ValueError("subject IDs must be unique within one batch")
    n_jobs = _positive_integer(n_jobs, "n_jobs")
    n_threads = _positive_integer(n_threads, "n_threads")
    if method not in ("dual_regression", "gig_ica"):
        raise ValueError("method must be 'dual_regression' or 'gig_ica'")
    if on_error not in ("continue", "raise"):
        raise ValueError("on_error must be 'continue' or 'raise'")
    templates = Path(templates).expanduser().resolve()
    mask = Path(mask).expanduser().resolve()
    paths = [str(Path(path).expanduser().resolve()) for path in paths]
    root = Path(output_root).expanduser().resolve()
    # Validate shared images once before creating output. Per-person image
    # failures are instead recorded individually according to on_error.
    mask_img, mask_data = load_mask(mask)
    reference = load_image(templates)
    check_grid(reference, mask_img, "templates")
    if reference.ndim != 4 or reference.shape[3] < 1:
        raise ValueError("templates must be a 4D image with one or more components")
    import numpy as np
    template_values = np.asarray(reference.dataobj, dtype=np.float64)[mask_data]
    if not np.isfinite(template_values).all():
        raise ValueError("templates must be finite inside the mask")
    with threadpool_limits(limits=n_threads):
        template_rank = _check_template_design(template_values)
    k = reference.shape[3]
    del template_values, reference, mask_img, mask_data
    if component_ids is None:
        component_ids = tuple("C" + str(i + 1) for i in range(k))
    else:
        if isinstance(component_ids, str):
            raise ValueError("component_ids must be a sequence, not a string")
        component_ids = tuple(component_ids)
        if (len(component_ids) != k or len(set(component_ids)) != k
                or any(not isinstance(name, str) or not name.strip() for name in component_ids)):
            raise ValueError("component_ids must have K unique nonempty string entries")
    if work_dir is not None:
        work_dir = str(Path(work_dir).expanduser().resolve())
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    outputs = [str(subject_directory(root, sid, label)) for sid in ids]
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    report_dir = root / "_batches" / batch_id
    report_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "batch_id": batch_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": method, "templates": str(templates), "mask": str(mask),
        "templates_sha256": file_sha256(templates), "mask_sha256": file_sha256(mask),
        "template_rank": template_rank,
        "component_ids": component_ids, "label": label, "n_subjects": len(ids),
        "n_jobs": n_jobs, "effective_workers": min(n_jobs, len(ids)),
        "n_threads": n_threads, "on_error": on_error, "runtime": runtime_record(),
        "input_preparation": "caller_prepared_single_4d_image; no run concatenation or normalization",
    }
    records = [{"subject_id": sid, "fmri_path": path, "output_dir": output,
                "status": "pending", "elapsed_seconds": 0.0, "worker_pid": "",
                "error_type": "", "error": ""}
               for sid, path, output in zip(ids, paths, outputs)]
    result = BatchResult(batch_id, records, report_dir, metadata)
    write_table(report_dir / "inputs.tsv", records, ["subject_id", "fmri_path", "output_dir"])
    _checkpoint(result, "running")
    jobs = [dict(fmri=path, subject_id=sid, output_dir=output, output_root=str(root),
                 templates=str(templates), mask=str(mask), method=method,
                 component_ids=component_ids, n_threads=n_threads,
                 label=label, work_dir=work_dir)
            for sid, path, output in zip(ids, paths, outputs)]
    stop = False
    try:
        if n_jobs == 1:
            for index, job in enumerate(jobs):
                records[index] = _run_subject(job)
                _checkpoint(result, "running")
                if records[index]["status"] == "failed" and on_error == "raise":
                    stop = True
                    break
        else:
            # Submit at most n_jobs tasks at once. Only paths/settings cross the
            # process seam; large fMRI arrays stay inside their own worker.
            with ProcessPoolExecutor(
                max_workers=min(n_jobs, len(jobs)),
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_worker, initargs=(n_threads,),
            ) as executor:
                pending = {}
                next_index = 0
                while next_index < min(n_jobs, len(jobs)):
                    pending[executor.submit(_run_subject, jobs[next_index])] = next_index
                    next_index += 1
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        index = pending.pop(future)
                        try:
                            records[index] = future.result()
                        except Exception as error:
                            records[index].update(status="failed", error_type=type(error).__name__, error=str(error))
                        if records[index]["status"] == "failed" and on_error == "raise":
                            stop = True
                    if stop:
                        for future, index in list(pending.items()):
                            if future.cancel():
                                records[index]["status"] = "cancelled"
                                pending.pop(future)
                    else:
                        while next_index < len(jobs) and len(pending) < n_jobs:
                            pending[executor.submit(_run_subject, jobs[next_index])] = next_index
                            next_index += 1
                    _checkpoint(result, "running")
        for record in records:
            if record["status"] == "pending":
                record.update(status="cancelled", error="not started after an earlier failure")
        metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["state"] = "failed" if result.failed else "complete"
        _checkpoint(result, metadata["state"])
    except BaseException as error:
        for record in records:
            if record["status"] == "pending":
                record.update(status="cancelled", error="batch interrupted")
        metadata.update(state="interrupted", error_type=type(error).__name__, error=str(error))
        _checkpoint(result, "interrupted")
        raise
    if stop:
        raise BatchExecutionError("batch stopped after a participant failed; see " + str(report_dir), result)
    return result
