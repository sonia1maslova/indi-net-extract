# Copyright (c) 2012 Yuhui DU and Yong FAN
# All rights reserved.
#
# Redistribution and use in source or any other forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in any other form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the names of the copyright holders nor the names of future
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""NumPy/SciPy translation of the locked official classic GIG-ICA core.

Source: GIFT ``icatb_gigicar.m``, SHA256 given by SOURCE_SHA256 below.
Inputs are data [time, voxel] and reference [component, voxel]. Outputs are
components [voxel, component] and timecourses [time, component], both float64.
The Python solver calls no MATLAB process or MATLAB Engine.

The official threshold/fallback, sample-standard-deviation convention, and
iteration control flow are retained, including its final-candidate assignment.
This is a translation requiring independent validation, not author software.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable

import numpy as np
from scipy import linalg


SOURCE_SHA256 = '6a0bbf665524d0122686ce6416d06350cb21d71f90460c282aa539f76803773c'
SOURCE_NAME = 'GIFT icatb_gigicar.m'
GAUSSIAN_EXPECTATION = 0.3745672075
FIRST_THRESHOLD = float(np.finfo(np.float64).eps)
FALLBACK_THRESHOLD = 0.02
UNIT_SD_TOLERANCE = 1e-2
MAX_ATTEMPTS = 100
WEIGHT_CHANGE_SQUARED_TOLERANCE = 1e-5
Callback = Callable[[dict], None] | None


def _emit(callback: Callback, stage: str, **values) -> None:
    if callback is not None:
        callback(dict(stage=stage, **values))


def _nege(values: np.ndarray) -> float:
    # Preserve the literal author expression. A stable alternative would be a
    # separately validated numerical change, not silently substituted here.
    with np.errstate(over='raise', invalid='raise'):
        expectation = float(np.mean(np.log(np.cosh(values))))
    return (expectation - GAUSSIAN_EXPECTATION) ** 2


@dataclass
class WhitenedData:
    y: np.ndarray
    threshold: float
    eigenvalues: np.ndarray
    metadata: dict
    pinv_cache: dict = field(default_factory=dict, repr=False)


@dataclass
class PreparedData:
    centered: np.ndarray
    spatial_means: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    metadata: dict
    whitened_cache: dict = field(default_factory=dict, repr=False)

    def clear_cache(self) -> None:
        """Release cached whitened maps and SVD factors, retaining input prep."""
        self.whitened_cache.clear()


def prepare(data, *, eigen_driver: str = 'evd', callback: Callback = None) -> PreparedData:
    """Copy to float64, spatially demean each timepoint, then diagonalize XX'/V.

    The caller's float32 memmap or ndarray is never modified. Negative
    eigenvalues are not clipped; whiten() implements the author's exact rule.
    """
    started = time.perf_counter()
    if np.iscomplexobj(data):
        raise ValueError('data must be real-valued')
    x = np.array(data, dtype=np.float64, order='C', copy=True)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError('data must be a finite real [time, voxel] matrix')
    _emit(callback, 'spatial_demean', shape=list(x.shape))
    spatial_means = x.mean(axis=1)
    x -= spatial_means[:, None]
    covariance = (x @ x.T) / x.shape[1]
    _emit(callback, 'covariance_eigh_start', driver=eigen_driver)
    eigenvalues, eigenvectors = linalg.eigh(
        covariance, lower=False, overwrite_a=True, check_finite=False,
        driver=eigen_driver)
    metadata = dict(
        n_timepoints=x.shape[0], n_voxels=x.shape[1], dtype='float64',
        eigen_driver=eigen_driver, covariance_divisor=x.shape[1],
        negative_eigenvalue_count=int(np.sum(eigenvalues < 0)),
        smallest_eigenvalues=eigenvalues[:min(16, len(eigenvalues))].tolist(),
        largest_eigenvalue=float(eigenvalues[-1]),
        elapsed_seconds=time.perf_counter() - started)
    _emit(callback, 'covariance_eigh_complete', **metadata)
    return PreparedData(x, spatial_means, eigenvalues, eigenvectors, metadata)


def whiten(prepared: PreparedData, threshold: float, *, callback: Callback = None,
           cache: bool = True) -> WhitenedData:
    """Apply the original absolute eigenvalue cutoff and conditional Y/SD(Y)."""
    threshold = float(threshold)
    if threshold in prepared.whitened_cache:
        return prepared.whitened_cache[threshold]
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('threshold must be finite and nonnegative')
    # Source removes negative eigenvalues, sorts the rest, then uses > thr.
    # For nonnegative thr the resulting retained set equals this mask.
    selected = np.flatnonzero((prepared.eigenvalues >= 0)
                              & (np.abs(prepared.eigenvalues) > threshold))[::-1]
    if selected.size == 0:
        raise ValueError('The official eigenvalue rule retained no dimensions')
    values = np.abs(prepared.eigenvalues[selected])
    whitener = prepared.eigenvectors[:, selected].T / np.sqrt(values)[:, None]
    _emit(callback, 'whitening_start', threshold=threshold, rank=int(selected.size))
    y = whitener @ prepared.centered
    row_sd_before = y.std(axis=1, ddof=1)
    rescaled = threshold < 1e-10 and selected.size < prepared.centered.shape[0]
    if rescaled:
        if np.any(row_sd_before <= 0) or not np.isfinite(row_sd_before).all():
            raise FloatingPointError('Whitened dimension has invalid sample SD')
        y /= row_sd_before[:, None]
    if not np.isfinite(y).all():
        raise FloatingPointError('Whitening produced nonfinite values')
    row_sd_after = y.std(axis=1, ddof=1)
    metadata = dict(
        threshold=threshold, rank=int(selected.size),
        retained_eigenvalue_min=float(values[-1]),
        retained_eigenvalue_max=float(values[0]),
        retained_smallest_eigenvalues=values[-min(16, len(values)):].tolist(),
        conditional_sample_sd_rescaling=bool(rescaled), sample_sd_ddof=1,
        row_sd_before_min=float(row_sd_before.min()),
        row_sd_before_max=float(row_sd_before.max()),
        row_sd_after_min=float(row_sd_after.min()),
        row_sd_after_max=float(row_sd_after.max()),
        max_abs_row_mean=float(np.max(np.abs(y.mean(axis=1)))))
    result = WhitenedData(y, threshold, values, metadata)
    if cache:
        prepared.whitened_cache[threshold] = result
    _emit(callback, 'whitening_complete', **metadata)
    return result


def _svd_coefficients(y, references, cache, callback):
    factors = cache.get('svd') if cache is not None else None
    if factors is None:
        _emit(callback, 'pinv_svd_start', shape=list(y.shape))
        u, singular, vh = linalg.svd(y, full_matrices=False,
                                    check_finite=False, lapack_driver='gesdd')
        # MATLAB pinv: tol=max(size(Y))*eps(norm(Y)); norm(Y) is its largest
        # singular value, and eps(x) is the local floating-point spacing.
        tolerance = float(max(y.shape) * np.spacing(singular[0]))
        keep = singular > tolerance
        if not keep.any():
            raise FloatingPointError('MATLAB pinv tolerance retained no singular values')
        factors = (u, singular, vh, keep, tolerance)
        if cache is not None:
            cache['svd'] = factors
        _emit(callback, 'pinv_svd_complete', retained_rank=int(keep.sum()),
              matlab_absolute_tolerance=tolerance)
    u, singular, vh, keep, tolerance = factors
    # pinv(Y).T @ R.T = U @ diag(1/s) @ Vh @ R.T. This computes its action
    # on only K reference vectors, never materializing the V-by-time pinv.
    projected = vh[keep] @ references.T
    coefficients = (u[:, keep] / singular[keep]) @ projected
    return coefficients, dict(
        requested_method='svd', resolved_method='svd',
        matlab_absolute_tolerance=tolerance, retained_rank=int(keep.sum()),
        largest_singular_value=float(singular[0]),
        smallest_retained_singular_value=float(singular[keep][-1]))


def matlab_pinv_coefficients(y, references, *, pinv_method='svd', cache=None,
                             callback: Callback = None):
    """Return pinv(Y).T @ R.T and diagnostics, using MATLAB's cutoff.

    The optional Gram route solves (Y Y')c=Y R'. It is equivalent when Y has
    full row rank and no singular values are discarded. Since forming a Gram
    matrix squares its condition number, ill-conditioned cases use SVD instead.
    """
    if np.iscomplexobj(y) or np.iscomplexobj(references):
        raise ValueError('Y and reference must be real-valued')
    y = np.asarray(y, dtype=np.float64)
    references = np.asarray(references, dtype=np.float64)
    if (y.ndim != 2 or references.ndim != 2 or y.shape[1] != references.shape[1]
            or not np.isfinite(y).all() or not np.isfinite(references).all()):
        raise ValueError('Expected finite Y [dimension, voxel] and R [component, voxel]')
    if pinv_method == 'svd':
        return _svd_coefficients(y, references, cache, callback)
    if pinv_method != 'gram':
        raise ValueError('pinv_method must be svd or gram')
    factors = cache.get('gram') if cache is not None else None
    if factors is None:
        gram = y @ y.T
        eigenvalues = linalg.eigvalsh(gram, check_finite=False, driver='evd')
        condition = (float(eigenvalues[-1] / eigenvalues[0])
                     if eigenvalues[0] > 0 else float('inf'))
        largest_singular = float(np.sqrt(max(0., eigenvalues[-1])))
        tolerance = float(max(y.shape) * np.spacing(largest_singular))
        full_rank = bool(eigenvalues[0] > tolerance ** 2)
        factors = (gram, condition, full_rank, tolerance, largest_singular,
                   float(np.sqrt(max(0., eigenvalues[0]))))
        if cache is not None:
            cache['gram'] = factors
    gram, condition, full_rank, tolerance, largest_singular, smallest = factors
    if not full_rank or condition > 1e6:
        result, metadata = _svd_coefficients(y, references, cache, callback)
        metadata.update(requested_method='gram', gram_condition_number=condition,
                        fallback_reason='Gram not demonstrably well-conditioned and full row rank')
        return result, metadata
    result = linalg.solve(gram, y @ references.T, assume_a='pos', check_finite=False)
    return result, dict(requested_method='gram', resolved_method='gram',
                        gram_condition_number=condition,
                        matlab_absolute_tolerance=tolerance, retained_rank=y.shape[0],
                        largest_singular_value=largest_singular,
                        smallest_retained_singular_value=smallest)


def _fit_components(y, normalized_reference, reference_s, callback):
    n_voxels = y.shape[1]
    sim_gradient = (y @ normalized_reference.T) / n_voxels
    weights = np.empty((y.shape[0], normalized_reference.shape[0]), dtype=np.float64)
    records = []
    for k, reference in enumerate(normalized_reference):
        _emit(callback, 'component_start', component=k + 1)
        wc = reference_s[:, k].copy()
        y1 = wc @ y
        initial_similarity = float((y1 @ reference) / n_voxels)
        initial_negentropy = _nege(y1)
        c = np.tan(initial_similarity * np.pi / 2) / initial_negentropy
        objective = float(.5 * (2 / np.pi) * np.arctan(c * initial_negentropy)
                          + .5 * initial_similarity)
        accepted_counter = 1
        accepted_proposals = 0
        step = 1.
        last_accepted = False
        stop_reason = 'attempt_limit'
        for attempt in range(1, MAX_ATTEMPTS + 1):
            with np.errstate(over='raise', invalid='raise'):
                expectation = float(np.mean(np.log(np.cosh(y1))))
            negama = expectation - GAUSSIAN_EXPECTATION
            jy1 = negama ** 2
            derivative = (2 / np.pi) * c / (1 + (c * jy1) ** 2)
            eygy = (y @ np.tanh(y1)) / n_voxels
            gradient = .5 * derivative * 2 * negama * eygy + .5 * sim_gradient[:, k]
            gradient_norm = float(np.sqrt(gradient @ gradient))
            if gradient_norm == 0 or not np.isfinite(gradient_norm):
                raise FloatingPointError('Official GIG update has zero/nonfinite gradient')
            direction = gradient / gradient_norm
            wx = wc + step * direction
            wx /= np.linalg.norm(wx)
            y3 = wx @ y
            proposed_objective = float(.5 * (2 / np.pi) * np.arctan(c * _nege(y3))
                                       + .5 * ((y3 @ reference) / n_voxels))
            change = proposed_objective - objective
            armijo_threshold = step * .02 * float(gradient @ direction)
            last_accepted = bool(change >= armijo_threshold)
            if not last_accepted:
                step /= 2
                continue
            accepted_proposals += 1
            change_squared = float((wc - wx) @ (wc - wx))
            if change_squared < WEIGHT_CHANGE_SQUARED_TOLERANCE:
                stop_reason = 'weight_change_squared'
                break
            if accepted_counter == MAX_ATTEMPTS:
                stop_reason = 'accepted_counter_limit'
                break
            objective = proposed_objective
            y1 = y3
            wc = wx
            accepted_counter += 1
        # Important source behavior: even a last rejected candidate is stored.
        # Do not silently return wc, add orthogonalization, or flip its sign.
        weights[:, k] = wx
        record = dict(component=k + 1, attempts=attempt,
                      accepted_proposals=accepted_proposals,
                      source_accepted_counter=accepted_counter,
                      stop_reason=stop_reason, final_candidate_accepted=last_accepted,
                      final_step=step, initial_similarity=initial_similarity,
                      initial_negentropy=initial_negentropy,
                      initial_c=float(c), final_candidate_objective=proposed_objective)
        records.append(record)
        _emit(callback, 'component_complete', **record)
    return weights, records


def fit_at_threshold(prepared: PreparedData, reference, threshold: float, *,
                     pinv_method='svd', callback: Callback = None):
    """Run one original threshold attempt; intended also for stagewise tests."""
    started = time.perf_counter()
    if np.iscomplexobj(reference):
        raise ValueError('reference must be real-valued')
    reference = np.asarray(reference, dtype=np.float64)
    if (reference.ndim != 2 or reference.shape[0] < 1
            or reference.shape[1] != prepared.centered.shape[1]
            or not np.isfinite(reference).all()):
        raise ValueError('reference must be finite [component, voxel] on the data grid')
    centered_reference = reference - reference.mean(axis=1, keepdims=True)
    reference_sd = centered_reference.std(axis=1, ddof=1)
    if np.any(reference_sd <= 0):
        raise ValueError('Each reference requires positive spatial sample SD')
    normalized_reference = centered_reference / reference_sd[:, None]
    whitened = whiten(prepared, threshold, callback=callback)
    coefficients, pinv_metadata = matlab_pinv_coefficients(
        whitened.y, normalized_reference, pinv_method=pinv_method,
        cache=whitened.pinv_cache, callback=callback)
    norms = np.sqrt(np.sum(coefficients ** 2, axis=0))
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise FloatingPointError('A reference has zero/nonfinite projection in whitened data')
    coefficients /= norms
    weights, component_records = _fit_components(
        whitened.y, normalized_reference, coefficients, callback)
    components = (weights.T @ whitened.y).T
    # Restore spatial means in bounded row blocks, matching FmriMatr+meanDat
    # without allocating another full time-by-voxel array.
    timecourses = np.empty((prepared.centered.shape[0], reference.shape[0]), dtype=np.float64)
    for start in range(0, len(timecourses), 128):
        stop = min(start + 128, len(timecourses))
        block = prepared.centered[start:stop] + prepared.spatial_means[start:stop, None]
        block /= prepared.centered.shape[1]
        timecourses[start:stop] = block @ components
    if not np.isfinite(components).all() or not np.isfinite(timecourses).all():
        raise FloatingPointError('GIG produced nonfinite output')
    component_sd = components.std(axis=0, ddof=1)
    sd_error = float(np.linalg.norm(1 - component_sd))
    metadata = dict(
        whitening=whitened.metadata, pinv=pinv_metadata,
        threshold=float(threshold), rank=whitened.metadata['rank'],
        n_components=reference.shape[0], sample_sd_ddof=1,
        max_iteration_attempts=MAX_ATTEMPTS,
        weight_change_squared_tolerance=WEIGHT_CHANGE_SQUARED_TOLERANCE,
        gaussian_expectation=GAUSSIAN_EXPECTATION,
        objective_weights=[.5, .5], reference_normalization='spatial_mean_and_sample_SD',
        component_sample_sd=component_sd.tolist(), unit_sd_error=sd_error,
        unit_sd_passed=bool(sd_error <= UNIT_SD_TOLERANCE),
        component_optimization=component_records,
        elapsed_seconds=time.perf_counter() - started,
        matrix_product_scalars_applied_after_reduction=True)
    _emit(callback, 'threshold_complete', threshold=float(threshold),
          rank=whitened.metadata['rank'], unit_sd_error=sd_error)
    return dict(components=components, timecourses=timecourses, metadata=metadata)


def fit_prepared(prepared: PreparedData, reference, *, pinv_method='svd',
                 callback: Callback = None):
    """Use the original eps(double), then 0.02 fallback based on output SD."""
    attempts = []
    for threshold in (FIRST_THRESHOLD, FALLBACK_THRESHOLD):
        _emit(callback, 'threshold_start', threshold=threshold)
        result = fit_at_threshold(prepared, reference, threshold,
                                  pinv_method=pinv_method, callback=callback)
        attempts.append(result['metadata'])
        if result['metadata']['unit_sd_error'] <= UNIT_SD_TOLERANCE:
            break
        if threshold == FIRST_THRESHOLD:
            _emit(callback, 'unit_sd_fallback', next_threshold=FALLBACK_THRESHOLD)
    # As in the source, the second attempt is returned even if SD tolerance
    # still fails. Its diagnostic remains explicit; there is no extra retry.
    result['metadata'] = dict(
        implementation='independent_numpy_scipy_classic_GIG_ICA_translation',
        source_name=SOURCE_NAME, source_sha256=SOURCE_SHA256,
        preparation=prepared.metadata, threshold_attempts=attempts,
        selected_threshold=threshold, fallback_used=len(attempts) > 1,
        pinv_method=pinv_method, final_unit_sd_passed=result['metadata']['unit_sd_passed'],
        input_modified=False, output_dtype='float64', matlab_invoked=False,
        independent_component_optimization=True,
        joint_orthogonality_constraint_added=False,
        **result['metadata'])
    return result


def fit_gig_ica(data, reference, *, pinv_method='svd', eigen_driver='evd',
                callback: Callback = None):
    """Fit classic GIG-ICA with no external MATLAB dependency.

    ``callback`` receives one JSON-compatible dictionary for each stage or
    component. The default SVD route is the faithful reference implementation;
    the optional Gram route must be independently compared before production.
    """
    started = time.perf_counter()
    prepared = prepare(data, eigen_driver=eigen_driver, callback=callback)
    result = fit_prepared(prepared, reference, pinv_method=pinv_method, callback=callback)
    result['metadata']['total_elapsed_seconds'] = time.perf_counter() - started
    return result
