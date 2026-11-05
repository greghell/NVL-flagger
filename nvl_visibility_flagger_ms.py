#!/usr/bin/env python3
"""
NVL visibility-domain RFI flagger: reference CPU implementation
================================================================

This is a standalone/reference implementation of the UV-grouped NVL flagger
that was developed and benchmarked in the accompanying DSA simulation work.
It does NOT depend on the simulator or the benchmark harness.

There are two intended ways to use this file:

1. As an algorithm reference for the GPU implementation

       result = flag_visibility_timestamp(
           visibilities=V,       # [baseline, channel, polarization]
           uvw_m=UVW,            # [baseline, 3], meters
           frequencies_hz=freq,  # [channel]
           input_flags=flags,    # optional, same shape as V
           config=config,
       )

   For one timestamp, the algorithm needs only:
       * complex visibilities;
       * UVW coordinates in meters for each baseline;
       * the channel-frequency axis;
       * optional pre-existing flags.

2. As a Measurement Set reader / demonstration driver

       python nvl_visibility_flagger_ms.py observation.ms --output nvl_flags

   This mode uses python-casacore to read DATA (or CORRECTED_DATA), UVW,
   ANTENNA1, ANTENNA2, FLAG, TIME, the SPECTRAL_WINDOW frequency axis, and
   POLARIZATION metadata.  It processes one timestamp and one DATA_DESC_ID at
   a time and writes NPZ flag products; it does not modify the input MS.

Algorithm
---------
For each timestamp and selected parallel-hand polarization:

  1. Convert UV coordinates to wavelengths at one reference frequency,
     normally the center/median frequency of the band.
  2. Fold conjugate UV points into one half-plane.
  3. Divide the UV plane into square cells.  Cells with at least
     min_group_size baselines are valid UV groups.
  4. For groups larger than member_cap, choose a fixed deterministic subset of
     contributors: the baselines closest to the group centroid in the
     reference-frequency UV plane.
  5. At every channel form the median of |V| over those contributors.
  6. Remove a smooth spectral baseline with a running median.
  7. Estimate the residual scale with an iteratively clipped MAD:
         sigma = 1.4826 * MAD
  8. Flag positive residuals above group_sigma_threshold * sigma.
  9. Propagate the group decision to every baseline in the UV group.
 10. Baselines outside a valid UV group use the same spectral detector
     individually, with individual_sigma_threshold.

The group assignment is fixed across ALL frequency channels of a timestamp.
The grouping is defined using UV coordinates at the reference frequency; it
is not recomputed channel by channel.

Important production note
-------------------------
This file prioritizes clarity and agreement with the offline benchmark.  It is
not a GPU-performance implementation.  In particular, group medians and
per-baseline fallback operations are expressed in NumPy/SciPy.  The NVL GPU
implementation should preserve the mathematics while replacing these pieces
with GPU kernels and rack-wide memory access.

Current benchmark defaults
--------------------------
The defaults below correspond to the currently selected 51-channel benchmark
configuration in the project archive:
    median width          = 51 channels
    UV-cell width         = 20,000 wavelengths at the reference frequency
    group member cap      = 640
    group threshold       = 10 sigma
    individual threshold  = 6 sigma
    minimum group size    = 5

These are intentionally command-line/configurable because the median-window
study is still ongoing and the production constants are not frozen.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:
    from scipy.ndimage import median_filter
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError(
        "This script requires SciPy. Install it with `pip install scipy` "
        "or `conda install scipy`."
    ) from exc


C_M_S = 299_792_458.0

# CASA/casacore Stokes/correlation enumeration values used by POLARIZATION.
CASA_CORR_TYPE_NAMES = {
    1: "I",
    2: "Q",
    3: "U",
    4: "V",
    5: "RR",
    6: "RL",
    7: "LR",
    8: "LL",
    9: "XX",
    10: "XY",
    11: "YX",
    12: "YY",
    13: "RX",
    14: "RY",
    15: "LX",
    16: "LY",
    17: "XR",
    18: "XL",
    19: "YR",
    20: "YL",
    21: "PP",
    22: "PQ",
    23: "QP",
    24: "QQ",
    25: "RCIRCULAR",
    26: "LCIRCULAR",
    27: "LINEAR",
    28: "PTOTAL",
    29: "PLINEAR",
    30: "PFTOTAL",
    31: "PFLINEAR",
    32: "PANGLE",
}


# =============================================================================
# Configuration and returned data structures
# =============================================================================


@dataclass(frozen=True)
class FlaggerConfig:
    """Parameters of the reference NVL flagger."""

    median_filter_width_channels: int = 51
    uv_cell_size_lambda: float = 20_000.0
    maximum_baselines_used_per_group: Optional[int] = 640
    minimum_uv_group_size: int = 5
    group_sigma_threshold: float = 10.0
    individual_sigma_threshold: float = 6.0
    robust_sigma_clip_iterations: int = 2
    robust_sigma_clip_threshold: float = 4.0
    reference_frequency_hz: Optional[float] = None

    def validated(self) -> "FlaggerConfig":
        width = int(self.median_filter_width_channels)
        if width < 3 or width % 2 == 0:
            raise ValueError(
                "median_filter_width_channels must be an odd integer >= 3."
            )
        if self.uv_cell_size_lambda <= 0:
            raise ValueError("uv_cell_size_lambda must be positive.")
        if self.minimum_uv_group_size < 1:
            raise ValueError("minimum_uv_group_size must be >= 1.")
        cap = self.maximum_baselines_used_per_group
        if cap is not None and cap < self.minimum_uv_group_size:
            raise ValueError(
                "maximum_baselines_used_per_group must be None or >= "
                "minimum_uv_group_size."
            )
        if self.group_sigma_threshold <= 0:
            raise ValueError("group_sigma_threshold must be positive.")
        if self.individual_sigma_threshold <= 0:
            raise ValueError("individual_sigma_threshold must be positive.")
        return self


@dataclass
class UVGrouping:
    """UV grouping for one timestamp, fixed across all frequency channels."""

    reference_frequency_hz: float
    reference_wavelength_m: float
    folded_uv_lambda: np.ndarray               # [baseline, 2]
    cell_index_per_baseline: np.ndarray        # [baseline, 2]
    group_row_for_baseline: np.ndarray         # [baseline], -1 means fallback
    all_members_by_group: list[np.ndarray]
    used_members_by_group: list[np.ndarray]
    full_group_counts: np.ndarray
    used_group_counts: np.ndarray

    @property
    def valid_baseline_mask(self) -> np.ndarray:
        return self.group_row_for_baseline >= 0

    @property
    def valid_group_count(self) -> int:
        return len(self.all_members_by_group)


@dataclass
class TimestampFlagResult:
    """Flagger output for one timestamp and one selected correlation cube."""

    new_flags: np.ndarray                      # [baseline, channel, pol], bool
    combined_flags: np.ndarray                 # input_flags OR new_flags
    grouping: UVGrouping
    group_flag_counts: np.ndarray              # [group, pol]
    fallback_baseline_count: int
    correlation_labels: tuple[str, ...]


# =============================================================================
# Generic array utilities
# =============================================================================


def ensure_odd_width(width: int, n_channels: int) -> int:
    """Return a valid odd median width no larger than the spectrum."""
    if n_channels < 3:
        raise ValueError("At least three frequency channels are required.")
    width = int(width)
    width = min(width, n_channels if n_channels % 2 else n_channels - 1)
    if width % 2 == 0:
        width -= 1
    if width < 3:
        raise ValueError("Median-filter width became smaller than 3 channels.")
    return width


def fold_uv_half_plane(uv_lambda: np.ndarray) -> np.ndarray:
    """Fold conjugate (u,v) points into a single deterministic half-plane."""
    uv = np.asarray(uv_lambda, dtype=np.float64).copy()
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv_lambda must have shape [baseline, 2].")

    flip = (
        (uv[:, 0] < 0.0)
        | (np.isclose(uv[:, 0], 0.0) & (uv[:, 1] < 0.0))
    )
    uv[flip] *= -1.0
    return uv


def _fill_invalid_spectrum(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill NaN/inf samples only for evaluating a spectral running median.

    The returned valid mask is used later so filled samples are never treated
    as real observations.  Linear interpolation is only a numerical device for
    making scipy.ndimage.median_filter well-defined near input flags/gaps.
    """
    x = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(x)
    if np.all(valid):
        return x.copy(), valid
    if not np.any(valid):
        return np.zeros_like(x), valid

    indices = np.arange(x.size, dtype=np.float64)
    finite_indices = indices[valid]
    finite_values = x[valid].astype(np.float64)
    filled = np.interp(indices, finite_indices, finite_values).astype(np.float32)
    return filled, valid


def robust_sigma_mad_1d(
    residual: np.ndarray,
    clip_iterations: int,
    clip_threshold: float,
) -> float:
    """Iteratively clipped 1.4826*MAD estimate matching the benchmark logic."""
    working = np.asarray(residual, dtype=np.float32).copy()
    working[~np.isfinite(working)] = np.nan

    for _ in range(max(0, int(clip_iterations))):
        finite = np.isfinite(working)
        if np.count_nonzero(finite) < 3:
            break
        center = float(np.nanmedian(working))
        mad = float(np.nanmedian(np.abs(working - center)))
        sigma = 1.4826 * mad
        if not np.isfinite(sigma) or sigma <= 0.0:
            break
        keep = np.abs(working - center) <= float(clip_threshold) * sigma
        working = np.where(keep, working, np.nan)

    finite = np.isfinite(working)
    if np.count_nonzero(finite) < 3:
        return float(np.finfo(np.float32).eps)

    center = float(np.nanmedian(working))
    mad = float(np.nanmedian(np.abs(working - center)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.finfo(np.float32).eps)
    return float(sigma)


def score_one_spectrum(
    amplitude_spectrum: np.ndarray,
    config: FlaggerConfig,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Running-median + clipped-MAD detector for one amplitude spectrum.

    Returns
    -------
    score : [channel]
        Positive values are excursions above the smooth spectral baseline.
        Invalid input samples are NaN.
    sigma : float
        Robust residual scale.
    spectral_baseline : [channel]
        Running-median baseline.
    """
    spectrum, valid = _fill_invalid_spectrum(amplitude_spectrum)
    width = ensure_odd_width(config.median_filter_width_channels, spectrum.size)

    baseline = median_filter(
        spectrum,
        size=width,
        mode="reflect",
    ).astype(np.float32)

    residual = spectrum - baseline
    residual_for_sigma = residual.astype(np.float32)
    residual_for_sigma[~valid] = np.nan

    sigma = robust_sigma_mad_1d(
        residual_for_sigma,
        clip_iterations=config.robust_sigma_clip_iterations,
        clip_threshold=config.robust_sigma_clip_threshold,
    )
    score = residual / sigma
    score = score.astype(np.float32)
    score[~valid] = np.nan
    return score, sigma, baseline


# =============================================================================
# UV grouping
# =============================================================================


def create_uv_grouping(
    uvw_m: np.ndarray,
    frequencies_hz: np.ndarray,
    config: FlaggerConfig,
) -> UVGrouping:
    """
    Build one grouping at a reference frequency and keep it for all channels.

    uvw_m is the projected MS UVW coordinate for each baseline at the current
    timestamp.  Only U and V are used for grouping.
    """
    config.validated()
    uvw_m = np.asarray(uvw_m, dtype=np.float64)
    frequencies_hz = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)

    if uvw_m.ndim != 2 or uvw_m.shape[1] < 2:
        raise ValueError("uvw_m must have shape [baseline, 3] (or at least 2).")
    if frequencies_hz.size < 3:
        raise ValueError("frequencies_hz must contain at least three channels.")

    reference_frequency_hz = (
        float(config.reference_frequency_hz)
        if config.reference_frequency_hz is not None
        else float(np.nanmedian(frequencies_hz))
    )
    if not np.isfinite(reference_frequency_hz) or reference_frequency_hz <= 0:
        raise ValueError("Invalid reference frequency.")

    reference_wavelength_m = C_M_S / reference_frequency_hz
    uv_lambda = uvw_m[:, :2] / reference_wavelength_m
    folded_uv_lambda = fold_uv_half_plane(uv_lambda)

    # Same square-cell convention used by the benchmark scripts.
    cell_index = np.floor(
        folded_uv_lambda / float(config.uv_cell_size_lambda) + 0.5
    ).astype(np.int32)

    _, inverse = np.unique(cell_index, axis=0, return_inverse=True)
    group_counts_all_cells = np.bincount(inverse).astype(np.int32)
    valid_group_ids = np.where(
        group_counts_all_cells >= int(config.minimum_uv_group_size)
    )[0]

    group_row_for_baseline = np.full(len(uvw_m), -1, dtype=np.int32)
    all_members_by_group: list[np.ndarray] = []
    used_members_by_group: list[np.ndarray] = []

    cap = config.maximum_baselines_used_per_group

    for group_row, group_id in enumerate(valid_group_ids):
        all_members = np.where(inverse == group_id)[0]

        # Fixed/deterministic median contributors: closest to group centroid.
        centroid = np.mean(folded_uv_lambda[all_members], axis=0)
        radial_distance = np.linalg.norm(
            folded_uv_lambda[all_members] - centroid[None, :], axis=1
        )
        ordering = np.argsort(radial_distance, kind="stable")
        ordered_members = all_members[ordering]

        if cap is None or len(ordered_members) <= int(cap):
            used_members = ordered_members
        else:
            used_members = ordered_members[: int(cap)]

        all_members = all_members.astype(np.int32)
        used_members = used_members.astype(np.int32)
        all_members_by_group.append(all_members)
        used_members_by_group.append(used_members)
        group_row_for_baseline[all_members] = group_row

    full_group_counts = np.asarray(
        [len(x) for x in all_members_by_group], dtype=np.int32
    )
    used_group_counts = np.asarray(
        [len(x) for x in used_members_by_group], dtype=np.int32
    )

    return UVGrouping(
        reference_frequency_hz=reference_frequency_hz,
        reference_wavelength_m=reference_wavelength_m,
        folded_uv_lambda=folded_uv_lambda,
        cell_index_per_baseline=cell_index,
        group_row_for_baseline=group_row_for_baseline,
        all_members_by_group=all_members_by_group,
        used_members_by_group=used_members_by_group,
        full_group_counts=full_group_counts,
        used_group_counts=used_group_counts,
    )


# =============================================================================
# Core flagger for one timestamp
# =============================================================================


def _validate_visibility_cube(
    visibilities: np.ndarray,
    uvw_m: np.ndarray,
    frequencies_hz: np.ndarray,
    input_flags: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vis = np.asarray(visibilities)
    uvw = np.asarray(uvw_m, dtype=np.float64)
    freq = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)

    if vis.ndim == 2:
        vis = vis[:, :, None]
    if vis.ndim != 3:
        raise ValueError(
            "visibilities must have shape [baseline, channel, polarization]."
        )
    if uvw.shape[0] != vis.shape[0] or uvw.ndim != 2 or uvw.shape[1] < 2:
        raise ValueError("uvw_m must match the visibility baseline dimension.")
    if freq.size != vis.shape[1]:
        raise ValueError("frequencies_hz must match the visibility channel dimension.")

    if input_flags is None:
        flags = np.zeros(vis.shape, dtype=bool)
    else:
        flags = np.asarray(input_flags, dtype=bool)
        if flags.ndim == 2 and vis.shape[2] == 1:
            flags = flags[:, :, None]
        if flags.shape != vis.shape:
            raise ValueError("input_flags must have the same shape as visibilities.")

    # Non-finite visibility samples are unavailable regardless of FLAG.
    finite = np.isfinite(vis.real) & np.isfinite(vis.imag)
    flags = flags | ~finite
    return vis, uvw, freq, flags


def _individual_score_for_baseline(
    visibilities: np.ndarray,
    input_flags: np.ndarray,
    baseline_index: int,
    pol_index: int,
    config: FlaggerConfig,
) -> np.ndarray:
    amplitude = np.abs(visibilities[baseline_index, :, pol_index]).astype(np.float32)
    amplitude[input_flags[baseline_index, :, pol_index]] = np.nan
    score, _, _ = score_one_spectrum(amplitude, config)
    return score


def flag_visibility_timestamp(
    visibilities: np.ndarray,
    uvw_m: np.ndarray,
    frequencies_hz: np.ndarray,
    input_flags: Optional[np.ndarray] = None,
    config: Optional[FlaggerConfig] = None,
    correlation_labels: Optional[Sequence[str]] = None,
) -> TimestampFlagResult:
    """
    Flag one timestamp of visibilities.

    Parameters
    ----------
    visibilities
        Complex array [baseline, channel, polarization/correlation].  Pass only
        the correlation products that should be processed (normally XX and YY,
        or RR and LL).
    uvw_m
        Projected UVW coordinates [baseline, 3] in meters for this timestamp.
    frequencies_hz
        Channel frequencies [channel].
    input_flags
        Optional pre-existing mask with the same shape as visibilities.  These
        samples do not contribute to medians and remain flagged in
        combined_flags.
    config
        Flagger parameters.  Defaults to the current benchmark configuration.
    correlation_labels
        Optional labels corresponding to the last visibility axis.

    Returns
    -------
    TimestampFlagResult
        new_flags contains only flags added by this algorithm.
        combined_flags = input_flags OR new_flags.
    """
    if config is None:
        config = FlaggerConfig()
    config = config.validated()

    vis, uvw, freq, existing_flags = _validate_visibility_cube(
        visibilities, uvw_m, frequencies_hz, input_flags
    )
    n_baselines, n_channels, n_pol = vis.shape

    if correlation_labels is None:
        labels = tuple(f"P{index}" for index in range(n_pol))
    else:
        labels = tuple(str(x) for x in correlation_labels)
        if len(labels) != n_pol:
            raise ValueError("correlation_labels length must match visibility axis 2.")

    grouping = create_uv_grouping(uvw, freq, config)
    new_flags = np.zeros(vis.shape, dtype=bool)
    group_flag_counts = np.zeros((grouping.valid_group_count, n_pol), dtype=np.int32)

    # Cache individual scores only for baselines that actually need fallback.
    individual_score_cache: dict[tuple[int, int], np.ndarray] = {}

    def individual_score(baseline_index: int, pol_index: int) -> np.ndarray:
        key = (int(baseline_index), int(pol_index))
        if key not in individual_score_cache:
            individual_score_cache[key] = _individual_score_for_baseline(
                vis,
                existing_flags,
                baseline_index,
                pol_index,
                config,
            )
        return individual_score_cache[key]

    for pol_index in range(n_pol):
        # --------------------------------------------------------------
        # Valid UV groups
        # --------------------------------------------------------------
        for group_row, (all_members, used_members) in enumerate(
            zip(grouping.all_members_by_group, grouping.used_members_by_group)
        ):
            member_vis = vis[used_members, :, pol_index]
            member_bad = existing_flags[used_members, :, pol_index]
            member_amplitude = np.abs(member_vis).astype(np.float32)
            member_amplitude[member_bad] = np.nan

            valid_member_count = np.sum(np.isfinite(member_amplitude), axis=0)

            # Suppress the expected "All-NaN slice" warning for channels in
            # which every contributor is unavailable. Those channels are
            # handled by individual fallback below.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                group_spectrum = np.nanmedian(member_amplitude, axis=0).astype(np.float32)

            enough_members = valid_member_count >= int(config.minimum_uv_group_size)
            group_spectrum[~enough_members] = np.nan

            group_score, _, _ = score_one_spectrum(group_spectrum, config)
            group_decision = (
                np.isfinite(group_score)
                & (group_score > float(config.group_sigma_threshold))
                & enough_members
            )
            group_flag_counts[group_row, pol_index] = int(np.sum(group_decision))

            # Propagate group decision to every baseline in the cell.
            new_flags[all_members, :, pol_index] = group_decision[None, :]

            # If a channel cannot form a valid group median because too many
            # contributors were unavailable/pre-flagged, use the individual
            # detector for that channel, as specified in the report edge cases.
            insufficient_channels = ~enough_members
            if np.any(insufficient_channels):
                for baseline_index in all_members:
                    score = individual_score(int(baseline_index), pol_index)
                    new_flags[baseline_index, insufficient_channels, pol_index] = (
                        np.isfinite(score[insufficient_channels])
                        & (
                            score[insufficient_channels]
                            > float(config.individual_sigma_threshold)
                        )
                    )

        # --------------------------------------------------------------
        # Baselines outside valid groups: individual detector
        # --------------------------------------------------------------
        fallback_baselines = np.where(~grouping.valid_baseline_mask)[0]
        for baseline_index in fallback_baselines:
            score = individual_score(int(baseline_index), pol_index)
            new_flags[baseline_index, :, pol_index] = (
                np.isfinite(score)
                & (score > float(config.individual_sigma_threshold))
            )

    # Do not call an already input-flagged sample a newly detected flag.
    new_flags &= ~existing_flags
    combined_flags = existing_flags | new_flags

    return TimestampFlagResult(
        new_flags=new_flags,
        combined_flags=combined_flags,
        grouping=grouping,
        group_flag_counts=group_flag_counts,
        fallback_baseline_count=int(np.sum(~grouping.valid_baseline_mask)),
        correlation_labels=labels,
    )


# =============================================================================
# Measurement Set helpers
# =============================================================================


def _import_casacore_table():
    try:
        from casacore.tables import table  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Measurement Set mode requires python-casacore. Install it in the "
            "environment used to run this script, e.g. `conda install -c "
            "conda-forge python-casacore`. The core flag_visibility_timestamp() "
            "function itself does not require casacore."
        ) from exc
    return table


def _subtable_path(ms_path: Path, name: str) -> str:
    return str(ms_path / name)


def read_ms_data_description_metadata(ms_path: Path) -> dict[int, dict[str, object]]:
    """Return DDID -> SPW/polarization/frequency/correlation metadata."""
    table = _import_casacore_table()
    metadata: dict[int, dict[str, object]] = {}

    dd = table(_subtable_path(ms_path, "DATA_DESCRIPTION"), readonly=True, ack=False)
    try:
        spw_ids = np.asarray(dd.getcol("SPECTRAL_WINDOW_ID"), dtype=np.int32)
        pol_ids = np.asarray(dd.getcol("POLARIZATION_ID"), dtype=np.int32)
    finally:
        dd.close()

    spw = table(_subtable_path(ms_path, "SPECTRAL_WINDOW"), readonly=True, ack=False)
    pol = table(_subtable_path(ms_path, "POLARIZATION"), readonly=True, ack=False)
    try:
        for ddid, (spw_id, pol_id) in enumerate(zip(spw_ids, pol_ids)):
            frequencies_hz = np.asarray(spw.getcell("CHAN_FREQ", int(spw_id)), dtype=np.float64)
            corr_types = np.asarray(pol.getcell("CORR_TYPE", int(pol_id)), dtype=np.int32)
            corr_labels = tuple(
                CASA_CORR_TYPE_NAMES.get(int(value), f"CORR_{int(value)}")
                for value in corr_types
            )
            metadata[int(ddid)] = {
                "spectral_window_id": int(spw_id),
                "polarization_id": int(pol_id),
                "frequencies_hz": frequencies_hz,
                "correlation_type_codes": corr_types,
                "correlation_labels": corr_labels,
            }
    finally:
        pol.close()
        spw.close()

    return metadata


def select_correlation_indices(
    available_labels: Sequence[str],
    requested: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Select parallel hands automatically, or parse a comma-separated list."""
    labels = tuple(str(x).upper() for x in available_labels)

    if requested.lower() == "auto":
        if "XX" in labels and "YY" in labels:
            desired = ("XX", "YY")
        elif "RR" in labels and "LL" in labels:
            desired = ("RR", "LL")
        else:
            parallel = [x for x in labels if x in {"XX", "YY", "RR", "LL"}]
            desired = tuple(parallel) if parallel else labels
    else:
        desired = tuple(
            token.strip().upper()
            for token in requested.split(",")
            if token.strip()
        )
        if not desired:
            raise ValueError("--correlations did not contain any labels.")

    missing = [label for label in desired if label not in labels]
    if missing:
        raise ValueError(
            f"Requested correlations {missing} are absent. Available: {labels}"
        )

    indices = np.asarray([labels.index(label) for label in desired], dtype=np.int32)
    return indices, desired


def _normalize_ms_cube(
    array: np.ndarray,
    n_channels: int,
    n_correlations: int,
    name: str,
) -> np.ndarray:
    """Normalize casacore DATA/FLAG getcol output to [row, channel, corr]."""
    x = np.asarray(array)
    if x.ndim != 3:
        raise ValueError(f"{name} must be a 3-D MS array; got shape {x.shape}.")

    if x.shape[1:] == (n_channels, n_correlations):
        return x
    if x.shape[1:] == (n_correlations, n_channels):
        return np.swapaxes(x, 1, 2)

    raise ValueError(
        f"Cannot identify channel/correlation axes of {name}: shape={x.shape}, "
        f"expected (*,{n_channels},{n_correlations}) or "
        f"(*,{n_correlations},{n_channels})."
    )


def _safe_query_row_numbers(reference_table) -> np.ndarray:
    """Best-effort original MS row numbers for a casacore reference table."""
    try:
        return np.asarray(reference_table.rownumbers(), dtype=np.int64)
    except Exception:
        # Row numbers are metadata only; ANTENNA1/2 and TIME remain sufficient
        # to identify the visibility rows if a casacore build lacks rownumbers().
        return np.arange(reference_table.nrows(), dtype=np.int64)


def _timestamp_filename(ddid: int, time_index: int) -> str:
    return f"nvl_flags_ddid{ddid:03d}_time{time_index:06d}.npz"


def process_measurement_set(
    ms_path: Path,
    output_directory: Path,
    config: FlaggerConfig,
    data_column: str,
    correlations: str,
    ddids: Optional[Sequence[int]],
    max_times: Optional[int],
) -> None:
    """Read an MS timestamp-by-timestamp and save standalone flag products."""
    table = _import_casacore_table()
    output_directory.mkdir(parents=True, exist_ok=True)

    dd_metadata = read_ms_data_description_metadata(ms_path)

    main = table(str(ms_path), readonly=True, ack=False)
    try:
        colnames = set(main.colnames())
        required = {
            data_column,
            "UVW",
            "ANTENNA1",
            "ANTENNA2",
            "TIME",
            "DATA_DESC_ID",
        }
        missing = sorted(required - colnames)
        if missing:
            raise KeyError(f"Measurement Set is missing columns: {missing}")

        has_flag = "FLAG" in colnames
        has_flag_row = "FLAG_ROW" in colnames

        if ddids is None:
            ddid_values = sorted(int(x) for x in np.unique(main.getcol("DATA_DESC_ID")))
        else:
            ddid_values = [int(x) for x in ddids]

        for ddid in ddid_values:
            if ddid not in dd_metadata:
                raise ValueError(f"DATA_DESC_ID {ddid} is not present in DATA_DESCRIPTION.")

            meta = dd_metadata[ddid]
            frequencies_hz = np.asarray(meta["frequencies_hz"], dtype=np.float64)
            available_labels = tuple(meta["correlation_labels"])
            corr_indices, selected_labels = select_correlation_indices(
                available_labels, correlations
            )

            # Discover integrations from TIME for this DDID. Autocorrelations
            # are excluded because the benchmark grouping is defined on cross
            # correlations only.
            dd_ref = main.query(
                f"DATA_DESC_ID=={ddid} && ANTENNA1!=ANTENNA2",
                columns="TIME",
                sortlist="TIME",
            )
            try:
                times = np.unique(np.asarray(dd_ref.getcol("TIME"), dtype=np.float64))
            finally:
                dd_ref.close()

            if max_times is not None:
                times = times[: int(max_times)]

            print("\n" + "=" * 78)
            print(f"DATA_DESC_ID {ddid}")
            print(f"SPW:             {meta['spectral_window_id']}")
            print(f"Channels:        {len(frequencies_hz)}")
            print(
                f"Frequency range: {frequencies_hz.min()/1e6:.6f}--"
                f"{frequencies_hz.max()/1e6:.6f} MHz"
            )
            print(f"Available corr:  {', '.join(available_labels)}")
            print(f"Processed corr:  {', '.join(selected_labels)}")
            print(f"Timestamps:      {len(times)}")

            manifest_rows: list[dict[str, object]] = []

            for time_index, timestamp in enumerate(times):
                # Use a high-precision literal because timestamp came directly
                # from the same MS column.
                time_literal = format(float(timestamp), ".17g")
                ref = main.query(
                    (
                        f"DATA_DESC_ID=={ddid} && ANTENNA1!=ANTENNA2 "
                        f"&& TIME=={time_literal}"
                    ),
                    sortlist="ANTENNA1,ANTENNA2",
                )
                try:
                    if ref.nrows() == 0:
                        continue

                    n_corr = len(available_labels)
                    data_full = _normalize_ms_cube(
                        ref.getcol(data_column),
                        len(frequencies_hz),
                        n_corr,
                        data_column,
                    )
                    uvw_m = np.asarray(ref.getcol("UVW"), dtype=np.float64)
                    antenna1 = np.asarray(ref.getcol("ANTENNA1"), dtype=np.int32)
                    antenna2 = np.asarray(ref.getcol("ANTENNA2"), dtype=np.int32)
                    row_numbers = _safe_query_row_numbers(ref)

                    if has_flag:
                        flag_full = _normalize_ms_cube(
                            ref.getcol("FLAG"),
                            len(frequencies_hz),
                            n_corr,
                            "FLAG",
                        ).astype(bool)
                    else:
                        flag_full = np.zeros(data_full.shape, dtype=bool)

                    if has_flag_row:
                        flag_row = np.asarray(ref.getcol("FLAG_ROW"), dtype=bool)
                        flag_full[flag_row, :, :] = True

                    data_selected = data_full[:, :, corr_indices]
                    flag_selected = flag_full[:, :, corr_indices]

                    result = flag_visibility_timestamp(
                        visibilities=data_selected,
                        uvw_m=uvw_m,
                        frequencies_hz=frequencies_hz,
                        input_flags=flag_selected,
                        config=config,
                        correlation_labels=selected_labels,
                    )

                    output_path = output_directory / _timestamp_filename(ddid, time_index)
                    np.savez_compressed(
                        output_path,
                        new_flags=result.new_flags,
                        combined_flags=result.combined_flags,
                        input_flags=flag_selected,
                        row_numbers=row_numbers,
                        antenna1=antenna1,
                        antenna2=antenna2,
                        uvw_m=uvw_m,
                        frequencies_hz=frequencies_hz,
                        correlation_indices=corr_indices,
                        correlation_labels=np.asarray(selected_labels),
                        time_seconds=np.asarray(float(timestamp)),
                        data_desc_id=np.asarray(ddid),
                        spectral_window_id=np.asarray(meta["spectral_window_id"]),
                        reference_frequency_hz=np.asarray(
                            result.grouping.reference_frequency_hz
                        ),
                        folded_uv_lambda=result.grouping.folded_uv_lambda,
                        uv_cell_indices=result.grouping.cell_index_per_baseline,
                        group_row_for_baseline=result.grouping.group_row_for_baseline,
                        full_group_counts=result.grouping.full_group_counts,
                        used_group_counts=result.grouping.used_group_counts,
                        group_flag_counts=result.group_flag_counts,
                        config_json=np.asarray(json.dumps(asdict(config))),
                    )

                    n_new = int(np.sum(result.new_flags))
                    n_samples = int(result.new_flags.size)
                    manifest_rows.append(
                        {
                            "data_desc_id": ddid,
                            "time_index": time_index,
                            "time_seconds": float(timestamp),
                            "n_baselines": int(ref.nrows()),
                            "n_channels": int(len(frequencies_hz)),
                            "correlations": ",".join(selected_labels),
                            "reference_frequency_hz": result.grouping.reference_frequency_hz,
                            "valid_groups": result.grouping.valid_group_count,
                            "fallback_baselines": result.fallback_baseline_count,
                            "new_flagged_samples": n_new,
                            "new_flagged_fraction": (
                                n_new / n_samples if n_samples else math.nan
                            ),
                            "output_file": output_path.name,
                        }
                    )

                    print(
                        f"  [{time_index + 1:4d}/{len(times):4d}] "
                        f"rows={ref.nrows():7d}, "
                        f"groups={result.grouping.valid_group_count:5d}, "
                        f"fallback={result.fallback_baseline_count:5d}, "
                        f"new flags={n_new / n_samples:.5%}"
                    )
                finally:
                    ref.close()

            manifest_path = (
                output_directory / f"nvl_flagger_manifest_ddid{ddid:03d}.csv"
            )
            if manifest_rows:
                with manifest_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=list(manifest_rows[0].keys())
                    )
                    writer.writeheader()
                    writer.writerows(manifest_rows)
    finally:
        main.close()


# =============================================================================
# Command line
# =============================================================================


def _parse_member_cap(value: str) -> Optional[int]:
    if value.strip().lower() in {"all", "none", "unlimited"}:
        return None
    cap = int(value)
    if cap < 1:
        raise argparse.ArgumentTypeError("member cap must be positive or 'all'.")
    return cap


def _parse_ddids(value: Optional[str]) -> Optional[list[int]]:
    if value is None or value.strip().lower() == "all":
        return None
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reference implementation of the UV-grouped NVL visibility RFI "
            "flagger for Measurement Sets."
        )
    )
    parser.add_argument("measurement_set", type=Path, help="Input Measurement Set directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("nvl_flagger_output"),
        help="Directory for per-timestamp NPZ masks and manifests.",
    )
    parser.add_argument(
        "--data-column",
        default="DATA",
        help="MS visibility column, usually DATA or CORRECTED_DATA (default: DATA).",
    )
    parser.add_argument(
        "--correlations",
        default="auto",
        help=(
            "Comma-separated correlation labels (e.g. XX,YY or RR,LL). "
            "Default 'auto' selects parallel hands."
        ),
    )
    parser.add_argument(
        "--ddids",
        default="all",
        help="Comma-separated DATA_DESC_ID values, or 'all' (default).",
    )
    parser.add_argument(
        "--max-times",
        type=int,
        default=None,
        help="Optional limit for quick tests.",
    )

    # Flagger parameters. These defaults are benchmark candidates, not frozen
    # production requirements.
    parser.add_argument("--median-width", type=int, default=51)
    parser.add_argument("--uv-cell-width", type=float, default=20_000.0)
    parser.add_argument("--member-cap", type=_parse_member_cap, default=640)
    parser.add_argument("--min-group-size", type=int, default=5)
    parser.add_argument("--group-threshold", type=float, default=10.0)
    parser.add_argument("--individual-threshold", type=float, default=6.0)
    parser.add_argument("--clip-iterations", type=int, default=2)
    parser.add_argument("--clip-threshold", type=float, default=4.0)
    parser.add_argument(
        "--reference-frequency-hz",
        type=float,
        default=None,
        help="Reference frequency for UV grouping; default is median channel frequency.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    ms_path = args.measurement_set.expanduser().resolve()
    if not ms_path.exists():
        raise FileNotFoundError(f"Measurement Set not found: {ms_path}")

    config = FlaggerConfig(
        median_filter_width_channels=args.median_width,
        uv_cell_size_lambda=args.uv_cell_width,
        maximum_baselines_used_per_group=args.member_cap,
        minimum_uv_group_size=args.min_group_size,
        group_sigma_threshold=args.group_threshold,
        individual_sigma_threshold=args.individual_threshold,
        robust_sigma_clip_iterations=args.clip_iterations,
        robust_sigma_clip_threshold=args.clip_threshold,
        reference_frequency_hz=args.reference_frequency_hz,
    ).validated()

    print("NVL visibility flagger reference implementation")
    print("=" * 78)
    print(f"Measurement Set: {ms_path}")
    print(f"Data column:     {args.data_column}")
    print(f"Output:          {args.output}")
    print("Configuration:")
    print(json.dumps(asdict(config), indent=2))

    process_measurement_set(
        ms_path=ms_path,
        output_directory=args.output.expanduser().resolve(),
        config=config,
        data_column=args.data_column,
        correlations=args.correlations,
        ddids=_parse_ddids(args.ddids),
        max_times=args.max_times,
    )


if __name__ == "__main__":
    main()
