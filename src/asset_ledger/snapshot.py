"""
ISO 13374 Block 3 — State Detection (asset ledger: derived health snapshot).

A health snapshot is the DERIVED view of one measurement: time-domain
indicators, the amplitude at the shaft frequency (1x), the bearing evidence
at the expected fault frequencies and the ISO 20816-3 severity, computed
from the signal, the declared measurement identity and the declared context
of the measurement point. It is composed from the pure engines of the
codebase (never from the diagnosis pipeline, which runs an unversioned
anomaly model and writes work-order prose) and it is PARTIAL whenever the
context is: every absent block carries a ``{reason, remedy}`` entry under
``missing`` instead of a silent default.

Lineage. Two snapshots are comparable only when their ``processing_id`` is
identical: ``health_snapshot/<ALGORITHM_VERSION>+<hash of the policy
parameters>``. ``ALGORITHM_VERSION`` is bumped BY HAND whenever a change
along the indicator paths (``signal_processing/features.py``,
``signal_processing/spectral.py``, ``diagnostics/bearing_analyzer.py``,
``diagnostics/iso20816.py``) can change the numbers; the golden tripwire
``tests/fixtures/asset_snapshot_golden.json`` goes red on such a change
without a bump. Data-dependent values (the envelope band resolved from the
sampling rate, the number of samples) live in the informative ``effective``
block, outside the key, so lineages never fragment by file. Package and
library versions are provenance, not key.

Context digest. ``context_digest`` hashes the RESOLVED inputs of the
snapshot (rpm and its source, bearing or fault orders, machine group and
support, power, unit, raw format, sampling rate, direction): correcting the
rpm in the companion yields a new snapshot, re-declaring the point with a
different note does not.

Pure module: numpy, the signal-processing and diagnostics engines and the
measurement vocabularies only. No repository, no ``models.py``, no MCP, no
file I/O, no randomness, no wall clock (the platform provenance is
overridable for tests).
"""

import hashlib
import json
import logging
import math
import platform
from typing import Any, Callable, Mapping, NamedTuple, Optional, Union

import numpy as np
import scipy

from ..diagnostics.bearing_analyzer import check_all_bearing_faults, check_frequency_set
from ..diagnostics.bearing_catalog import compute_fault_frequencies
from ..diagnostics.iso20816 import assess_severity_raw
from ..signal_acquisition.measurement import VALID_DIRECTIONS
from ..signal_processing.features import extract_time_domain_features
from ..signal_processing.spectral import (
    amplitude_near_frequency,
    amplitude_spectrum,
    envelope_spectrum_arrays,
    resolve_envelope_band,
    select_leading_segment,
)

__all__ = [
    "ALGORITHM_VERSION",
    "PROCESSING_FAMILY",
    "BEARING_LABELS",
    "SNAPSHOT_PROVENANCE_KEYS",
    "SNAPSHOT_BLOCKS",
    "ENVELOPE_BAND_DEFAULT",
    "SnapshotPolicy",
    "policy_params",
    "processing_id",
    "resolve_context",
    "context_digest",
    "collect_provenance",
    "expected_frequencies",
    "compute_health_snapshot",
    "snapshot_id",
]

logger = logging.getLogger(__name__)

#: Hand-bumped whenever the numbers of ANY snapshot block can change
#: (see the module docstring). Part of ``processing_id``.
ALGORITHM_VERSION = 1

#: Prefix of every ``processing_id`` emitted by this module.
PROCESSING_FAMILY = "health_snapshot"

#: Bearing fault labels of the catalog route, in canonical order.
BEARING_LABELS: tuple[str, ...] = ("BPFO", "BPFI", "BSF", "FTF")

#: Keys of the provenance block: what the numbers were computed with. The
#: relevant subset of ``benchmarks/cwru/runner.py::PROVENANCE_KEYS`` (no
#: ``date``, no ``git_describe``: a snapshot is timestamped by the ledger
#: event that records it, and the code is versioned by ``pipeline_version``).
SNAPSHOT_PROVENANCE_KEYS: frozenset[str] = frozenset(
    {"platform", "python_version", "numpy_version", "scipy_version", "pipeline_version"}
)

#: The optional blocks; each one absent has an entry in ``missing``.
SNAPSHOT_BLOCKS: tuple[str, ...] = ("one_x", "bearing", "iso")

#: Policy value meaning "the fs-aware default band of
#: :func:`~..signal_processing.spectral.resolve_envelope_band`".
ENVELOPE_BAND_DEFAULT = "default"

#: Length of every identifier derived here: the first 16 hex of a SHA-256.
_ID_HEX_CHARS = 16

#: Remedy shared by the blocks that need a shaft speed.
_RPM_REMEDY = (
    'declare "rpm" in the "measurement" object of the companion and re-load the '
    "file, or declare nominal_rpm for the measurement point via "
    "declare_measurement_point and re-process"
)


class SnapshotPolicy(NamedTuple):
    """Parameters that DEFINE the snapshot algorithm (the lineage key).

    Only what changes the numbers independently of the data belongs here;
    values resolved from the signal (band edges from fs, sample counts) are
    reported in the snapshot's ``effective`` block instead.

    Attributes:
        tolerance_pct: Frequency tolerance (percent) of the bearing fault
            matching and of the envelope-amplitude window per label.
        num_harmonics: Harmonics (2x, 3x, ...) checked per fault label.
        envelope_band_policy: ``"default"`` for the fs-aware default band,
            or an explicit ``(low_hz, high_hz)`` tuple honoured verbatim.
        fft_segment_s: Leading segment (seconds) the 1x amplitude spectrum
            is computed on, or ``None`` for the whole signal.
        one_x_tolerance_pct: Half-width (percent of the shaft frequency) of
            the window the 1x amplitude is searched in.
    """

    tolerance_pct: float = 5.0
    num_harmonics: int = 3
    envelope_band_policy: Union[str, tuple[float, float]] = ENVELOPE_BAND_DEFAULT
    fft_segment_s: Optional[float] = 1.0
    one_x_tolerance_pct: float = 5.0


def _canonical_json(payload: Any) -> str:
    """Canonical JSON: sorted keys, compact, ASCII, no NaN/Infinity."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    )


def _short_digest(payload: Any) -> str:
    """First 16 hex digits of the SHA-256 of the canonical JSON of *payload*."""
    encoded = _canonical_json(payload).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()[:_ID_HEX_CHARS]


def _positive_number(name: str, value: object) -> float:
    """Return *value* as a float, refusing bools, non-numbers, NaN, <= 0."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
    return float(value)


def _finite_or_none(value: float) -> Optional[float]:
    """JSON cannot carry NaN or Infinity: a non-finite indicator becomes None."""
    number = float(value)
    return number if math.isfinite(number) else None


def policy_params(policy: SnapshotPolicy) -> dict[str, Any]:
    """Validate *policy* and return its canonical, JSON-ready parameters.

    The returned dict is what ``processing_id`` hashes and what the
    snapshot records under ``processing.params``; every value has one
    canonical representation (floats for the percentages and the segment,
    an int for the harmonics, ``"default"`` or ``[low, high]`` for the
    band), so equal policies always hash equal.

    Raises:
        ValueError: If *policy* is not a :class:`SnapshotPolicy` or one of
            its fields is out of range.
    """
    if not isinstance(policy, SnapshotPolicy):
        raise ValueError(
            f"policy must be a SnapshotPolicy, got {type(policy).__name__}."
        )
    harmonics = policy.num_harmonics
    if isinstance(harmonics, bool) or not isinstance(harmonics, int) or harmonics < 0:
        raise ValueError(
            f"num_harmonics must be a non-negative integer, got {harmonics!r}."
        )
    band = policy.envelope_band_policy
    band_param: Union[str, list[float]]
    if isinstance(band, str):
        if band != ENVELOPE_BAND_DEFAULT:
            raise ValueError(
                f"envelope_band_policy must be {ENVELOPE_BAND_DEFAULT!r} or an "
                f"explicit (low_hz, high_hz) tuple, got {band!r}."
            )
        band_param = ENVELOPE_BAND_DEFAULT
    else:
        if not isinstance(band, (tuple, list)) or len(band) != 2:
            raise ValueError(
                f"envelope_band_policy must be {ENVELOPE_BAND_DEFAULT!r} or an "
                f"explicit (low_hz, high_hz) tuple, got {band!r}."
            )
        low = _positive_number("envelope_band_policy low edge", band[0])
        high = _positive_number("envelope_band_policy high edge", band[1])
        if high <= low:
            raise ValueError(
                f"envelope_band_policy high edge {high:g} Hz must exceed the low "
                f"edge {low:g} Hz."
            )
        band_param = [low, high]
    segment = policy.fft_segment_s
    segment_param = (
        None if segment is None else _positive_number("fft_segment_s", segment)
    )
    return {
        "tolerance_pct": _positive_number("tolerance_pct", policy.tolerance_pct),
        "num_harmonics": harmonics,
        "envelope_band_policy": band_param,
        "fft_segment_s": segment_param,
        "one_x_tolerance_pct": _positive_number(
            "one_x_tolerance_pct", policy.one_x_tolerance_pct
        ),
    }


def processing_id(policy: Optional[SnapshotPolicy] = None) -> str:
    """Lineage key: ``health_snapshot/<ALGORITHM_VERSION>+<policy hash>``.

    Args:
        policy: The snapshot policy; ``None`` means the defaults (resolved
            at call time, so a changed default changes the id).

    Returns:
        The processing id string. Snapshots are comparable only when their
        ids are identical.
    """
    resolved = SnapshotPolicy() if policy is None else policy
    return f"{PROCESSING_FAMILY}/{ALGORITHM_VERSION}+{_short_digest(policy_params(resolved))}"


def _normalize_fault_orders(value: object) -> Optional[dict[str, float]]:
    """Canonical ``{label: order}`` of a declared fault-order mapping.

    Orders are multiples of the shaft frequency (BPFO of a 6205 is about
    3.58), so they scale with the rpm of every measurement. ``None`` and an
    empty mapping both mean "not declared".

    Raises:
        ValueError: If *value* is not a mapping of non-empty string labels
            to positive finite numbers.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(
            f"fault_orders must be a mapping {{label: order}} such as "
            f"{{'BPFO': 3.58}}, got {type(value).__name__}."
        )
    if not value:
        return None
    orders: dict[str, float] = {}
    for label, order in value.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError(
                f"fault_orders label {label!r} must be a non-empty string."
            )
        orders[label] = _positive_number(f"fault_orders[{label!r}]", order)
    return orders


def resolve_context(
    declaration: Mapping[str, Any], point: Optional[Mapping[str, Any]]
) -> dict[str, Any]:
    """Resolve the diagnostic context of one measurement.

    Precedence (the plan's "parametri diagnostici"): the measurement
    declaration wins for the rpm, the measurement point supplies the
    nominal rpm as fallback and everything mechanical (bearing designation
    or fault orders, machine group, support type, rated power); the unit is
    the DECLARED one, never guessed; nothing here has a silent default.

    Args:
        declaration: Effective declaration of the measurement: the
            normalized identity block (``rpm``, ``direction``, ...) plus
            ``signal_unit``, ``sampling_rate`` and ``raw_format`` as stored
            on the signal. Absent keys read as "not declared".
        point: Latest declaration of the measurement point (``nominal_rpm``,
            ``bearing_id``, ``fault_orders``, ``machine_group``,
            ``support_type``, ``machine_power_kw``, ...), or ``None`` when
            the point was never declared.

    Returns:
        Dict with ``rpm`` (float or None), ``rpm_source``
        (``"measurement"``, ``"point"`` or None), ``bearing_id``,
        ``fault_orders`` (canonical mapping or None), ``machine_group``,
        ``support_type``, ``machine_power_kw``, ``signal_unit`` and
        ``direction``.

    Raises:
        ValueError: If a declared value is malformed (non-positive rpm, a
            direction outside the vocabulary, a bad fault-order mapping):
            these are caller bugs, since every declaration was validated at
            its own boundary before reaching the ledger.
    """
    point_ctx: Mapping[str, Any] = point if point is not None else {}

    rpm: Optional[float]
    rpm_source: Optional[str]
    if declaration.get("rpm") is not None:
        rpm, rpm_source = _positive_number("rpm", declaration["rpm"]), "measurement"
    elif point_ctx.get("nominal_rpm") is not None:
        rpm = _positive_number("nominal_rpm", point_ctx["nominal_rpm"])
        rpm_source = "point"
    else:
        rpm, rpm_source = None, None

    direction = declaration.get("direction")
    if direction is not None and direction not in VALID_DIRECTIONS:
        raise ValueError(
            f"direction {direction!r} is outside the declared vocabulary "
            f"{list(VALID_DIRECTIONS)}; normalize it at the declaration boundary."
        )

    bearing_id = point_ctx.get("bearing_id")
    if bearing_id is not None and (not isinstance(bearing_id, str) or not bearing_id):
        raise ValueError(f"bearing_id must be a non-empty string, got {bearing_id!r}.")

    power = point_ctx.get("machine_power_kw")
    return {
        "rpm": rpm,
        "rpm_source": rpm_source,
        "bearing_id": bearing_id,
        "fault_orders": _normalize_fault_orders(point_ctx.get("fault_orders")),
        "machine_group": point_ctx.get("machine_group"),
        "support_type": point_ctx.get("support_type"),
        "machine_power_kw": None if power is None else float(power),
        "signal_unit": declaration.get("signal_unit"),
        "direction": direction,
    }


def context_digest(context: Mapping[str, Any], declaration: Mapping[str, Any]) -> str:
    """Short hash of the RESOLVED inputs the snapshot numbers depend on.

    Rpm value and source, bearing designation or fault orders, machine
    group, support type, rated power, declared unit, effective raw format,
    sampling rate and direction. A point re-declared with only a different
    note keeps the digest; an rpm corrected in the companion changes it.

    Args:
        context: Output of :func:`resolve_context`.
        declaration: The effective declaration (read for ``raw_format`` and
            ``sampling_rate``).

    Returns:
        16 lowercase hex characters.
    """
    sampling_rate = declaration.get("sampling_rate")
    payload = {
        "rpm": context.get("rpm"),
        "rpm_source": context.get("rpm_source"),
        "bearing_id": context.get("bearing_id"),
        "fault_orders": context.get("fault_orders"),
        "machine_group": context.get("machine_group"),
        "support_type": context.get("support_type"),
        "machine_power_kw": context.get("machine_power_kw"),
        "signal_unit": context.get("signal_unit"),
        "raw_format": declaration.get("raw_format"),
        "sampling_rate": None if sampling_rate is None else float(sampling_rate),
        "direction": context.get("direction"),
    }
    return _short_digest(payload)


def _package_version() -> str:
    """The server version, imported lazily (see ``advisory._build_provenance``:
    the package ``__init__`` imports the server, which imports this chain)."""
    from .. import __version__

    return str(__version__)


def collect_provenance(overrides: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Collect what the snapshot was computed with, deterministically overridable.

    Mirrors ``benchmarks/cwru/runner.py::collect_provenance``: every value
    that varies between machines is produced lazily and can be pinned, so
    golden tests never depend on the host. Provenance is recorded, never
    part of the lineage key.

    Args:
        overrides: Values to use verbatim; keys must be a subset of
            :data:`SNAPSHOT_PROVENANCE_KEYS`.

    Returns:
        A dict with exactly :data:`SNAPSHOT_PROVENANCE_KEYS`.

    Raises:
        ValueError: On an unknown override key (fail closed: a misspelled
            override silently ignored would leave a varying value in a
            fixture a test believed pinned).
    """
    resolved = dict(overrides or {})
    unknown = sorted(set(resolved) - SNAPSHOT_PROVENANCE_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown provenance override key(s) {unknown} — valid keys are "
            f"{sorted(SNAPSHOT_PROVENANCE_KEYS)}. Fix the caller."
        )
    producers: dict[str, Callable[[], str]] = {
        "platform": platform.platform,
        "python_version": platform.python_version,
        "numpy_version": lambda: str(np.__version__),
        "scipy_version": lambda: str(scipy.__version__),
        "pipeline_version": _package_version,
    }
    return {
        key: str(resolved[key]) if key in resolved else producer()
        for key, producer in producers.items()
    }


def expected_frequencies(
    rpm: float,
    *,
    bearing_id: Optional[str] = None,
    fault_orders: Optional[Mapping[str, float]] = None,
) -> dict[str, Any]:
    """Shaft frequency and expected fault frequencies for *rpm*.

    THE single place of the ledger where rpm becomes Hz: the 1x target and
    every bearing frequency. A catalog designation takes precedence and
    goes through :func:`compute_fault_frequencies` (which also returns the
    shaft frequency); declared orders are multiplied by the shaft frequency;
    with neither, only the shaft frequency is returned.

    Args:
        rpm: Resolved shaft speed (rpm), > 0.
        bearing_id: Catalog designation, or ``None``.
        fault_orders: ``{label: order}`` multiples of the shaft frequency,
            used only when *bearing_id* is ``None``.

    Returns:
        Dict with ``shaft_hz``, ``source`` (``"catalog"``,
        ``"fault_orders"`` or ``None``), ``bearing_id`` (catalog route only),
        ``frequencies`` (``{label: hz}`` or ``None``) and ``catalog_source``
        (the catalog entry's citation, catalog route only).

    Raises:
        ValueError: If *bearing_id* is not in the catalog. The declared
            orders are NOT used as a silent fallback: a designation that
            does not resolve is a declaration to fix, not to work around.
    """
    if bearing_id is not None:
        freq_data = compute_fault_frequencies(bearing_id, rpm)
        if freq_data is None:
            raise ValueError(
                f"Bearing '{bearing_id}' not found in the catalog — check the "
                f"designation (search_bearing_catalog) and re-declare the "
                f"measurement point, or declare fault_orders instead of a "
                f"designation."
            )
        return {
            "shaft_hz": float(freq_data["shaft_freq_hz"]),
            "source": "catalog",
            "bearing_id": bearing_id,
            "frequencies": {label: float(freq_data[label]) for label in BEARING_LABELS},
            "catalog_source": (freq_data.get("bearing_info") or {}).get("source"),
        }
    shaft_hz = rpm / 60.0
    if fault_orders:
        return {
            "shaft_hz": shaft_hz,
            "source": "fault_orders",
            "bearing_id": None,
            "frequencies": {
                label: float(order) * shaft_hz for label, order in fault_orders.items()
            },
            "catalog_source": None,
        }
    return {
        "shaft_hz": shaft_hz,
        "source": None,
        "bearing_id": None,
        "frequencies": None,
        "catalog_source": None,
    }


def _missing(reason: str, remedy: str) -> dict[str, str]:
    """A ``missing`` entry (the ``reason``/``remedy`` shape of the ISO refusal)."""
    return {"reason": reason, "remedy": remedy}


def _bearing_block(
    signal: np.ndarray,
    fs: float,
    rpm: float,
    targets: Mapping[str, Any],
    band: tuple[float, float],
    policy: SnapshotPolicy,
) -> dict[str, Any]:
    """Bearing evidence per label: analyzer verdict + envelope amplitude.

    The verdict comes from the same analyzers ``check_bearing_faults`` uses
    (catalog route: :func:`check_all_bearing_faults`; declared orders:
    :func:`check_frequency_set`). The envelope amplitude at every expected
    frequency is read with :func:`amplitude_near_frequency` on the SAME
    envelope spectrum the matching consumed, so it exists for every label
    in every snapshot, detected or not.

    Raises:
        ValueError: From the analyzers (empty or non-positive frequencies).
    """
    frequencies: Mapping[str, float] = targets["frequencies"]
    if targets["source"] == "catalog":
        summary = check_all_bearing_faults(
            signal,
            fs,
            bearing_id=targets["bearing_id"],
            rpm=rpm,
            tolerance_pct=policy.tolerance_pct,
            envelope_freq_range=band,
            num_harmonics=policy.num_harmonics,
        )
    else:
        summary = check_frequency_set(
            signal,
            fs,
            frequencies=dict(frequencies),
            rpm=rpm,
            tolerance_pct=policy.tolerance_pct,
            envelope_freq_range=band,
            num_harmonics=policy.num_harmonics,
        )

    env_freqs, env_mags = envelope_spectrum_arrays(signal, fs, band)
    labels: dict[str, dict[str, Any]] = {}
    for check in summary["fault_checks"]:
        label = check["fault_type"]
        expected_hz = float(frequencies[label])
        envelope = amplitude_near_frequency(
            env_freqs, env_mags, expected_hz, policy.tolerance_pct
        )
        labels[label] = {
            "expected_hz": expected_hz,
            "detected": bool(check["detected"]),
            "evidence_strength": check["evidence_strength"],
            "magnitude": check["magnitude"],
            "deviation_pct": check["deviation_pct"],
            "harmonics_detected": [dict(h) for h in check["harmonics_detected"]],
            "envelope_amplitude": float(envelope["amplitude"]),
            "envelope_peak_hz": envelope["frequency_hz"],
        }
    return {
        "source": targets["source"],
        "bearing_id": targets["bearing_id"],
        "shaft_hz": float(targets["shaft_hz"]),
        "catalog_source": targets["catalog_source"],
        "labels": labels,
    }


def compute_health_snapshot(
    signal: np.ndarray,
    fs: float,
    *,
    declaration: Mapping[str, Any],
    point: Optional[Mapping[str, Any]] = None,
    policy: Optional[SnapshotPolicy] = None,
    provenance_overrides: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Compute the derived health snapshot of one measurement.

    Composition rules:

    * ``indicators`` always: rms, peak (``max(|max|, |min|)``), crest
      factor and kurtosis from :func:`extract_time_domain_features`, in the
      DECLARED unit (no conversion here; ``unit`` may be ``None``).
    * ``one_x`` only with a resolved rpm: the largest amplitude within
      ``one_x_tolerance_pct`` of the shaft frequency on the amplitude
      spectrum of the leading ``fft_segment_s`` seconds (the same spectrum
      ``analyze_fft`` reports by default, in the same convention: a
      sinusoid of amplitude ``A`` reads about ``0.54 x A``, the Hamming
      coherent gain left uncorrected on purpose so the value equals the
      tool's peak magnitude; only the trend of the number matters).
    * ``bearing`` only with a resolved rpm AND a catalog designation or
      declared fault orders; per label the analyzer verdict plus the
      envelope amplitude at the expected frequency.
    * ``iso`` only when the unit, the machine group and the support type
      are all declared, via :func:`assess_severity_raw`; the block records
      the DECLARED direction (never the ``"vertical"`` default of the older
      model).
    * Every absent block has ``missing[block] = {reason, remedy}``; a
      ``ValueError`` of the ISO engine or of the bearing catalog becomes
      such an entry, never an exception.

    Args:
        signal: 1-D waveform (at least 3 samples), in the declared unit.
        fs: Sampling frequency (Hz), the rate the numbers are computed at.
        declaration: Effective declaration of the measurement (see
            :func:`resolve_context`).
        point: Latest declaration of the measurement point, or ``None``.
        policy: Snapshot policy; ``None`` means ``SnapshotPolicy()`` resolved
            at call time (so a changed default is visible to the golden
            tripwire, which a definition-time default would hide).
        provenance_overrides: See :func:`collect_provenance`.

    Returns:
        The snapshot dict (the caller adds ``snapshot_id`` and
        ``measurement_id``)::

            {
              "processing": {"processing_id", "algorithm_version", "params",
                             "effective": {"envelope_band_hz",
                                           "fft_segment_samples",
                                           "num_samples", "sampling_rate"},
                             "provenance"},
              "context_digest": str,
              "context": {...},          # resolve_context output
              "indicators": {"rms", "peak", "crest_factor", "kurtosis", "unit"},
              "one_x": {"target_hz", "amplitude", "frequency_hz",
                        "tolerance_pct", "bins_searched", "unit"} | None,
              "bearing": {"source", "bearing_id", "shaft_hz",
                          "catalog_source", "labels": {label: {...}}} | None,
              "iso": {"velocity_rms_mm_s", "zone", "machine_group",
                      "support_type", "direction", "operating_speed_rpm",
                      "machine_power_kw", "frequency_range",
                      "boundaries"} | None,
              "missing": {block: {"reason", "remedy"}},
            }

    Raises:
        ValueError: If the signal (shape, length, NaN/Inf samples) or *fs*
            is unusable, the policy is out of range, or a declared value is
            malformed (see :func:`resolve_context`). These are caller bugs;
            a context that is merely incomplete never raises.
    """
    signal = np.asarray(signal)
    if signal.ndim != 1 or signal.size < 3:
        raise ValueError(
            f"compute_health_snapshot needs a 1-D signal of at least 3 samples, "
            f"got shape {signal.shape}."
        )
    if not np.all(np.isfinite(signal)):
        raise ValueError(
            "compute_health_snapshot needs a finite signal: NaN or Inf samples "
            "found — decode or scale the file correctly and re-load it."
        )
    if not fs > 0:
        raise ValueError(f"compute_health_snapshot needs fs > 0 Hz, got {fs!r}.")
    resolved_policy = SnapshotPolicy() if policy is None else policy
    params = policy_params(resolved_policy)
    context = resolve_context(declaration, point)
    digest = context_digest(context, {**declaration, "sampling_rate": float(fs)})
    missing: dict[str, dict[str, str]] = {}

    # Effective (data-dependent) values: informative, outside the lineage key.
    band_policy = params["envelope_band_policy"]
    requested_band = (
        None
        if band_policy == ENVELOPE_BAND_DEFAULT
        else (float(band_policy[0]), float(band_policy[1]))
    )
    band: Optional[tuple[float, float]]
    band_problem: Optional[str] = None
    try:
        band = resolve_envelope_band(fs, requested_band)
    except ValueError as exc:
        band, band_problem = None, str(exc)
    segment = select_leading_segment(signal, fs, resolved_policy.fft_segment_s)
    effective = {
        "envelope_band_hz": None if band is None else [band[0], band[1]],
        "fft_segment_samples": int(len(segment)),
        "num_samples": int(signal.size),
        "sampling_rate": float(fs),
    }

    # Indicators: always, in the declared unit.
    features = extract_time_domain_features(signal)
    unit = context["signal_unit"]
    indicators: dict[str, Any] = {
        "rms": _finite_or_none(features["rms"]),
        "peak": _finite_or_none(max(abs(features["max"]), abs(features["min"]))),
        "crest_factor": _finite_or_none(features["crest_factor"]),
        "kurtosis": _finite_or_none(features["kurtosis"]),
        "unit": unit,
    }

    # 1x and bearing: only with a shaft speed.
    one_x: Optional[dict[str, Any]] = None
    bearing: Optional[dict[str, Any]] = None
    rpm = context["rpm"]
    if rpm is None:
        reason = (
            "rpm not declared: neither the measurement nor the measurement point "
            "declares a shaft speed"
        )
        missing["one_x"] = _missing(reason, _RPM_REMEDY)
        missing["bearing"] = _missing(reason, _RPM_REMEDY)
    else:
        catalog_problem: Optional[str] = None
        try:
            targets = expected_frequencies(
                rpm,
                bearing_id=context["bearing_id"],
                fault_orders=context["fault_orders"],
            )
        except ValueError as exc:
            catalog_problem = str(exc)
            targets = expected_frequencies(rpm)

        freqs, mags = amplitude_spectrum(segment, fs)
        one_x = amplitude_near_frequency(
            freqs, mags, targets["shaft_hz"], resolved_policy.one_x_tolerance_pct
        )
        one_x["unit"] = unit

        if catalog_problem is not None:
            missing["bearing"] = _missing(
                catalog_problem,
                "re-declare the measurement point via declare_measurement_point "
                "with a catalog designation (search_bearing_catalog) or with "
                "fault_orders",
            )
        elif targets["frequencies"] is None:
            missing["bearing"] = _missing(
                "no bearing declared for the measurement point",
                "declare bearing_id (catalog designation) or fault_orders "
                "(BPFO/BPFI/BSF/FTF as multiples of the shaft frequency) via "
                "declare_measurement_point and re-process",
            )
        elif band is None:
            missing["bearing"] = _missing(
                f"envelope band unusable at fs={fs:g} Hz: {band_problem}",
                "re-acquire at a higher sampling rate or set an explicit "
                "envelope_band_policy in the snapshot policy",
            )
        else:
            try:
                bearing = _bearing_block(
                    signal, fs, rpm, targets, band, resolved_policy
                )
            except ValueError as exc:
                missing["bearing"] = _missing(
                    str(exc),
                    "fix the declared bearing_id or fault_orders via "
                    "declare_measurement_point and re-process",
                )

    # ISO severity: only with unit, group and support declared.
    iso: Optional[dict[str, Any]] = None
    group = context["machine_group"]
    support = context["support_type"]
    if unit is None:
        missing["iso"] = _missing(
            "signal unit not declared",
            'declare "signal_unit" in the companion (or load_signal(signal_unit=...)) '
            "and re-load the file",
        )
    elif group is None or support is None:
        missing["iso"] = _missing(
            "machine group / support type not declared for the measurement point",
            "declare machine_group (1 or 2) and support_type ('rigid' or "
            "'flexible') via declare_measurement_point and re-process",
        )
    else:
        try:
            raw = assess_severity_raw(
                signal,
                fs,
                machine_group=group,
                support_type=support,
                signal_unit=unit,
                operating_speed_rpm=rpm,
                machine_power_kw=context["machine_power_kw"],
            )
        except ValueError as exc:
            missing["iso"] = _missing(
                str(exc),
                "re-declare the measurement point via declare_measurement_point "
                "or re-acquire the signal as the reason says",
            )
        else:
            iso = {
                "velocity_rms_mm_s": float(raw["rms_velocity_mm_s"]),
                "zone": raw["zone"],
                "machine_group": raw["machine_group"],
                "support_type": raw["support_type"],
                "direction": context["direction"],
                "operating_speed_rpm": rpm,
                "machine_power_kw": context["machine_power_kw"],
                "frequency_range": raw["frequency_range"],
                "boundaries": dict(raw["boundaries"]),
            }

    if missing:
        logger.debug("Partial health snapshot: missing %s", sorted(missing))

    return {
        "processing": {
            "processing_id": processing_id(resolved_policy),
            "algorithm_version": ALGORITHM_VERSION,
            "params": params,
            "effective": effective,
            "provenance": collect_provenance(provenance_overrides),
        },
        "context_digest": digest,
        "context": context,
        "indicators": indicators,
        "one_x": one_x,
        "bearing": bearing,
        "iso": iso,
        "missing": missing,
    }


def snapshot_id(measurement_id: str, processing_id: str, context_digest: str) -> str:
    """Deterministic identity of a snapshot: measurement, lineage, context.

    Re-processing the same measurement with the same lineage and the same
    resolved context produces the same id, so the ledger absorbs a retry;
    a corrected rpm or a bumped algorithm produces a new one, and the old
    snapshot stays.

    Args:
        measurement_id: The measurement's identity (file bytes + channel).
        processing_id: Output of :func:`processing_id`.
        context_digest: Output of :func:`context_digest`.

    Returns:
        The first 16 hex digits of the SHA-256 of the three joined by ``:``.

    Raises:
        ValueError: If any component is not a non-empty string.
    """
    parts = {
        "measurement_id": measurement_id,
        "processing_id": processing_id,
        "context_digest": context_digest,
    }
    for name, value in parts.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string, got {value!r}.")
    joined = ":".join((measurement_id, processing_id, context_digest)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:_ID_HEX_CHARS]
