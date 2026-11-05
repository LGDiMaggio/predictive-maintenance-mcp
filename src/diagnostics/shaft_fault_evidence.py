"""Experimental evidence assessment for unbalance, misalignment and looseness.

Reads the shaft-order measurements of :mod:`.shaft_orders` (one or more
channels of one machine at one operating condition) and reports, for each
fault family, which observations in the records match the pattern this
assessment uses for it (a spectral pattern that practitioner guidance and
part of the literature associate with the family), which observations point
to a competing reading, which measurements are missing, and which other
explanations the records cannot exclude. EXPERIMENTAL: the patterns are not
validated as fault detectors, and no fault is attributed: an observed
pattern is reported as ``inconclusive``, with ``pattern_observed`` true.
No outcome is a diagnosis.

Why so restrained (sources in :data:`SOURCES`):

- No peer-reviewed validation was found for the ratio rules of wall charts
  (2X/1X, axial over radial, horizontal over vertical); a fault-simulator
  study found no consistent misalignment signature (S5) and a review finds
  no universal rule (S6). 2X also comes from cracks, rubs and looseness
  (S6, S10); half orders from looseness, rubs and clearance nonlinearity
  (S12, S14, S15); a 1X-dominant spectrum from unbalance, bow, eccentricity
  and misalignment (S25) and from a resonance near running speed (S1).
- ISO 13373-3 checks overall severity before the signal content and needs a
  reference signal for phase and run-up, coast-down or impact tests for
  resonance (S1, preview only). ISO 20816-3 zones grade broadband severity
  within its scope, never fault identity (S3, preview only).
- Shaft-order lines stand above the noise floor in almost every machine. On
  public test-rig data the patterns were not specific. UPATRAS (Bourdalos and
  Sakellariou, MSSP 238 (2025) 113204; doi:10.17632/42v3s74gf9.1; 2925
  records): the unbalance pattern appeared in 42 % of records of the healthy
  state and in 0 % of the records with a 5 g added unbalance (1.12e-4 kg m),
  where the radial misalignment pattern appeared in 100 %; the radial
  misalignment pattern appeared in 49 % of healthy records; a 0.5X, 1.5X or
  2.5X line appeared in 29 % of healthy records and in 21 % to 34 % of the
  records with loosened foot bolts. COMFAULDA (de Sa So Martins et al.,
  doi:10.21227/89ye-ap56; 2 x 49 healthy records, 49 per fault class): an
  axial 1X or 2X line above every radial channel appeared in 76 % to 100 %
  of healthy records, depending on the bearing. Neither rig is an applicable
  machine with declared units and class (units unverified, machines outside
  the ISO 20816-3 scope), so no step from a pattern to a fault has been
  evaluated on labelled data.

Statuses. ``insufficient_data`` when the lines a pattern needs could not be
tested; ``not_observed`` when the pattern is absent (``pattern_observed``
false; never "free of the fault"); ``inconclusive`` when it is observed
(``pattern_observed`` true), with the reasons: always
``attribution_not_validated``, plus what else stops a reading (confounders,
missing channels, the severity context, conflicting content, the
family-specific limits of misalignment and looseness). Resonance is always
``not_assessable_from_record``. An observed pattern stays useful evidence
even though no fault is attributed. ISO 20816-3 severity, evaluated only on
a declared unit, machine group and support type (never the defaults), is
reported per channel as context and is never evidence of a specific fault.

The pattern definitions (:data:`PATTERN_DEFINITIONS`) use the line test of
:mod:`.shaft_orders` and fixed cut points that were not tuned to data and
are the same for every machine (most of the band power; at least as large;
above every radial channel). They are this assessment's own operational
definitions, not validated thresholds, and the wall-chart ratios (for
example 2X at 50 % of 1X) are not used.
"""

from __future__ import annotations

import copy
from typing import Any, Optional, Sequence

import numpy as np

from ..signal_acquisition.measurement import normalize_direction
from ..signal_validation import InvalidInputError, validate_positive_finite
from .bearing_claims import KNOWN_INAPPLICABLE
from .iso20816 import assess_severity_raw
from .shaft_orders import (
    DEFAULT_MAX_ORDER,
    DEFAULT_SPEED_TOLERANCE_PCT,
    measure_shaft_orders,
)

ASSESSMENT_VERSION = 1
MAX_CHANNELS = 6

UNBALANCE = "unbalance"
MISALIGNMENT = "misalignment"
LOOSENESS = "looseness"
RESONANCE = "resonance"
FAMILIES = (UNBALANCE, MISALIGNMENT, LOOSENESS, RESONANCE)

INSUFFICIENT_DATA = "insufficient_data"
NOT_OBSERVED = "not_observed"
INCONCLUSIVE = "inconclusive"
NOT_ASSESSABLE_FROM_RECORD = "not_assessable_from_record"

RADIAL_DIRECTIONS = ("horizontal", "vertical")
AXIAL_DIRECTION = "axial"
ELEVATED_ZONES = ("C", "D")
#: Lowest machine power in the scope of ISO 20816-3 (kW).
ISO_SCOPE_MIN_POWER_KW = 15.0
#: Harmonics compared with 1X for the radial misalignment pattern (S5, S6:
#: the dominant harmonic under misalignment varies with coupling and speed).
MISALIGNMENT_HARMONICS = (2.0, 3.0, 4.0)
#: Half orders of the looseness pattern: the subharmonic and the low
#: half-order lines reported for looseness (S15: 1/2X; S27, industry
#: guidance: 0.5X, 1.5X, 2.5X). Higher half orders stay reported as lines.
LOOSENESS_HALF_ORDERS = (0.5, 1.5, 2.5)
#: Orders compared between axial and radial channels.
AXIAL_ORDERS = (1.0, 2.0)
#: "Dominated by shaft-rate content": the 1X line carries more than this
#: share of the band's velocity power, i.e. most of it. A fixed, untuned cut
#: point of this assessment, not a validated threshold (no source gives one).
DOMINANT_SHARE = 0.5

#: Sources the assessment rests on. ``access`` says how much was read.
SOURCES: dict[str, dict[str, str]] = {
    "S1": {
        "citation": "ISO 13373-3:2015, Condition monitoring and diagnostics of machines, "
        "Vibration condition monitoring, Part 3: Guidelines for vibration diagnosis",
        "access": "standard preview (clauses 1 to 8, Table A.1)",
        "supports": "severity is checked before signal content; phase needs a reference "
        "signal; resonance needs impact, run-up or coast-down tests; natural frequencies "
        "at rest can differ from those in operation; physical-state changes (unbalance "
        "test, turning the coupling, uncoupled run, alignment check) are listed as "
        "diagnostic tests",
    },
    "S2": {
        "citation": "ISO 13379-1:2025, Data interpretation and diagnostics techniques, "
        "Part 1: General guidelines (second edition)",
        "access": "standard preview (contents, introduction)",
        "supports": "diagnosis identifies the cause of a detected anomaly and carries its "
        "confidence, applicability and limitations",
    },
    "S3": {
        "citation": "ISO 20816-3:2022, Measurement and evaluation of machine vibration, "
        "Part 3: Industrial machinery above 15 kW",
        "access": "standard preview (scope, clause 4); zone limits as implemented in "
        "this server",
        "supports": "zones grade broadband severity of machines above 15 kW, 120 to "
        "30 000 r/min, in steady state; specific frequency components do not always "
        "relate to severity; externally transmitted vibration is excluded; a single "
        "transducer may not capture the maximum in its plane",
    },
    "S5": {
        "citation": "Ganeriwala S., Patel S., Hartung H.A. (1999), The truth behind "
        "misalignment vibration spectra of rotating machinery, Proc. IMAC",
        "access": "full text",
        "supports": "no consistent misalignment signature on one rig; the dominant "
        "harmonic varied with coupling and speed (1X often highest); axial usually but "
        "not always highest; one spectrum at one condition cannot establish misalignment",
    },
    "S6": {
        "citation": "Kumar A., Sathujoda P., Ranjan V. (2021), Vibration characteristics "
        "of a rotor-bearing system caused due to coupling misalignment: a review, "
        "Vibroengineering Procedia 39, doi:10.21595/vp.2021.22195",
        "access": "full text",
        "supports": "2X also from rubs and cracks; 2X can be absent; 3X or 4X can "
        "dominate; no universal rule",
    },
    "S8": {
        "citation": "Patel T.H., Darpe A.K. (2009), Vibration response of misaligned "
        "rotors, Journal of Sound and Vibration 325(3), doi:10.1016/j.jsv.2009.03.024",
        "access": "abstract",
        "supports": "misalignment couples bending, longitudinal and torsional vibration",
    },
    "S10": {
        "citation": "Sinha J.K. (2007), Higher order spectra for crack and misalignment "
        "identification in the shaft of a rotating machine, Structural Health "
        "Monitoring 6(4), doi:10.1177/1475921707082309",
        "access": "abstract",
        "supports": "crack and misalignment both produce 2X and higher harmonics",
    },
    "S12": {
        "citation": "Muszynska A., Goldman P. (1995), Chaotic responses of unbalanced "
        "rotor/bearing/stator systems with looseness or rubs, Chaos, Solitons & "
        "Fractals 5(9), doi:10.1016/0960-0779(94)00171-L",
        "access": "abstract",
        "supports": "an unbalanced rotor with looseness or a rub can produce "
        "subsynchronous fractional responses with higher harmonics",
    },
    "S13": {
        "citation": "Muszynska A. (1986), Whirl and whip: rotor/bearing stability "
        "problems, Journal of Sound and Vibration 110(3)",
        "access": "abstract",
        "supports": "oil whirl is subsynchronous near, not locked at, half speed; whip "
        "locks to a natural frequency",
    },
    "S14": {
        "citation": "Ehrich F.F. (1988), High order subharmonic response of high speed "
        "rotors in bearing clearance, J. Vib. Acoust. 110(1), doi:10.1115/1.3269488",
        "access": "abstract",
        "supports": "bearing clearance nonlinearity can produce subharmonics",
    },
    "S15": {
        "citation": "Lu W., Chu F. (2009), Experimental investigation of pedestal "
        "looseness in a rotor-bearing system, Key Engineering Materials 413-414",
        "access": "abstract",
        "supports": "pedestal looseness showed 1/2X with 2X and 3X, intermittently",
    },
    "S16": {
        "citation": "Ma H., Zhao X., Teng Y., Wen B. (2011), Analysis of dynamic "
        "characteristics for a rotor system with pedestal looseness, Shock and "
        "Vibration 18(1-2), doi:10.3233/SAV-2010-0588",
        "access": "abstract",
        "supports": "the looseness signature depends on foundation stiffness, from "
        "integer harmonics to a near-continuous spectrum",
    },
    "S17": {
        "citation": "Finley W.R., Hodowanec M.M., Holter W.G. (2000), An analytical "
        "approach to solving motor vibration problems, IEEE Trans. Ind. Appl. 36(5), "
        "doi:10.1109/28.871297",
        "access": "abstract",
        "supports": "twice line frequency vibration of two-pole motors lies close to 2X",
    },
    "S25": {
        "citation": "Tranter J. (2018), How do you diagnose unbalance?, CBM Connect "
        "(industry guidance, not peer reviewed)",
        "access": "full text",
        "supports": "unbalance shows high radial 1X; bent shaft, eccentricity and "
        "misalignment can produce the same phase relation; overhung rotors rock axially",
    },
    "S27": {
        "citation": "Wascat B. (2019), 3 types of mechanical looseness, Acoem "
        "(industry guidance, not peer reviewed)",
        "access": "full text",
        "supports": "component-fit looseness: raised floor, many harmonics and half "
        "orders (0.5X, 1.5X, 2.5X ...); the looseness typology is vendor practice",
    },
}

#: Display names of the families inside sentences.
_NAMES = {
    UNBALANCE: "rotor unbalance",
    MISALIGNMENT: "shaft or coupling misalignment",
    LOOSENESS: "mechanical looseness",
    RESONANCE: "resonance",
}

#: Each family's pattern, as the assessment defines it.
PATTERN_DEFINITIONS: dict[str, str] = {
    UNBALANCE: (
        "On a channel with a declared horizontal or vertical direction, the 1X line "
        "carries more than half of the velocity power in 10 Hz to 1 kHz (2 Hz to 1 kHz "
        "below 600 rpm). The overall vibration of the same channel (ISO 20816-3 zone) is "
        "reported as context only. This assessment's own cut point; not taken from a "
        "source and not validated."
    ),
    MISALIGNMENT: (
        "On a radial channel, a 2X, 3X or 4X line at least as large in velocity as the "
        "1X line, or standing above the noise floor where the 1X line does not; or an "
        "axial 1X or 2X line larger than the same line on every radial channel. This "
        "assessment's own cut points; not taken from a source and not validated."
    ),
    LOOSENESS: (
        "A line at 0.5X, 1.5X or 2.5X standing above the local noise floor: one element "
        "of the pattern industry guidance associates with looseness. This assessment's "
        "own definition; not validated."
    ),
    RESONANCE: (
        "Not assessed: a single steady-state record cannot establish or exclude "
        "resonance."
    ),
}

#: Other explanations each pattern cannot exclude. Resonance lists none: it is
#: never assessed, and it is itself an alternative of every other family.
ALTERNATIVES: dict[str, tuple[str, ...]] = {
    UNBALANCE: (
        "rotor bow or a bent shaft",
        "eccentricity of a rotor, pulley or coupling",
        "misalignment",
        "a resonance near running speed",
        "vibration transmitted from another machine",
        "a soft foot or a weak support",
        "electrical vibration at the supply frequency in a two-pole motor",
    ),
    MISALIGNMENT: (
        "unbalance acting through a nonlinear coupling or support",
        "mechanical looseness",
        "a rub",
        "a shaft crack",
        "rotor bow or a bent shaft",
        "unbalance of an overhung rotor (for axial lines)",
        "coupling wear or the coupling's own geometry",
        "a blade, vane or gear tone at an integer order (for example vane pass at 3X)",
        "a resonance near the harmonic",
        "electrical vibration at twice the supply frequency in a motor",
        "vibration transmitted from another machine",
        "signal clipping or sensor saturation",
    ),
    LOOSENESS: (
        "a rub",
        "clearance nonlinearity in the bearings",
        "fluid-film whirl near half speed or whip at a natural frequency in journal "
        "bearings",
        "the cage frequency or another rolling-bearing tone near a half order",
        "a gear or belt tone or another machine running at a nearby speed",
        "beats between close lines",
        "aliasing of a component above half the sampling rate",
    ),
    RESONANCE: (),
}

#: Measurements that would discriminate, per family.
MISSING: dict[str, tuple[str, ...]] = {
    UNBALANCE: (
        "phase between the horizontal and vertical directions with a once-per-revolution "
        "reference",
        "the 1X amplitude at several speeds",
        "comparison with a baseline of the same point at the same speed and load",
        "a trial-weight balancing check",
    ),
    MISALIGNMENT: (
        "axial measurements on both sides of the coupling",
        "phase across the coupling with a once-per-revolution reference",
        "measurements at other speeds and loads",
        "a run with the machines uncoupled, or with the coupling turned 180 degrees",
        "an alignment measurement",
    ),
    LOOSENESS: (
        "measurements in more directions and across the bolted joints",
        "repeated records, because looseness can be intermittent",
        "comparison with a baseline of the same point at the same speed and load",
        "a mechanical inspection of the mountings and fits",
    ),
    RESONANCE: (
        "run-up or coast-down records with a speed reference",
        "an impact test with the machine stopped",
    ),
}

ORTHOGONAL_RADIAL = (
    "a measurement in the orthogonal radial direction at the same bearing (a single "
    "transducer may miss the maximum)"
)
SUPPLY_FREQUENCY = (
    "the electrical supply frequency (the drive output frequency for an inverter-fed "
    "motor)"
)
AXIAL_CHANNEL = "an axial channel"

#: Sources behind each family's pattern, alternatives and missing list.
FAMILY_SOURCES: dict[str, tuple[str, ...]] = {
    UNBALANCE: ("S1", "S2", "S3", "S25"),
    MISALIGNMENT: ("S1", "S2", "S5", "S6", "S8", "S10", "S17", "S25"),
    LOOSENESS: ("S2", "S12", "S13", "S14", "S15", "S16", "S27"),
    RESONANCE: ("S1", "S2"),
}

#: What "not observed" cannot exclude, per family.
_NOT_OBSERVED_CAVEAT = {
    UNBALANCE: (
        "a 1X response also depends on the support and the measurement direction"
    ),
    MISALIGNMENT: (
        "misalignment can show no 2X, can leave 1X as the largest line and can appear "
        "mainly in the axial direction"
    ),
    LOOSENESS: (
        "looseness can be intermittent, or appear only as integer harmonics or as a "
        "raised broadband floor instead of half orders"
    ),
}

#: Why a family is inconclusive or has insufficient data.
REASONS: dict[str, str] = {
    "generated_without_noise": (
        "the record is a generated signal without noise, outside the line test's model"
    ),
    "one_x_untestable": "the 1X line could not be tested on any usable channel",
    "unit_undeclared": (
        "the signal unit is not declared, so velocity amplitudes and the 1X share could "
        "not be computed"
    ),
    "no_half_order_testable": "none of the 0.5X, 1.5X and 2.5X lines could be tested",
    "level_unit_undeclared": (
        "the signal unit is not declared, so overall vibration severity was not evaluated"
    ),
    "level_class_undeclared": (
        "the machine group and support type are not declared, so overall vibration "
        "severity was not evaluated"
    ),
    "level_outside_scope": (
        "the declared machine power is below the 15 kW scope of ISO 20816-3, so its "
        "zones are not a valid context"
    ),
    "level_not_evaluable": (
        "overall vibration severity could not be evaluated on a channel with the pattern "
        "(it needs a declared horizontal or vertical direction, a declared unit and a "
        "sampling rate that covers 1 kHz)"
    ),
    "level_not_elevated": (
        "overall vibration on the channel with the pattern is in ISO 20816-3 zone "
        "{zone}; a zone grades overall vibration and neither establishes nor excludes "
        "a fault"
    ),
    "conflicting_observations": (
        "the records also show content this mechanism alone does not explain"
    ),
    "axial_channel_missing": (
        "no axial channel was assessed, and misalignment can appear mainly in the axial "
        "direction or in other harmonics"
    ),
    "radial_channel_undeclared": (
        "no channel with a declared horizontal or vertical direction was assessed next to "
        "the axial channel"
    ),
    "direction_undeclared": (
        "the measurement direction of the channel with the pattern is not declared as "
        "horizontal, vertical or axial"
    ),
    "shaft_frequency_declared": (
        "the 1x line was not located in the record, so the orders come from the declared "
        "speed and its uncertainty, and nearby lines that are not shaft orders cannot be "
        "separated"
    ),
    "line_frequency_multiple": (
        "a multiple of the electrical supply frequency lies inside a window the pattern "
        "uses, so electrical vibration cannot be separated from it at this resolution"
    ),
    "single_record_insufficient": (
        "records at one operating condition cannot establish misalignment, because the "
        "dominant harmonic and the axial share vary with the coupling, the speed and the "
        "support"
    ),
    "half_orders_not_specific": (
        "half-order lines alone do not single out looseness, because a rub, clearance "
        "nonlinearity, fluid-film whirl or another machine can also produce them"
    ),
    "attribution_not_validated": (
        "the step from this pattern to a fault has not been evaluated on labelled data "
        "from applicable machinery, so no fault is attributed"
    ),
    "speed_changed": (
        "the 1x frequency differs between the two halves of every usable record, so the "
        "speed changed during the capture and the shaft orders are smeared"
    ),
}

SCOPE_STATEMENT = (
    "Experimental evidence assessment: it reports which of the patterns this assessment "
    "uses for unbalance, misalignment and looseness are observed in these records "
    "(pattern_observed), what conflicts with them, what is missing and what else could "
    "explain them. It attributes no fault: an observed pattern is reported as "
    "inconclusive, and 'not_observed' does not mean the machine is free of the fault. "
    "No outcome is a diagnosis; see limitations."
)

#: What every result states about its own reach.
LIMITATIONS: tuple[str, ...] = (
    "A significant line shows a periodic component standing above the local noise "
    "floor, not an abnormality: shaft-order lines are present in almost every machine.",
    "The patterns use fixed cut points of this assessment, not validated thresholds. On "
    "public test-rig data they appeared in a large share of records labelled as the "
    "normal condition, and the unbalance pattern did not appear in records with a small "
    "added unbalance.",
    "No fault is attributed: the step from a pattern to a fault has not been evaluated on "
    "labelled data from applicable machinery, so an observed pattern is reported as "
    "inconclusive.",
    "ISO 20816-3 severity is context: a zone grades overall vibration and identifies no "
    "fault.",
    "The line test's 1 % family-wise false-alarm budget is nominal, not an empirically "
    "demonstrated guarantee. On seeded noise the realized rate was about 0.3 to 0.4 % "
    "without a located 1x line and 1.33 % (95 % interval 0.95 to 1.81 %) for chance "
    "lines next to located 1X and 2X lines, above the budget.",
    "Resonance cannot be established or excluded from a steady-state record.",
)

#: Response views: 'summary' (default) keeps the evidence and the significant lines;
#: 'full' adds every order line, every cross-channel comparison, the full line-test
#: policy and what each source supports.
DETAILS = ("summary", "full")

RESONANCE_STATEMENT = (
    "Resonance can amplify any of these patterns. A single steady-state record cannot "
    "establish or exclude it: that needs run-up or coast-down records, or an impact test "
    "with the machine stopped, and natural frequencies at rest can differ from those in "
    "operation."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_orders(orders: Sequence[float]) -> str:
    return ", ".join(f"{o:g}X" for o in orders)


def _join(items: Sequence[str], word: str = "or") -> str:
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" {word} " + items[-1]


def _line(channel: dict[str, Any], order: float) -> dict[str, Any]:
    """The line of an order that is always measured (max_order >= 2)."""
    found = _line_or_none(channel, order)
    assert found is not None, order
    return found


def _line_or_none(channel: dict[str, Any], order: float) -> Optional[dict[str, Any]]:
    """The line of an order, or None above the channel's max_order."""
    for line in channel["lines"]:
        if line["order"] == order:
            return dict(line)
    return None


def _label(channel: dict[str, Any]) -> str:
    direction = channel["direction"] or "undeclared direction"
    return f"{channel['signal_id']} ({direction})"


def _observation(
    code: str,
    statement: str,
    channels: Sequence[str],
    values: dict[str, Any],
    sources: Sequence[str],
) -> dict[str, Any]:
    return {
        "code": code,
        "statement": statement,
        "channels": list(channels),
        "values": values,
        "sources": list(sources),
    }


def _confounders(channel: dict[str, Any], orders: Sequence[float]) -> list[str]:
    """Measured confounders of a pattern that uses *orders* on *channel*."""
    reasons = []
    if channel["speed"]["source"] != "located_line":
        reasons.append("shaft_frequency_declared")
    lines = [_line_or_none(channel, o) for o in orders if o >= 1.0]
    if any(line and line["line_frequency_multiples"] for line in lines):
        reasons.append("line_frequency_multiple")
    return reasons


def _steady(channels: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Channels whose 1x frequency did not change between the record's halves.

    Order analysis assumes a constant speed over the record; a channel whose
    speed changed is left out of every pattern and of the severity context
    (its lines stay reported).
    """
    return [c for c in channels if not c["stationarity"].get("speed_changed")]


def _comparable(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Same shaft frequency within the two channels' uncertainties."""
    sa, sb = a["speed"], b["speed"]
    gap = abs(float(sa["shaft_hz"]) - float(sb["shaft_hz"]))
    return gap <= float(sa["uncertainty_hz"]) + float(sb["uncertainty_hz"])


def _same_location(a: dict[str, Any], b: dict[str, Any]) -> Optional[bool]:
    """True or False when both channels declare a location; None otherwise."""
    la, lb = a.get("location"), b.get("location")
    if la is None or lb is None:
        return None
    return bool(la == lb)


# ---------------------------------------------------------------------------
# Level context
# ---------------------------------------------------------------------------


def level_context(
    channels: Sequence[dict[str, Any]],
    signals: Sequence[Any],
    machine_group: Optional[int],
    support_type: Optional[str],
    rpm: float,
    machine_power_kw: Optional[float] = None,
) -> dict[str, Any]:
    """Overall vibration severity (ISO 20816-3) per steady radial channel.

    Evaluated only on channels with a declared horizontal or vertical
    direction whose speed did not change, with a declared unit, a declared
    machine group and support type, and a declared power inside the scope
    when one is declared. The zone is context for the patterns, never a
    fault.
    """
    out: dict[str, Any] = {
        "status": "not_assessed",
        "reason": None,
        "zone": None,
        "rms_velocity_mm_s": None,
        "signal_id": None,
        "machine_group": machine_group,
        "support_type": support_type,
        "machine_power_kw": machine_power_kw,
        "zones": [],
        "detail": None,
    }
    if machine_power_kw is not None and machine_power_kw < ISO_SCOPE_MIN_POWER_KW:
        return {**out, "reason": "level_outside_scope"}
    candidates = [
        (c, s)
        for c, s in zip(channels, signals)
        if c["direction"] in RADIAL_DIRECTIONS
        and not c["stationarity"].get("speed_changed")
    ]
    if not candidates:
        return {**out, "reason": "level_not_evaluable"}
    if machine_group is None or support_type is None:
        return {**out, "reason": "level_class_undeclared"}
    zones = []
    detail: Optional[str] = None
    for channel, signal in candidates:
        if channel["unit"]["signal_unit"] is None:
            continue
        try:
            result = assess_severity_raw(
                signal,
                channel["analysis"]["sampling_rate_hz"],
                machine_group=machine_group,  # type: ignore[arg-type]
                support_type=support_type,  # type: ignore[arg-type]
                signal_unit=channel["unit"]["signal_unit"],
                operating_speed_rpm=rpm,
                machine_power_kw=machine_power_kw,
            )
        except ValueError as exc:
            detail = detail or f"{channel['signal_id']}: {exc}"
            continue
        zones.append(
            {
                "signal_id": channel["signal_id"],
                "zone": result["zone"],
                "rms_velocity_mm_s": float(result["rms_velocity_mm_s"]),
            }
        )
    if not zones:
        undeclared = all(c["unit"]["signal_unit"] is None for c, _ in candidates)
        return {
            **out,
            "reason": "level_unit_undeclared" if undeclared else "level_not_evaluable",
            "detail": detail,
        }
    worst = max(zones, key=lambda z: (z["zone"], z["rms_velocity_mm_s"]))
    return {
        **out,
        "status": "assessed",
        "zone": worst["zone"],
        "rms_velocity_mm_s": worst["rms_velocity_mm_s"],
        "signal_id": worst["signal_id"],
        "zones": zones,
        "detail": detail,
    }


def _zone_of(level: dict[str, Any], signal_id: str) -> Optional[str]:
    for z in level["zones"]:
        if z["signal_id"] == signal_id:
            return str(z["zone"])
    return None


def _level_reasons(level: dict[str, Any], signal_ids: Sequence[str]) -> list[str]:
    """Why the severity context does not support a pattern on *signal_ids*."""
    if level["status"] != "assessed":
        return [level["reason"] or "level_not_evaluable"]
    zones = [_zone_of(level, sid) for sid in signal_ids]
    if any(z in ELEVATED_ZONES for z in zones):
        return []
    if all(z is None for z in zones):
        return ["level_not_evaluable"]
    return ["level_not_elevated"]


def _pattern_zone(level: dict[str, Any], signal_ids: Sequence[str]) -> Optional[str]:
    zones = [z for z in (_zone_of(level, sid) for sid in signal_ids) if z is not None]
    return max(zones) if zones else None


# ---------------------------------------------------------------------------
# Cross-channel descriptors
# ---------------------------------------------------------------------------


def cross_channel(channels: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Velocity amplitude ratios between declared directions, descriptors only.

    Axial over radial at 1X and 2X, and horizontal over vertical at 1X, when
    both lines are significant, both units are declared, the channels see the
    same shaft frequency and do not declare different locations. No ratio is
    compared with a threshold here.
    """
    out = []
    pairs: list[tuple[str, dict[str, Any], dict[str, Any], tuple[float, ...]]] = []
    for a in channels:
        for b in channels:
            if a is b:
                continue
            if (
                a["direction"] == AXIAL_DIRECTION
                and b["direction"] in RADIAL_DIRECTIONS
            ):
                pairs.append(("axial_over_radial", a, b, AXIAL_ORDERS))
            if a["direction"] == "horizontal" and b["direction"] == "vertical":
                pairs.append(("horizontal_over_vertical", a, b, (1.0,)))
    for kind, a, b, orders in pairs:
        for order in orders:
            entry: dict[str, Any] = {
                "kind": kind,
                "order": order,
                "numerator_signal_id": a["signal_id"],
                "denominator_signal_id": b["signal_id"],
                "ratio": None,
                "status": "computed",
            }
            la, lb = _line(a, order), _line(b, order)
            if _same_location(a, b) is False:
                entry["status"] = "not_comparable_location"
            elif not _comparable(a, b):
                entry["status"] = "not_comparable_speed"
            elif (
                la["velocity_amplitude_mm_s"] is None
                or lb["velocity_amplitude_mm_s"] is None
            ):
                entry["status"] = "not_computed"
            else:
                entry["ratio"] = (
                    la["velocity_amplitude_mm_s"] / lb["velocity_amplitude_mm_s"]
                )
            out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------


def _result(
    family: str,
    status: str,
    reasons: list[str],
    supporting: list[dict[str, Any]],
    conflicting: list[dict[str, Any]],
    *,
    assessed: Sequence[str] = (),
    missing_extra: Sequence[str] = (),
    zone: Optional[str] = None,
) -> dict[str, Any]:
    reasons = list(dict.fromkeys(reasons))
    alternatives = list(ALTERNATIVES[family])
    name = _NAMES[family]
    reason_text = "; ".join(REASONS[r].format(zone=zone) for r in reasons)
    pattern = "; ".join(o["statement"] for o in supporting)
    where = f"on {_join(list(assessed), 'and')}" if assessed else "in these records"
    if status == INSUFFICIENT_DATA:
        statement = (
            f"Insufficient data for {name}: {reason_text}. This assessment says nothing "
            f"about {name}."
        )
    elif status == NOT_OBSERVED:
        statement = (
            f"The pattern this assessment uses for {name} was not observed {where}. "
            f"This does not exclude {name}: {_NOT_OBSERVED_CAVEAT[family]}."
        )
    else:
        statement = (
            f"The pattern this assessment uses for {name} is observed: {pattern}. Fault "
            f"attribution is inconclusive: {reason_text}. The observed pattern is "
            f"compatible with {name} and equally with each of: "
            f"{'; '.join(alternatives)}."
        )
    return {
        "family": family,
        "status": status,
        "pattern_observed": {INCONCLUSIVE: True, NOT_OBSERVED: False}.get(status),
        "statement": statement,
        "pattern_definition": PATTERN_DEFINITIONS[family],
        "reasons": reasons,
        "supporting": supporting,
        "conflicting": conflicting,
        "missing": list(dict.fromkeys([*missing_extra, *MISSING[family]])),
        "alternatives": alternatives,
        "sources": list(FAMILY_SOURCES[family]),
    }


def _half_orders(channel: dict[str, Any]) -> list[dict[str, Any]]:
    """Significant lines at the half orders of the looseness pattern."""
    return [
        line
        for line in channel["lines"]
        if line["order"] in LOOSENESS_HALF_ORDERS and line["significant"]
    ]


def _context_missing(channels: Sequence[dict[str, Any]]) -> list[str]:
    """Missing measurements that depend on which channels were assessed."""
    missing = []
    directions = {c["direction"] for c in channels}
    if not set(RADIAL_DIRECTIONS) <= directions:
        missing.append(ORTHOGONAL_RADIAL)
    if all(c.get("line_frequency_hz") is None for c in channels):
        missing.append(SUPPLY_FREQUENCY)
    return missing


def _assess_unbalance(
    channels: Sequence[dict[str, Any]], level: dict[str, Any], ineligible: bool
) -> dict[str, Any]:
    if ineligible:
        return _result(
            UNBALANCE, INSUFFICIENT_DATA, ["generated_without_noise"], [], []
        )
    if not _steady(channels):
        return _result(UNBALANCE, INSUFFICIENT_DATA, ["speed_changed"], [], [])
    usable = [c for c in _steady(channels) if c["direction"] != AXIAL_DIRECTION]
    if not usable or not any(_line(c, 1.0)["testable"] for c in usable):
        return _result(UNBALANCE, INSUFFICIENT_DATA, ["one_x_untestable"], [], [])
    assessed = [c for c in usable if c["velocity_band"]["status"] == "computed"]
    if not assessed:
        return _result(UNBALANCE, INSUFFICIENT_DATA, ["unit_undeclared"], [], [])
    missing = _context_missing(channels)

    supporting, per_channel = [], []
    for c in assessed:
        share = c["velocity_band"]["one_x_share"]
        if share is None or not share > DOMINANT_SHARE:
            continue
        band = c["velocity_band"]
        zone = _zone_of(level, c["signal_id"])
        text = (
            f"on {_label(c)} the 1X line carries {share:.1%} of the velocity power in "
            f"{band['low_hz']:g} to {band['high_hz']:g} Hz"
        )
        if zone is not None:
            text += (
                f", and overall vibration on {c['signal_id']} is in ISO 20816-3 zone "
                f"{zone}, context that identifies no fault"
            )
        supporting.append(
            _observation(
                "one_x_dominant",
                text,
                [c["signal_id"]],
                {
                    "one_x_share": share,
                    "band_low_hz": band["low_hz"],
                    "band_high_hz": band["high_hz"],
                    "one_x_velocity_amplitude_mm_s": _line(c, 1.0)[
                        "velocity_amplitude_mm_s"
                    ],
                    "iso_zone": zone,
                },
                ("S25",),
            )
        )
        reasons = (
            [] if c["direction"] in RADIAL_DIRECTIONS else ["direction_undeclared"]
        )
        reasons += _confounders(c, (1.0,))
        reasons += _level_reasons(level, [c["signal_id"]])
        per_channel.append((c, text, zone, reasons))
    ids = [c["signal_id"] for c in assessed]
    if not supporting:
        return _result(
            UNBALANCE, NOT_OBSERVED, [], [], [], assessed=ids, missing_extra=missing
        )

    conflicting = []
    for c in usable:
        half = _half_orders(c)
        if half:
            orders = [h["order"] for h in half]
            conflicting.append(
                _observation(
                    "half_orders_present",
                    f"on {_label(c)} lines at half orders ({_fmt_orders(orders)}) stand "
                    f"above the local noise floor; unbalance alone does not explain "
                    f"half-order content, which looseness, a rub or clearance "
                    f"nonlinearity can produce",
                    [c["signal_id"]],
                    {"orders": orders},
                    ("S12", "S14", "S15"),
                )
            )
    reasons = [r for p in per_channel for r in p[3]]
    if conflicting:
        reasons.append("conflicting_observations")
    reasons.append("attribution_not_validated")
    return _result(
        UNBALANCE,
        INCONCLUSIVE,
        reasons,
        supporting,
        conflicting,
        assessed=ids,
        missing_extra=missing,
        zone=_pattern_zone(level, [p[0]["signal_id"] for p in per_channel]),
    )


def _radial_harmonics(c: dict[str, Any]) -> list[dict[str, Any]]:
    """The radial misalignment observations of one channel."""
    out: list[dict[str, Any]] = []
    one = _line(c, 1.0)
    if not one["testable"]:
        return out
    v1 = one["velocity_amplitude_mm_s"]
    for order in MISALIGNMENT_HARMONICS:
        harmonic = _line_or_none(c, order)
        vk = None if harmonic is None else harmonic["velocity_amplitude_mm_s"]
        if vk is None:
            continue
        if one["significant"] and v1 is not None and vk >= v1:
            out.append(
                _observation(
                    "harmonic_at_least_one_x",
                    f"on {_label(c)} the {order:g}X line has {vk / v1:.2f} times the "
                    f"velocity amplitude of the 1X line",
                    [c["signal_id"]],
                    {"order": order, "ratio_to_one_x": vk / v1},
                    ("S5", "S6"),
                )
            )
        elif not one["significant"]:
            out.append(
                _observation(
                    "harmonic_above_floor_without_one_x",
                    f"on {_label(c)} the {order:g}X line stands above the local noise "
                    f"floor while the 1X line does not",
                    [c["signal_id"]],
                    {"order": order, "ratio_to_one_x": None},
                    ("S5", "S6"),
                )
            )
    return out


def _axial_exceeds(
    a: dict[str, Any], radial: Sequence[dict[str, Any]], order: float
) -> Optional[dict[str, Any]]:
    """The observation when axial *order* exceeds it on every radial channel."""
    la = _line(a, order)
    va = la["velocity_amplitude_mm_s"]
    if va is None or not radial:
        return None
    ratios: list[tuple[Optional[float], dict[str, Any]]] = []
    assumed = False
    for r in radial:
        same = _same_location(a, r)
        if same is False or not _comparable(a, r):
            return None
        assumed = assumed or same is None
        lr = _line(r, order)
        vr = lr["velocity_amplitude_mm_s"]
        if vr is not None:
            if not va > vr:
                return None
            ratios.append((va / vr, r))
        elif lr["testable"] and not lr["significant"]:
            ratios.append((None, r))
        else:
            return None
    numeric = [x for x in ratios if x[0] is not None]
    caveat = ", assuming the channels are at the same bearing" if assumed else ""
    if numeric:
        ratio, r = min(numeric, key=lambda x: x[0])
        text = (
            f"at {order:g}X the axial channel {a['signal_id']} has {ratio:.2f} times the "
            f"velocity amplitude of the largest radial channel ({r['signal_id']}){caveat}"
        )
        values = {"order": order, "ratio": ratio}
    else:
        text = (
            f"at {order:g}X the axial channel {a['signal_id']} has a line above the noise "
            f"floor where no radial channel has one{caveat}"
        )
        values = {"order": order, "ratio": None}
    return _observation(
        "axial_exceeds_radial",
        text,
        [a["signal_id"], *(r["signal_id"] for r in radial)],
        values,
        ("S5", "S8"),
    )


def _assess_misalignment(
    channels: Sequence[dict[str, Any]], level: dict[str, Any], ineligible: bool
) -> dict[str, Any]:
    if ineligible:
        return _result(
            MISALIGNMENT, INSUFFICIENT_DATA, ["generated_without_noise"], [], []
        )
    if not _steady(channels):
        return _result(MISALIGNMENT, INSUFFICIENT_DATA, ["speed_changed"], [], [])
    steady = _steady(channels)
    if not any(_line(c, 1.0)["testable"] for c in steady):
        return _result(MISALIGNMENT, INSUFFICIENT_DATA, ["one_x_untestable"], [], [])
    with_unit = [c for c in steady if c["unit"]["signal_unit"] is not None]
    if not with_unit:
        return _result(MISALIGNMENT, INSUFFICIENT_DATA, ["unit_undeclared"], [], [])

    axial = [c for c in with_unit if c["direction"] == AXIAL_DIRECTION]
    radial = [c for c in with_unit if c["direction"] in RADIAL_DIRECTIONS]
    missing = ([] if axial else [AXIAL_CHANNEL]) + _context_missing(channels)
    supporting, reasons = [], []
    for c in with_unit:
        if c["direction"] == AXIAL_DIRECTION:
            continue
        found = _radial_harmonics(c)
        supporting += found
        for o in found:
            if c["direction"] not in RADIAL_DIRECTIONS:
                reasons.append("direction_undeclared")
            reasons += _confounders(c, (1.0, o["values"]["order"]))
    for a in axial:
        for order in AXIAL_ORDERS:
            found_axial = _axial_exceeds(a, radial, order)
            if found_axial is not None:
                supporting.append(found_axial)
                reasons += _confounders(a, (order,))
    ids = [c["signal_id"] for c in with_unit]
    if not supporting:
        return _result(
            MISALIGNMENT, NOT_OBSERVED, [], [], [], assessed=ids, missing_extra=missing
        )
    if not axial:
        reasons.append("axial_channel_missing")
    elif not radial:
        reasons.append("radial_channel_undeclared")
    # Not attributable from one record (S5): the dominant harmonic and the
    # axial share vary
    # with coupling, speed and support, and on a public test rig (COMFAULDA)
    # an axial line above every radial one appeared in most healthy records.
    pattern_ids = [sid for o in supporting for sid in o["channels"]]
    reasons += _level_reasons(level, pattern_ids)
    reasons += ["single_record_insufficient", "attribution_not_validated"]
    return _result(
        MISALIGNMENT,
        INCONCLUSIVE,
        reasons,
        supporting,
        [],
        assessed=ids,
        missing_extra=missing,
        zone=_pattern_zone(level, pattern_ids),
    )


def _assess_looseness(
    channels: Sequence[dict[str, Any]], level: dict[str, Any], ineligible: bool
) -> dict[str, Any]:
    if ineligible:
        return _result(
            LOOSENESS, INSUFFICIENT_DATA, ["generated_without_noise"], [], []
        )
    if not _steady(channels):
        return _result(LOOSENESS, INSUFFICIENT_DATA, ["speed_changed"], [], [])
    testable = [
        c
        for c in _steady(channels)
        if any(
            line["testable"]
            for line in c["lines"]
            if line["order"] in LOOSENESS_HALF_ORDERS
        )
    ]
    if not testable:
        return _result(LOOSENESS, INSUFFICIENT_DATA, ["no_half_order_testable"], [], [])

    supporting, reasons = [], []
    for c in testable:
        half = _half_orders(c)
        if not half:
            continue
        orders = [h["order"] for h in half]
        levels = ", ".join(f"{h['level_db']:.1f}" for h in half)
        supporting.append(
            _observation(
                "half_order_lines",
                f"half-order content, one element of the pattern industry guidance "
                f"associates with looseness: on {_label(c)} lines at "
                f"{_fmt_orders(orders)} stand above the local noise floor (levels "
                f"{levels} dB over the floor)",
                [c["signal_id"]],
                {"orders": orders, "level_db": [h["level_db"] for h in half]},
                ("S12", "S15", "S27"),
            )
        )
        if c["direction"] not in (*RADIAL_DIRECTIONS, AXIAL_DIRECTION):
            reasons.append("direction_undeclared")
        reasons += _confounders(c, ())
    ids = [c["signal_id"] for c in testable]
    if not supporting:
        return _result(LOOSENESS, NOT_OBSERVED, [], [], [], assessed=ids)
    # Half orders are not specific to looseness (see the
    # reason); on a public test rig a 0.5X, 1.5X or 2.5X line appeared in
    # healthy records about as often as in records with loosened bolts.
    pattern_ids = [o["channels"][0] for o in supporting]
    reasons += _level_reasons(level, pattern_ids)
    reasons += ["half_orders_not_specific", "attribution_not_validated"]
    return _result(
        LOOSENESS,
        INCONCLUSIVE,
        reasons,
        supporting,
        [],
        assessed=ids,
        zone=_pattern_zone(level, pattern_ids),
    )


def _resonance() -> dict[str, Any]:
    return {
        "family": RESONANCE,
        "status": NOT_ASSESSABLE_FROM_RECORD,
        "pattern_observed": None,
        "statement": RESONANCE_STATEMENT,
        "pattern_definition": PATTERN_DEFINITIONS[RESONANCE],
        "reasons": [],
        "supporting": [],
        "conflicting": [],
        "missing": list(MISSING[RESONANCE]),
        "alternatives": [],
        "sources": list(FAMILY_SOURCES[RESONANCE]),
    }


def _summary(hypotheses: Sequence[dict[str, Any]]) -> str:
    by_status: dict[str, list[str]] = {}
    for h in hypotheses:
        if h["family"] == RESONANCE:
            continue
        by_status.setdefault(h["status"], []).append(_NAMES[h["family"]])
    parts = []
    for status, label, tail in (
        (
            INCONCLUSIVE,
            "pattern observed for",
            " (fault attribution inconclusive)",
        ),
        (NOT_OBSERVED, "pattern not observed for", " (which does not exclude it)"),
        (INSUFFICIENT_DATA, "insufficient data for", ""),
    ):
        if status in by_status:
            parts.append(f"{label} {_join(by_status[status], 'and')}{tail}")
    return (
        "Experimental shaft-order evidence: "
        + "; ".join(parts)
        + ". Resonance is not assessable from a steady-state record. No outcome is a "
        "diagnosis."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def assess_shaft_fault_evidence(
    channels: Sequence[dict[str, Any]],
    rpm: float,
    *,
    machine_group: Optional[int] = None,
    support_type: Optional[str] = None,
    machine_power_kw: Optional[float] = None,
    speed_tolerance_pct: float = DEFAULT_SPEED_TOLERANCE_PCT,
    max_order: int = DEFAULT_MAX_ORDER,
    line_frequency_hz: Optional[float] = None,
) -> dict[str, Any]:
    """Measure every channel and assess the evidence for each fault family.

    Args:
        channels: One dict per channel of one machine at one operating
            condition: ``signal`` (1-D samples), ``fs`` (Hz), ``signal_id``,
            ``signal_unit`` (declared or None), ``direction`` ('horizontal',
            'vertical', 'axial', 'x', 'y', 'z' or None) and optionally
            ``location`` (a measurement-point identifier; channels declaring
            different locations are never compared) and
            ``acquisition_eligibility`` ('known_inapplicable' for a generated
            signal without noise).
        rpm: Declared shaft speed.
        machine_group: Declared ISO 20816-3 machine group (1 or 2), or None.
        support_type: Declared support type ('rigid' or 'flexible'), or None.
        machine_power_kw: Declared machine power in kW, or None (unknown).
        speed_tolerance_pct: Declared speed uncertainty in percent.
        max_order: Highest integer order measured.
        line_frequency_hz: Declared electrical supply frequency, or None.

    Returns:
        ``channels`` (the measurements), ``cross_channel`` descriptors,
        ``level_context``, ``hypotheses`` (one per family), ``summary``,
        ``scope_statement``, ``experimental`` (True), ``assessment_version``
        and ``sources``.

    Raises:
        InvalidInputError: For no channel, too many channels, duplicate
            signal ids, or any invalid sample or parameter.
    """
    if not channels:
        raise InvalidInputError(
            "At least one channel is needed — pass the signal ids of the channels to assess."
        )
    if len(channels) > MAX_CHANNELS:
        raise InvalidInputError(
            f"At most {MAX_CHANNELS} channels can be assessed together, got "
            f"{len(channels)} — pass the channels of one machine at one operating condition."
        )
    ids = [c["signal_id"] for c in channels]
    if len(set(ids)) != len(ids):
        raise InvalidInputError(
            f"Each channel must be a different signal, got {ids} — remove the repeated ids."
        )
    if machine_power_kw is not None:
        machine_power_kw = validate_positive_finite(
            machine_power_kw,
            name="machine_power_kw",
            remedy="declare the rated machine power in kW, or omit it when unknown",
        )

    directions = []
    for c in channels:
        direction = c.get("direction")
        if direction is not None:
            try:
                direction = normalize_direction(direction)
            except ValueError as exc:
                raise InvalidInputError(f"Channel {c['signal_id']!r}: {exc}") from exc
        directions.append(direction)

    measured = []
    for c, direction in zip(channels, directions):
        m = measure_shaft_orders(
            c["signal"],
            c["fs"],
            rpm,
            signal_unit=c.get("signal_unit"),
            speed_tolerance_pct=speed_tolerance_pct,
            max_order=max_order,
            line_frequency_hz=line_frequency_hz,
        )
        m["signal_id"] = c["signal_id"]
        m["direction"] = direction
        m["location"] = c.get("location")
        measured.append(m)

    policy = measured[0]["policy"]
    for m in measured:
        m.pop("policy", None)

    ineligible = any(
        c.get("acquisition_eligibility") == KNOWN_INAPPLICABLE for c in channels
    )
    level = level_context(
        measured,
        # The samples the lines were measured on (bounded length).
        [
            np.asarray(c["signal"], dtype=float)[: m["analysis"]["samples"]]
            for c, m in zip(channels, measured)
        ],
        machine_group,
        support_type,
        float(rpm),
        machine_power_kw,
    )
    hypotheses = [
        _assess_unbalance(measured, level, ineligible),
        _assess_misalignment(measured, level, ineligible),
        _assess_looseness(measured, level, ineligible),
        _resonance(),
    ]
    return {
        "experimental": True,
        "assessment_version": ASSESSMENT_VERSION,
        "scope_statement": SCOPE_STATEMENT,
        "limitations": list(LIMITATIONS),
        "summary": _summary(hypotheses),
        "channels": measured,
        "cross_channel": cross_channel(measured),
        "level_context": level,
        "hypotheses": hypotheses,
        "order_line_policy": policy,
        "sources": [{"key": k, **v} for k, v in SOURCES.items()],
    }


def with_detail(result: dict[str, Any], detail: str) -> dict[str, Any]:
    """The result in the requested view, deterministically.

    'full' keeps everything. 'summary' keeps every hypothesis (status,
    pattern_observed, statement, reasons, supporting and conflicting
    observations with their values, missing measurements, alternatives),
    the severity context, the limitations and, per channel, the speed, the
    band, the screening statistics and the significant lines; it leaves out
    the lines that are not significant (their orders stay listed), the
    cross-channel comparisons that could not be computed, the long policy
    texts and what each source supports. Both views list every channel's
    untestable orders.

    Raises:
        InvalidInputError: If *detail* is not 'summary' or 'full'.
    """
    if detail not in DETAILS:
        raise InvalidInputError(
            f"detail must be 'summary' or 'full', got {detail!r} — pass 'full' for "
            f"every order line and the full policy and source texts."
        )
    out = copy.deepcopy(result)
    out["detail"] = detail
    for channel in out["channels"]:
        channel["untestable_orders"] = [
            line["order"] for line in channel["lines"] if not line["testable"]
        ]
    if detail == "full":
        return out
    for channel in out["channels"]:
        channel["lines"] = [line for line in channel["lines"] if line["significant"]]
    out["cross_channel"] = [
        x for x in out["cross_channel"] if x["status"] == "computed"
    ]
    policy = out["order_line_policy"]
    out["order_line_policy"] = {
        key: policy[key]
        for key in (
            "method",
            "version",
            "false_alarm_probability_per_call",
            "measured_false_alarm_rates",
        )
    }
    for source in out["sources"]:
        source["supports"] = None
    return out
