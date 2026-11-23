"""The anomaly-model labels, in one place.

``predict_anomalies`` and the diagnosis pipeline map an anomaly ratio to the
same health label with :func:`health_label`; the PCA report maps the true
label a caller gives a signal to the prediction it expects with
:func:`expected_prediction`. The cut-offs are fixed ratios of flagged windows;
they are not calibrated to a model's own false-alarm level.
"""

from typing import Any

from ..signal_validation import InvalidInputError

#: An anomaly ratio below this is labelled ``Healthy``.
HEALTHY_BELOW = 0.1
#: An anomaly ratio below this, and not below :data:`HEALTHY_BELOW`, is
#: labelled ``Suspicious``; any higher ratio is labelled ``Faulty``.
SUSPICIOUS_BELOW = 0.3

#: True labels the PCA report accepts, compared without regard to case, and
#: the prediction each one expects: +1 for an inlier, -1 for an anomaly.
TRUE_LABELS = {
    "healthy": 1,
    "normal": 1,
    "baseline": 1,
    "faulty": -1,
    "fault": -1,
    "anomaly": -1,
    "anomalous": -1,
}


def health_label(anomaly_ratio: float) -> str:
    """The health label of an anomaly ratio (the share of flagged windows)."""
    if anomaly_ratio < HEALTHY_BELOW:
        return "Healthy"
    if anomaly_ratio < SUSPICIOUS_BELOW:
        return "Suspicious"
    return "Faulty"


def expected_prediction(signal_id: str, label: Any) -> int:
    """The prediction a true label expects for *signal_id*: +1 or -1.

    Raises:
        InvalidInputError: If the label is not one of :data:`TRUE_LABELS`.
    """
    expected = TRUE_LABELS.get(label.lower()) if isinstance(label, str) else None
    if expected is None:
        raise InvalidInputError(
            f"true_labels gives signal {signal_id!r} the label {label!r}, which "
            f"the report does not recognize. Use healthy, normal or baseline for "
            f"a healthy signal, or faulty, fault, anomaly or anomalous for a "
            f"faulty one."
        )
    return expected
