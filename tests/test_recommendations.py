"""Tests for decision_support.recommendations — ISO 13374 Block 6."""

import pytest

from predictive_maintenance_mcp.decision_support.recommendations import (
    generate_recommendations,
)


class TestGenerateRecommendations:
    """Tests for the rule-based recommendation engine."""

    def test_zone_a_recommendations(self):
        """Zone A should produce a single low-urgency recommendation."""
        recs = generate_recommendations("A")

        assert len(recs) == 1
        assert recs[0]["urgency"] == "low"
        assert (
            "normal" in recs[0]["action"].lower()
            or "monitoring" in recs[0]["action"].lower()
        )

    def test_zone_d_recommendations(self):
        """Zone D should produce a critical-urgency recommendation."""
        recs = generate_recommendations("D")

        assert len(recs) >= 1
        assert recs[0]["urgency"] == "critical"
        assert "shutdown" in recs[0]["action"].lower()

    def test_with_fault_types(self):
        """Fault types should add bearing-specific recommendations."""
        recs = generate_recommendations(
            "C",
            fault_types=["outer_race", "misalignment"],
        )

        # 1 zone rec + 2 fault recs
        assert len(recs) == 3
        actions = [r["action"] for r in recs]
        assert any("bearing" in a.lower() for a in actions)
        assert any("align" in a.lower() for a in actions)

    def test_confidence_parameter_removed(self):
        """The engine must not accept a caller-dictated confidence."""
        with pytest.raises(TypeError):
            generate_recommendations("C", fault_types=["outer_race"], confidence=0.85)

    def test_no_confidence_in_output_text(self):
        """Recommendation text must not echo any confidence figure."""
        recs = generate_recommendations("C", fault_types=["outer_race"])
        for rec in recs:
            assert "confidence" not in rec["description"].lower()
            assert "confidence" not in rec["action"].lower()

    def test_unknown_zone_fallback(self):
        """Unknown zone should return a fallback recommendation."""
        recs = generate_recommendations("X")

        assert len(recs) == 1
        assert recs[0]["urgency"] == "medium"
        assert "review" in recs[0]["action"].lower()

    def test_empty_fault_types(self):
        """Empty fault_types list should not add extra recommendations."""
        recs = generate_recommendations("B", fault_types=[])
        assert len(recs) == 1
        assert recs[0]["urgency"] == "medium"

    def test_unknown_fault_type_raises_listing_vocabulary(self):
        """U9: unknown fault types raise (the old silent drop hid typos);
        the message names the full canonical vocabulary."""
        with pytest.raises(ValueError) as exc:
            generate_recommendations("C", fault_types=["BPFO"])
        msg = str(exc.value)
        assert "BPFO" in msg
        assert "outer_race" in msg and "looseness" in msg

    def test_caller_supplied_fault_keeps_its_action_and_is_not_called_detected(self):
        """T1 PR C: the action texts are unchanged; the description says the
        fault type came from the caller and is not verified, never that it
        was detected."""
        recs = generate_recommendations("B", fault_types=["outer_race"])

        assert [(r["action"], r["urgency"]) for r in recs] == [
            ("Schedule inspection", "medium"),
            ("Replace bearing, check alignment", "medium"),
        ]
        assert recs[1]["description"] == (
            "Fault type supplied by the caller (outer_race); this tool does not "
            "verify it."
        )
        assert not any("detected" in r["description"].lower() for r in recs)

    def test_zone_d_keeps_its_zone_action_with_a_caller_supplied_fault(self):
        recs = generate_recommendations("D", fault_types=["inner_race"])

        assert recs[0]["action"] == "Immediate shutdown recommended"
        assert recs[0]["urgency"] == "critical"
        assert recs[1]["action"] == "Replace bearing, inspect shaft condition"

    def test_vocabulary_covers_canonical_bearing_faults(self):
        """VALID_FAULT_TYPES includes every canonical bearing fault."""
        from predictive_maintenance_mcp.decision_support.recommendations import (
            VALID_FAULT_TYPES,
        )
        from predictive_maintenance_mcp.diagnostics.bearing_analyzer import (
            FAULT_TYPE_CANONICAL,
        )

        assert set(FAULT_TYPE_CANONICAL.values()) <= set(VALID_FAULT_TYPES)


class TestToolDescription:
    """What a model reads before passing fault types to the tool."""

    def test_fault_types_are_described_as_caller_supplied_and_unverified(self):
        from predictive_maintenance_mcp.mcp_tools.decision_support_tools import (
            generate_maintenance_recommendations,
        )

        doc = " ".join((generate_maintenance_recommendations.__doc__ or "").split())

        assert "supplied by the caller" in doc
        assert "not verified by this tool" in doc
        assert "inspection" in doc
        assert (
            "must not be a type taken from an envelope indication of "
            "check_bearing_faults or diagnose_vibration"
        ) in doc
        assert "Detected fault types" not in doc
