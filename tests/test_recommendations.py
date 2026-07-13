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
        assert "normal" in recs[0]["action"].lower() or "monitoring" in recs[0]["action"].lower()

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

    def test_vocabulary_covers_canonical_bearing_faults(self):
        """VALID_FAULT_TYPES includes every canonical bearing fault."""
        from predictive_maintenance_mcp.decision_support.recommendations import (
            VALID_FAULT_TYPES,
        )
        from predictive_maintenance_mcp.diagnostics.bearing_analyzer import (
            FAULT_TYPE_CANONICAL,
        )

        assert set(FAULT_TYPE_CANONICAL.values()) <= set(VALID_FAULT_TYPES)
