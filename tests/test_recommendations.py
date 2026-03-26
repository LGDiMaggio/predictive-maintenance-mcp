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
            confidence=0.85,
        )

        # 1 zone rec + 2 fault recs
        assert len(recs) == 3
        actions = [r["action"] for r in recs]
        assert any("bearing" in a.lower() for a in actions)
        assert any("align" in a.lower() for a in actions)

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
