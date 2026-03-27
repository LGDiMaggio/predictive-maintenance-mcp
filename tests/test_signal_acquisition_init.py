"""
Tests for src/signal_acquisition/__init__.py — Block 1 public API.

Verifies that the package exports are importable.
"""


class TestSignalAcquisitionImports:
    """Verify all public names exported from signal_acquisition."""

    def test_import_load_signal_data(self):
        from predictive_maintenance_mcp.signal_acquisition import load_signal_data
        assert callable(load_signal_data)

    def test_import_extract_segment(self):
        from predictive_maintenance_mcp.signal_acquisition import extract_segment
        assert callable(extract_segment)

    def test_import_get_metadata_path(self):
        from predictive_maintenance_mcp.signal_acquisition import get_metadata_path
        assert callable(get_metadata_path)

    def test_import_signal_repository(self):
        from predictive_maintenance_mcp.signal_acquisition import SignalRepository, get_repository
        assert callable(get_repository)

    def test_import_supported_extensions(self):
        from predictive_maintenance_mcp.signal_acquisition import SUPPORTED_EXTENSIONS
        assert isinstance(SUPPORTED_EXTENSIONS, (list, tuple, set, frozenset))
