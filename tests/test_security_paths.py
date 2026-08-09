"""Path-traversal regression tests for user-controlled filesystem paths.

Locks in the fixes for three vulnerable classes found in the 2026-07-10 audit:

1. Write side: ``train_anomaly_model`` builds ``MODELS_DIR / f"{model_name}_model.pkl"``
   from an unvalidated ``model_name`` and pickles to it. Reachable from any MCP
   client, in both the modular server and the legacy monolith.
2. Read side: model loading in ``predict_anomalies`` / PCA reports, and
   ``read_report_metadata`` which opens ``REPORTS_DIR / file_name``.
3. Sibling-directory bypass: the first fix (``61627b0``) used ``str.startswith``,
   so ``<base>_evil`` passed containment. Fixed in ``d689886`` via
   ``Path.is_relative_to``; asserted here so it cannot regress.
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest
from mcp.server.mcpserver import MCPServer

from predictive_maintenance_mcp.path_safety import (
    resolve_model_paths,
    safe_resolve,
    sanitize_filename,
    validate_name_component,
)

# Names that must never reach the filesystem as a path component.
TRAVERSAL_NAMES = [
    "../../evil",
    "../models_evil/x",
    "..",
    "subdir/evil",
    "/etc/passwd",
    "C:\\Windows\\Temp\\evil",
    "",
]


@pytest.fixture
def mock_ctx():
    ctx = AsyncMock()
    ctx.info = AsyncMock()
    ctx.warning = AsyncMock()
    return ctx


# ---------------------------------------------------------------------------
# path_safety helpers (unit)
# ---------------------------------------------------------------------------


class TestPathSafetyHelpers:
    def test_safe_resolve_allows_contained(self, tmp_path):
        base = tmp_path / "models"
        base.mkdir()
        resolved = safe_resolve(base, "bearing_health_model.pkl")
        assert resolved == (base / "bearing_health_model.pkl").resolve()
        assert resolved.parent == base.resolve()

    def test_safe_resolve_rejects_parent_traversal(self, tmp_path):
        base = tmp_path / "models"
        base.mkdir()
        with pytest.raises(ValueError):
            safe_resolve(base, "../../etc/passwd")

    def test_safe_resolve_rejects_sibling_directory(self, tmp_path):
        """The d689886 case: /models_evil must not pass /models containment."""
        base = tmp_path / "models"
        base.mkdir()
        (tmp_path / "models_evil").mkdir()
        with pytest.raises(ValueError):
            safe_resolve(base, "../models_evil/x.pkl")

    def test_safe_resolve_rejects_absolute(self, tmp_path):
        base = tmp_path / "models"
        base.mkdir()
        # An absolute path resolves outside base and must be rejected.
        outside = tmp_path / "outside.pkl"
        with pytest.raises(ValueError):
            safe_resolve(base, str(outside))

    def test_sanitize_filename_strips_separators(self):
        assert sanitize_filename("../../evil") == "evil"
        assert sanitize_filename("a/b/c.pkl") == "c.pkl"
        assert sanitize_filename("clean_name-1.json") == "clean_name-1.json"

    def test_validate_name_component_accepts_clean(self):
        assert validate_name_component("bearing_health") == "bearing_health"
        assert validate_name_component("model.v2-final") == "model.v2-final"

    @pytest.mark.parametrize("name", TRAVERSAL_NAMES)
    def test_validate_name_component_rejects(self, name):
        with pytest.raises(ValueError):
            validate_name_component(name, kind="model_name")

    def test_resolve_model_paths_contained(self, tmp_path):
        base = tmp_path / "models"
        base.mkdir()
        paths = resolve_model_paths(base, "bearing_health")
        assert paths._fields == ("model", "scaler", "pca", "metadata")
        for p in paths:
            assert p.parent == base.resolve()
        assert paths.model.name == "bearing_health_model.pkl"
        assert paths.metadata.name == "bearing_health_metadata.json"

    def test_validate_name_component_accepts_traversal_adjacent_but_safe(self):
        # These look risky but resolve to safe single components (they get a
        # `_model.pkl` suffix and stay inside the base) — they must NOT be
        # over-rejected, or legitimate model names would break.
        for name in ["foo..bar", "...", "v1.2.3", "model-final_2026"]:
            assert validate_name_component(name) == name

    @pytest.mark.parametrize("name", TRAVERSAL_NAMES)
    def test_resolve_model_paths_rejects_traversal(self, tmp_path, name):
        base = tmp_path / "models"
        base.mkdir()
        with pytest.raises(ValueError):
            resolve_model_paths(base, name)


# ---------------------------------------------------------------------------
# Modular server — write side (the R1 primary case)
# ---------------------------------------------------------------------------


@pytest.fixture
def models_sandbox(tmp_path, monkeypatch):
    """Isolated MODELS_DIR with a sibling attacker directory alongside it."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (tmp_path / "models_evil").mkdir()
    monkeypatch.setattr(
        "predictive_maintenance_mcp.mcp_tools.diagnostics_tools.MODELS_DIR", models_dir
    )
    return tmp_path, models_dir


@pytest.fixture
def diagnostics_tools(models_sandbox):
    from predictive_maintenance_mcp.mcp_tools.diagnostics_tools import register

    server = MCPServer("test-security")
    register(server)
    return {t.name: t.fn for t in server._tool_manager._tools.values()}


def _no_files_written(tmp_path: Path) -> bool:
    """No pickle/json artifact escaped into any directory under the sandbox."""
    return not list(tmp_path.rglob("*.pkl")) and not list(
        tmp_path.rglob("*_metadata.json")
    )


class TestModularTrainWriteSide:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "name", ["../../evil", "../models_evil/x", "/tmp/evil", ".."]
    )
    async def test_train_rejects_unsafe_model_name(
        self, diagnostics_tools, models_sandbox, mock_ctx, name
    ):
        tmp_path, _ = models_sandbox
        train = diagnostics_tools["train_anomaly_model"]
        with pytest.raises(ValueError):
            await train(healthy_signal_ids=[], model_name=name, ctx=mock_ctx)
        assert _no_files_written(tmp_path), "traversal name must not write any file"


class TestModularReadSide:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", ["../../evil", "../models_evil/x", ".."])
    async def test_predict_rejects_unsafe_model_name(
        self, diagnostics_tools, mock_ctx, name
    ):
        predict = diagnostics_tools["predict_anomalies"]
        with pytest.raises(ValueError):
            await predict(signal_id="whatever", model_name=name, ctx=mock_ctx)


class TestReportMetadataReadSide:
    @pytest.fixture
    def reports_sandbox(self, tmp_path, monkeypatch):
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        (tmp_path / "reports_evil").mkdir()
        secret = tmp_path / "reports_evil" / "secret.html"
        secret.write_text("<html>top secret</html>", encoding="utf-8")
        monkeypatch.setattr(
            "predictive_maintenance_mcp.report_generator.REPORTS_DIR", reports_dir
        )
        return reports_dir, secret

    # Forward-slash and absolute cases traverse on every platform. The backslash
    # case is Windows-only: on POSIX a backslash is a literal filename character,
    # not a separator, so "..\..\x" is a harmless in-directory name there.
    @pytest.mark.parametrize(
        "name", ["../../../etc/passwd", "../reports_evil/secret.html"]
    )
    def test_read_report_metadata_rejects_traversal(self, reports_sandbox, name):
        from predictive_maintenance_mcp.report_generator import read_report_metadata

        with pytest.raises(ValueError, match="Invalid report filename") as exc_info:
            read_report_metadata(name)
        # The oracle is closed: no leak of the sibling file's existence/size
        # (no directory listing in the rejection message).
        assert "secret" not in str(exc_info.value).replace(name, "")
        assert "available" not in str(exc_info.value)

    @pytest.mark.skipif(
        os.name != "nt", reason="backslash is a separator only on Windows"
    )
    def test_read_report_metadata_rejects_windows_backslash(self, reports_sandbox):
        from predictive_maintenance_mcp.report_generator import read_report_metadata

        with pytest.raises(ValueError, match="Invalid report filename") as exc_info:
            read_report_metadata("..\\..\\x")
        assert "available" not in str(exc_info.value)


class TestDefenseInDepthPipeline:
    @pytest.mark.parametrize("name", ["../../evil", "../models_evil/x", ".."])
    def test_run_anomaly_detection_degrades_to_none(self, tmp_path, monkeypatch, name):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        monkeypatch.setattr(
            "predictive_maintenance_mcp.decision_support.diagnosis_pipeline.MODELS_DIR",
            models_dir,
        )
        from predictive_maintenance_mcp.decision_support.diagnosis_pipeline import (
            _run_anomaly_detection,
        )

        # An invalid model name must degrade to None (this function's documented
        # contract), not abort the whole diagnose_vibration pipeline via ValueError.
        assert (
            _run_anomaly_detection(np.zeros(1000), fs=10000.0, model_name=name) is None
        )


# ---------------------------------------------------------------------------
# Modular report_tools PCA report — the LIVE (registered) PCA tool, distinct
# from the monolith copy. This is the production-reachable model-load site.
# ---------------------------------------------------------------------------


class TestModularPCAReportReadSide:
    @pytest.fixture
    def report_tools(self, tmp_path, monkeypatch):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (tmp_path / "models_evil").mkdir()
        monkeypatch.setattr(
            "predictive_maintenance_mcp.mcp_tools.report_tools.MODELS_DIR", models_dir
        )
        from predictive_maintenance_mcp.mcp_tools.report_tools import register

        server = MCPServer("test-security-reports")
        register(server)
        return {t.name: t.fn for t in server._tool_manager._tools.values()}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", ["../../evil", "../models_evil/x", ".."])
    async def test_pca_report_rejects_unsafe_model_name(
        self, report_tools, mock_ctx, name
    ):
        pca_report = report_tools["generate_pca_visualization_report"]
        with pytest.raises(ValueError):
            await pca_report(model_name=name, ctx=mock_ctx)


# ---------------------------------------------------------------------------
# Signal read path — the shared load_signal_data sink behind every analysis
# tool. A traversal filename must not read a file outside DATA_DIR.
# ---------------------------------------------------------------------------


class TestSignalReadContainment:
    @pytest.mark.parametrize(
        "name", ["../../secret.csv", "../outside.csv", "/etc/passwd"]
    )
    def test_load_signal_data_rejects_traversal(self, tmp_path, monkeypatch, name):
        import pandas as pd

        data_dir = tmp_path / "data" / "signals"
        data_dir.mkdir(parents=True)
        # Plant a real, loadable CSV OUTSIDE the data dir that a traversal would hit.
        secret = tmp_path / "secret.csv"
        pd.DataFrame([1.0, 2.0, 3.0]).to_csv(secret, index=False, header=False)
        monkeypatch.setattr(
            "predictive_maintenance_mcp.signal_acquisition.loaders.DATA_DIR", data_dir
        )
        from predictive_maintenance_mcp.signal_acquisition.loaders import (
            load_signal_data,
        )

        # Traversal is contained: the outside file's contents are never returned.
        assert load_signal_data(name) is None

    def test_load_signal_data_allows_contained_relative_path(
        self, tmp_path, monkeypatch
    ):
        import pandas as pd

        data_dir = tmp_path / "data" / "signals"
        (data_dir / "sub").mkdir(parents=True)
        pd.DataFrame([1.0, 2.0, 3.0]).to_csv(
            data_dir / "sub" / "good.csv", index=False, header=False
        )
        monkeypatch.setattr(
            "predictive_maintenance_mcp.signal_acquisition.loaders.DATA_DIR", data_dir
        )
        from predictive_maintenance_mcp.signal_acquisition.loaders import (
            load_signal_data,
        )

        # A legitimate relative path inside DATA_DIR still loads (no over-rejection).
        result = load_signal_data("sub/good.csv")
        assert result is not None
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Raw binary read path — the repository's .bin route runs safe_resolve on
# DATA_DIR-relative names before the decoder ever reads a byte. A traversal
# name must be rejected with a CLOSED oracle (no outside-file contents, no
# directory listing), mirroring TestReportMetadataReadSide.
# ---------------------------------------------------------------------------


class TestRawBinaryReadContainment:
    #: Full raw declaration, so the containment rejection is what fires —
    #: not the missing-declaration refusal (which would mask the traversal).
    RAW_DECL = {"sampling_rate": 10000.0, "sample_format": "float32"}

    @pytest.fixture
    def raw_sandbox(self, tmp_path, monkeypatch):
        """DATA_DIR sandbox with planted outside/sibling .bin attack targets."""
        data_dir = tmp_path / "data" / "signals"
        data_dir.mkdir(parents=True)
        # Decoy INSIDE the data dir: its name must never leak into a
        # rejection message (that would be a directory-listing oracle).
        (data_dir / "inside_decoy.bin").write_bytes(np.zeros(8, dtype="<f4").tobytes())
        # Real files OUTSIDE the data dir that a traversal would hit — the
        # marker must never appear in any error message.
        (tmp_path / "data" / "evil.bin").write_bytes(b"TOPSECRET_OUTSIDE")
        sibling = tmp_path / "data" / "signals_evil"
        sibling.mkdir()
        (sibling / "x.bin").write_bytes(b"TOPSECRET_SIBLING")
        for target in (
            "predictive_maintenance_mcp.signal_acquisition.loaders.DATA_DIR",
            "predictive_maintenance_mcp.signal_acquisition.repository.DATA_DIR",
        ):
            monkeypatch.setattr(target, data_dir)
        from predictive_maintenance_mcp.signal_acquisition.repository import (
            SignalRepository,
        )

        return SignalRepository()

    def _assert_closed_oracle(self, exc_info, name: str) -> None:
        """The rejection reveals nothing beyond the offending input itself.

        NOTE: exists-outside vs nonexistent messages are NOT compared here —
        the exists() -> containment ordering in _prepare_entry is a
        pre-existing differential oracle on all formats, owned by the
        loader-unification follow-up.
        """
        msg = str(exc_info.value)
        assert "TOPSECRET" not in msg  # outside file contents never leak
        assert "available" not in msg  # no directory listing
        assert "inside_decoy" not in msg.replace(name, "")  # no DATA_DIR listing

    # Forward-slash traversal and the sibling directory escape work on every
    # platform; the sibling case is the d689886 class (signals_evil must not
    # pass a naive signals prefix check).
    @pytest.mark.parametrize("name", ["../evil.bin", "../signals_evil/x.bin"])
    def test_raw_load_rejects_traversal(self, raw_sandbox, name):
        with pytest.raises(ValueError) as exc_info:
            raw_sandbox.load_signal(name, **self.RAW_DECL)
        self._assert_closed_oracle(exc_info, name)
        assert raw_sandbox.signal_count == 0  # nothing was stored

    @pytest.mark.skipif(
        os.name != "nt", reason="backslash is a separator only on Windows"
    )
    def test_raw_load_rejects_windows_backslash(self, raw_sandbox):
        name = "..\\evil.bin"
        with pytest.raises(ValueError) as exc_info:
            raw_sandbox.load_signal(name, **self.RAW_DECL)
        self._assert_closed_oracle(exc_info, name)
        assert raw_sandbox.signal_count == 0


# ---------------------------------------------------------------------------
# Single source of truth: the _utils re-exports must be the same objects as
# path_safety, so a future edit cannot silently reintroduce a divergent copy.
# ---------------------------------------------------------------------------


class TestReExportSingleSourceOfTruth:
    def test_utils_reexports_are_identical(self):
        from predictive_maintenance_mcp import path_safety
        from predictive_maintenance_mcp.mcp_tools import _utils

        for name in (
            "safe_resolve",
            "sanitize_filename",
            "validate_name_component",
            "resolve_model_paths",
            "ModelPaths",
        ):
            assert getattr(_utils, name) is getattr(path_safety, name)
