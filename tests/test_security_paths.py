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
import time
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest
from mcp.server.mcpserver import MCPServer

from predictive_maintenance_mcp import config, path_safety
from predictive_maintenance_mcp.path_safety import (
    _without_extended_prefix,
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

    @pytest.mark.skipif(os.name != "nt", reason="Windows extended-length prefix")
    def test_safe_resolve_tolerates_extended_length_prefix(self, tmp_path, monkeypatch):
        """A ``\\\\?\\``-prefixed resolution is still contained.

        ``Path.resolve`` keeps the Win32 extended-length prefix when the
        prefix-stripping check it performs hits a transient sharing violation
        (another process appending to the file, an antivirus scan). The prefix
        is a namespace marker, not a different location, so containment must
        not report a false escape.
        """
        base = tmp_path / "ledger"
        base.mkdir()
        target = base / "P-101.jsonl"
        target.write_text("x")
        real_resolve = Path.resolve
        plain = str(real_resolve(target))

        def prefixed_resolve(self, strict=False):
            resolved = real_resolve(self, strict=strict)
            if self.name == "P-101.jsonl":
                return Path("\\\\?\\" + str(resolved))
            return resolved

        monkeypatch.setattr(Path, "resolve", prefixed_resolve)
        assert str(safe_resolve(base, "P-101.jsonl")) == plain
        with pytest.raises(ValueError):
            safe_resolve(base, "../P-101.jsonl")

    @pytest.mark.skipif(os.name != "nt", reason="Windows extended-length prefix")
    def test_safe_resolve_tolerates_extended_length_unc_prefix(
        self, tmp_path, monkeypatch
    ):
        """The UNC form of the prefix, ``\\\\?\\UNC\\server\\share\\...``, is
        the plain ``\\\\server\\share\\...`` location.

        Same transient as the drive-letter case, on a network share: the
        base resolves plain while the candidate keeps the prefix. The
        result must be the plain UNC path, and a traversal on the share is
        still refused.
        """
        base = tmp_path / "ledger"
        base.mkdir()
        (base / "P-101.jsonl").write_text("x")
        real_resolve = Path.resolve
        root = str(real_resolve(tmp_path))

        def unc_resolve(self, strict=False):
            tail = str(real_resolve(self, strict=strict))[len(root) :]
            if self.name == "P-101.jsonl":
                return Path("\\\\?\\UNC\\server\\share" + tail)
            return Path("\\\\server\\share" + tail)

        monkeypatch.setattr(Path, "resolve", unc_resolve)
        resolved = safe_resolve(base, "P-101.jsonl")
        assert str(resolved) == "\\\\server\\share\\ledger\\P-101.jsonl"
        assert not str(resolved).startswith("\\\\?\\")
        with pytest.raises(ValueError):
            safe_resolve(base, "../P-101.jsonl")

    @pytest.mark.parametrize(
        ("prefixed", "plain"),
        [
            ("\\\\?\\C:\\data\\ledger\\P-101.jsonl", "C:\\data\\ledger\\P-101.jsonl"),
            (
                "\\\\?\\UNC\\server\\share\\ledger\\P-101.jsonl",
                "\\\\server\\share\\ledger\\P-101.jsonl",
            ),
            ("C:\\data\\ledger\\P-101.jsonl", "C:\\data\\ledger\\P-101.jsonl"),
        ],
    )
    def test_without_extended_prefix_strips_both_prefix_forms(self, prefixed, plain):
        """The prefix strip is pure string work, so both forms are checked on
        every platform even though only Windows resolves paths with the prefix."""
        assert str(_without_extended_prefix(Path(prefixed))) == str(Path(plain))

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


# ---------------------------------------------------------------------------
# Import paths (Signal Import Engine, U3): the lexical gate runs before any
# operating-system call, the roots come from the data directory and
# PMM_IMPORT_ROOTS, and every read goes through one verified descriptor.
# ---------------------------------------------------------------------------

WIN_ROOT = "C:\\pmm\\data"
POSIX_ROOT = "/srv/pmm/data"


@pytest.fixture
def no_filesystem_calls(monkeypatch):
    """Every call that could reach a disk, a share or a device fails the
    test; the list records which was attempted."""
    attempted: list[str] = []

    def trap(name):
        def call(*args, **kwargs):
            attempted.append(name)
            raise AssertionError(f"{name} reached the operating system")

        return call

    for name in ("stat", "lstat", "open", "scandir", "listdir", "readlink"):
        monkeypatch.setattr(os, name, trap(f"os.{name}"))
    monkeypatch.setattr(os.path, "realpath", trap("os.path.realpath"))
    monkeypatch.setattr(Path, "resolve", trap("Path.resolve"))
    monkeypatch.setattr(Path, "stat", trap("Path.stat"))
    return attempted


class TestImportPathGate:
    @pytest.mark.parametrize(
        "raw",
        [
            "\\\\server\\share\\x.csv",
            "//server/share/x.csv",
            "\\\\?\\C:\\pmm\\data\\x.csv",
            "\\\\?\\UNC\\server\\share\\x.csv",
            "\\\\.\\C:\\pmm\\data\\x.csv",
            "\\??\\C:\\pmm\\data\\x.csv",
            "C:\\pmm\\data\\x.csv:hidden",
            "C:\\pmm\\data\\x.csv::$DATA",
        ],
    )
    def test_network_device_and_stream_forms_make_no_os_call(
        self, raw, no_filesystem_calls
    ):
        start = time.perf_counter()
        with pytest.raises(path_safety.PathRefused) as caught:
            path_safety.gate_import_path(raw, [WIN_ROOT], base=WIN_ROOT, windows=True)
        assert time.perf_counter() - start < 0.05
        assert caught.value.code == "unsafe_path"
        assert no_filesystem_calls == []

    @pytest.mark.skipif(os.name != "nt", reason="the host's own path grammar")
    def test_the_full_open_refuses_a_unc_path_before_any_os_call(
        self, no_filesystem_calls, tmp_path
    ):
        with pytest.raises(path_safety.PathRefused) as caught:
            path_safety.open_import_file(
                "\\\\203.0.113.9\\share\\x.csv",
                [tmp_path],
                base=tmp_path,
                max_size=1 << 30,
            )
        assert caught.value.code == "unsafe_path"
        assert no_filesystem_calls == []

    @pytest.mark.parametrize(
        ("raw", "problem"),
        [
            ("C:\\pmm\\data\\x.csv.", "ends with a dot or a space"),
            ("C:\\pmm\\data\\x.csv ", "ends with a dot or a space"),
            ("C:\\pmm\\data\\NUL", "device name"),
            ("C:\\pmm\\data\\com1.csv", "device name"),
            ("C:\\pmm\\data\\CON .csv", "device name"),
            ("C:\\pmm\\data\\COM" + chr(0xB9) + ".csv", "device name"),
            ("C:\\pmm\\data\\conin$.csv", "device name"),
            ("C:x.csv", "drive-relative"),
            ("\\pmm\\data\\x.csv", "root-relative"),
            ("C:\\pmm\\data\\x" + chr(0x202E) + "vsc.csv", "invisible formatting"),
            ("C:\\pmm\\data\\x\n.csv", "control"),
            ("C:\\pmm\\data\\x?.csv", "reserves in names"),
            ('C:\\pmm\\data\\"x".csv', "reserves in names"),
            ("C:\\pmm\\data\\x|y.csv", "reserves in names"),
            ("C:\\pmm\\data\\" + "a" * 256 + ".csv", "over 255 characters"),
        ],
    )
    def test_windows_aliases_and_devices_are_refused(self, raw, problem):
        with pytest.raises(path_safety.PathRefused, match=problem) as caught:
            path_safety.gate_import_path(raw, [WIN_ROOT], base=WIN_ROOT, windows=True)
        assert caught.value.code == "unsafe_path"
        assert chr(0x202E) not in str(caught.value)

    @pytest.mark.parametrize(
        "raw",
        [
            "C:\\pmm\\data_evil\\x.csv",
            "C:\\pmm\\x.csv",
            "C:\\pmm\\data\\..\\x.csv",
            "D:\\pmm\\data\\x.csv",
        ],
    )
    def test_paths_outside_every_root_are_refused(self, raw):
        with pytest.raises(path_safety.PathRefused) as caught:
            path_safety.gate_import_path(raw, [WIN_ROOT], base=WIN_ROOT, windows=True)
        assert caught.value.code == "path_outside_roots"
        assert "PMM_IMPORT_ROOTS" in str(caught.value)

    def test_windows_containment_ignores_letter_case_and_slash_direction(self):
        gated = path_safety.gate_import_path(
            "c:/PMM/Data/sub/x.CSV", [WIN_ROOT], base=WIN_ROOT, windows=True
        )
        assert gated.root == WIN_ROOT
        assert gated.relative == "sub\\x.CSV"

    def test_relative_paths_join_the_base_and_the_most_specific_root_wins(self):
        nested = WIN_ROOT + "\\plant"
        gated = path_safety.gate_import_path(
            "plant\\pump\\x.csv", [WIN_ROOT, nested], base=WIN_ROOT, windows=True
        )
        assert gated.path == "C:\\pmm\\data\\plant\\pump\\x.csv"
        assert gated.root == nested
        assert gated.relative == "pump\\x.csv"

    @pytest.mark.parametrize(
        ("raw", "code"),
        [
            ("//host/x.csv", "unsafe_path"),
            ("/srv/pmm/data/x.csv.", "unsafe_path"),
            ("/srv/pmm/data/aux.csv", "unsafe_path"),
            ("/srv/pmm/data/../../../etc/passwd.csv", "path_outside_roots"),
            ("/srv/pmm/data_evil/x.csv", "path_outside_roots"),
        ],
    )
    def test_posix_grammar(self, raw, code):
        with pytest.raises(path_safety.PathRefused) as caught:
            path_safety.gate_import_path(
                raw, [POSIX_ROOT], base=POSIX_ROOT, windows=False
            )
        assert caught.value.code == code

    def test_posix_allows_a_colon_and_a_relative_path(self):
        gated = path_safety.gate_import_path(
            "runs/12:30.csv", [POSIX_ROOT], base=POSIX_ROOT, windows=False
        )
        assert gated.path == "/srv/pmm/data/runs/12:30.csv"

    @pytest.mark.parametrize("raw", [None, "", "   ", 5, "x" * 5000])
    def test_non_paths_are_refused(self, raw):
        with pytest.raises(path_safety.PathRefused) as caught:
            path_safety.gate_import_path(raw, [WIN_ROOT], base=WIN_ROOT, windows=True)
        assert caught.value.code == "unsafe_path"

    def test_device_names_cover_the_measurement_ledger_names(self):
        from predictive_maintenance_mcp.signal_acquisition.measurement import (
            WIN32_RESERVED_NAMES,
        )

        assert WIN32_RESERVED_NAMES <= path_safety.WINDOWS_DEVICE_NAMES

    def test_every_refusal_code_is_an_issue_code(self):
        import ast

        from predictive_maintenance_mcp.signal_acquisition import import_config

        sources = [
            Path(path_safety.__file__),
            Path(import_config.__file__).parent / "text_scan.py",
            Path(import_config.__file__).parent / "text_decoder.py",
        ]
        codes = set()
        for source in sources:
            for node in ast.walk(ast.parse(source.read_text("utf-8"))):
                if isinstance(node, ast.Call) and node.args:
                    func = node.func
                    name = (
                        func.attr
                        if isinstance(func, ast.Attribute)
                        else getattr(func, "id", "")
                    )
                    first = node.args[0]
                    if name in (
                        "PathRefused",
                        "TextScanRefused",
                        "TextDecodeError",
                        "refuse",
                    ) and isinstance(first, ast.Constant):
                        codes.add(first.value)
        assert codes, "no refusal code found: the scan is vacuous"
        assert codes <= set(import_config.ISSUE_CODES), codes - set(
            import_config.ISSUE_CODES
        )


class TestImportRoots:
    def test_unset_means_the_data_directory_only(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "DATA_DIR", tmp_path)
        monkeypatch.delenv("PMM_IMPORT_ROOTS", raising=False)
        assert config.get_import_roots() == (tmp_path,)
        monkeypatch.setenv("PMM_IMPORT_ROOTS", "   ")
        assert config.get_import_roots() == (tmp_path,)

    def test_entries_follow_the_data_directory(self, monkeypatch, tmp_path):
        first, second = tmp_path / "a", tmp_path / "b"
        monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
        monkeypatch.setenv(
            "PMM_IMPORT_ROOTS", os.pathsep.join([str(first), str(second)])
        )
        assert config.get_import_roots() == (tmp_path / "data", first, second)

    @pytest.mark.parametrize(
        ("value", "problem"),
        [
            ("{a}" + os.pathsep + os.pathsep + "{a}", "empty entry"),
            ("relative" + os.sep + "dir", "relative"),
            ("\\\\server\\share", "UNC"),
            ("//server/share", "UNC"),
        ],
    )
    def test_unsafe_entries_are_refused(self, monkeypatch, tmp_path, value, problem):
        monkeypatch.setenv("PMM_IMPORT_ROOTS", value.replace("{a}", str(tmp_path)))
        with pytest.raises(
            ValueError, match=problem if problem != "relative" else "relative|absolute"
        ):
            config.get_import_roots()

    @pytest.mark.skipif(os.name != "nt", reason="Windows root-relative form")
    def test_a_root_relative_windows_entry_is_refused(self, monkeypatch):
        monkeypatch.setenv("PMM_IMPORT_ROOTS", "\\pmm\\data")
        with pytest.raises(ValueError, match="drive"):
            config.get_import_roots()


def _write(root: Path, name: str, data: bytes = b"1;2\n3;4\n") -> Path:
    path = root / name
    path.write_bytes(data)
    return path


def _open(root: Path, name: str, max_size: int = 1 << 30) -> path_safety.VerifiedFile:
    return path_safety.open_import_file(name, [root], base=root, max_size=max_size)


def _after_the_check(monkeypatch, path: Path, action) -> None:
    """Run *action* right after the ``lstat`` of the resolved *path*, i.e.
    between the check and the open. Armed only once ``safe_resolve`` has
    returned: on POSIX, resolving a path ``lstat``-s every component too."""
    real_lstat = os.lstat
    real_resolve = path_safety.safe_resolve
    resolved_path = path.resolve()  # before patching: resolving calls lstat
    state = {"armed": False, "done": False}

    def resolve_then_arm(*args, **kwargs):
        resolved = real_resolve(*args, **kwargs)
        state["armed"] = True
        return resolved

    def lstat_then_act(target, *args, **kwargs):
        seen = real_lstat(target, *args, **kwargs)
        if state["armed"] and not state["done"] and Path(target) == resolved_path:
            state["done"] = True
            action()
        return seen

    monkeypatch.setattr(path_safety, "safe_resolve", resolve_then_arm)
    monkeypatch.setattr(os, "lstat", lstat_then_act)


class TestOpenImportFile:
    def test_a_regular_file_opens_and_reads_only_up_to_its_size(self, tmp_path):
        _write(tmp_path, "pump.csv", b"0123456789")
        with _open(tmp_path, "pump.csv") as file:
            assert file.size == 10
            assert file.read(4, 100) == b"456789"
            assert file.read(10, 5) == b""
            assert b"".join(file.iter_chunks(3)) == b"0123456789"
            assert file.bytes_read == 16

    @pytest.mark.parametrize("name", ["PUMP.CSV", "pump.Txt"])
    def test_suffix_letter_case_is_ignored(self, tmp_path, name):
        _write(tmp_path, name)
        with _open(tmp_path, name) as file:
            assert file.path.name == name

    @pytest.mark.parametrize("name", ["pump.dat", "pump.json", "pump.csv.gz"])
    def test_other_suffixes_are_refused_with_the_rename_remedy(self, tmp_path, name):
        _write(tmp_path, name)
        with pytest.raises(path_safety.PathRefused) as caught:
            _open(tmp_path, name)
        assert caught.value.code == "unsupported_suffix"
        assert "renamed to .csv" in str(caught.value)

    def test_a_missing_file_is_named(self, tmp_path):
        with pytest.raises(path_safety.PathRefused) as caught:
            _open(tmp_path, "absent.csv")
        assert caught.value.code == "file_not_found"

    def test_a_folder_is_not_a_regular_file(self, tmp_path):
        (tmp_path / "folder.csv").mkdir()
        with pytest.raises(path_safety.PathRefused) as caught:
            _open(tmp_path, "folder.csv")
        assert caught.value.code == "not_a_regular_file"

    def test_the_size_cap_names_the_size_the_cap_and_the_variable(self, tmp_path):
        _write(tmp_path, "big.csv", b"x" * 100)
        with pytest.raises(path_safety.PathRefused) as caught:
            _open(tmp_path, "big.csv", max_size=99)
        message = str(caught.value)
        assert caught.value.code == "file_too_large"
        assert "100 bytes" in message and "99-byte" in message
        assert "PMM_MAX_SIGNAL_SIZE" in message

    def test_a_file_that_grows_after_open_is_refused_without_reading_past(
        self, tmp_path
    ):
        path = _write(tmp_path, "growing.csv", b"1;2\n" * 10)
        with _open(tmp_path, "growing.csv") as file:
            with path.open("ab") as appender:
                appender.write(b"5;6\n")
            seen = b""
            with pytest.raises(path_safety.PathRefused) as caught:
                for chunk in file.iter_chunks(16):
                    seen += chunk
        assert caught.value.code == "source_changed"
        assert "export may still be in progress" in str(caught.value)
        assert seen == b"1;2\n" * 10

    def test_a_file_swapped_between_check_and_open_is_refused(
        self, tmp_path, monkeypatch
    ):
        path = _write(tmp_path, "swap.csv")

        def swap():
            replacement = tmp_path / "other.tmp"
            replacement.write_bytes(b"9;9\n")
            os.replace(replacement, path)

        _after_the_check(monkeypatch, path, swap)
        with pytest.raises(path_safety.PathRefused) as caught:
            _open(tmp_path, "swap.csv")
        assert caught.value.code == "source_changed"

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO")
    def test_a_fifo_is_refused_without_blocking(self, tmp_path):
        os.mkfifo(tmp_path / "pipe.csv")
        start = time.perf_counter()
        with pytest.raises(path_safety.PathRefused) as caught:
            _open(tmp_path, "pipe.csv")
        assert caught.value.code == "not_a_regular_file"
        assert time.perf_counter() - start < 1.0

    @pytest.mark.skipif(os.name == "nt", reason="POSIX no-follow open")
    def test_a_symlink_swapped_in_after_the_check_is_refused_at_open(
        self, tmp_path, monkeypatch
    ):
        path = _write(tmp_path, "link.csv")
        (tmp_path / "target.csv").write_bytes(b"7;7\n")

        def link():
            path.unlink()
            os.symlink(tmp_path / "target.csv", path)

        _after_the_check(monkeypatch, path, link)
        with pytest.raises(path_safety.PathRefused) as caught:
            _open(tmp_path, "link.csv")
        assert caught.value.code == "source_changed"
        assert "replaced by a link" in str(caught.value)

    def test_a_locked_or_forbidden_file_is_unreadable_not_replaced(
        self, tmp_path, monkeypatch
    ):
        _write(tmp_path, "locked.csv")

        def denied(*_args, **_kwargs):
            raise PermissionError(13, "denied: /full/server/path/locked.csv")

        monkeypatch.setattr(os, "open", denied)
        with pytest.raises(path_safety.PathRefused) as caught:
            _open(tmp_path, "locked.csv")
        assert caught.value.code == "file_unreadable"
        assert "/full/server/path" not in str(caught.value)
        assert "another program holds the file open" in str(caught.value)

    @pytest.mark.parametrize(
        ("error", "code"),
        [
            (NotADirectoryError(20, "x"), "file_not_found"),
            (PermissionError(13, "x"), "file_unreadable"),
            (OSError(22, "x"), "file_unreadable"),
        ],
    )
    def test_lstat_errors_become_refusals(self, tmp_path, monkeypatch, error, code):
        _write(tmp_path, "x.csv")

        def failing(*_args, **_kwargs):
            raise error

        monkeypatch.setattr(os, "lstat", failing)
        monkeypatch.setattr(
            path_safety, "safe_resolve", lambda root, rel: Path(root) / rel
        )
        with pytest.raises(path_safety.PathRefused) as caught:
            _open(tmp_path, "x.csv")
        assert caught.value.code == code

    def test_a_link_loop_while_resolving_is_a_refusal(self, tmp_path, monkeypatch):
        def looping(*_args, **_kwargs):
            raise RuntimeError("Symlink loop from '/full/path'")

        monkeypatch.setattr(path_safety, "safe_resolve", looping)
        with pytest.raises(path_safety.PathRefused) as caught:
            _open(tmp_path, "x.csv")
        assert caught.value.code == "file_unreadable"
        assert "/full/path" not in str(caught.value)

    @pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
    def test_a_symlink_leading_outside_the_root_is_refused(self, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.csv").write_bytes(b"1;2\n")
        root = tmp_path / "root"
        root.mkdir()
        os.symlink(outside / "secret.csv", root / "x.csv")
        with pytest.raises(path_safety.PathRefused) as caught:
            _open(root, "x.csv")
        assert caught.value.code == "path_outside_roots"

    @pytest.mark.skipif(os.name != "nt", reason="Windows 8.3 short names")
    def test_a_short_name_alias_of_another_extension_is_refused(self, tmp_path):
        import ctypes

        long_path = _write(tmp_path, "report.csvx")
        buffer = ctypes.create_unicode_buffer(1024)
        ctypes.windll.kernel32.GetShortPathNameW(str(long_path), buffer, 1024)
        short_name = Path(buffer.value).name
        if (
            not short_name
            or short_name == long_path.name
            or not short_name.upper().endswith(".CSV")
        ):
            pytest.skip("8.3 names are not generated on this volume")
        with pytest.raises(path_safety.PathRefused) as caught:
            _open(tmp_path, short_name)
        assert caught.value.code == "unsupported_suffix"
