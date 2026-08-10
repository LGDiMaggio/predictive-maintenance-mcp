"""
Tests for the Signal Loader module.

Covers:
- Multi-format signal loading: CSV, NPY, MAT, WAV, Parquet
- Raw binary decoding (.bin/.raw/.dat) via load_raw_binary
- Segment extraction with seed reproducibility
- Segment boundary conditions
- Metadata path derivation for all formats
- Error handling for missing / invalid files
- PMM_MAX_SIGNAL_SIZE call-time getter
"""

import inspect
import struct

import pytest
import numpy as np
import pandas as pd

from conftest import write_raw_file
from predictive_maintenance_mcp.config import get_max_signal_size
from predictive_maintenance_mcp.signal_acquisition.loaders import (
    load_signal_data,
    load_raw_binary,
    extract_segment,
    get_metadata_path,
    get_metadata_path_from_dir,
    DATA_DIR,
    RAW_EXTENSIONS,
    RAW_PARAM_DEFAULTS,
    SELF_DESCRIBING_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
)

# ── load_signal_data ───────────────────────────────────────────────────────


class TestLoadSignalData:

    def test_load_csv(self, tmp_path, monkeypatch):
        """Load a CSV signal file."""
        data = np.array([1.0, 2.5, -1.3, 0.5, 3.2])
        csv_file = tmp_path / "test.csv"
        pd.DataFrame(data).to_csv(csv_file, header=False, index=False)

        monkeypatch.setattr(
            "predictive_maintenance_mcp.signal_acquisition.loaders.DATA_DIR", tmp_path
        )
        signal = load_signal_data("test.csv")
        np.testing.assert_array_almost_equal(signal, data)

    def test_load_npy(self, tmp_path, monkeypatch):
        """Load a NPY binary file."""
        data = np.array([1.1, 2.2, 3.3, 4.4])
        np.save(tmp_path / "test.npy", data)

        monkeypatch.setattr(
            "predictive_maintenance_mcp.signal_acquisition.loaders.DATA_DIR", tmp_path
        )
        signal = load_signal_data("test.npy")
        np.testing.assert_array_equal(signal, data)

    def test_load_mat(self, tmp_path, monkeypatch):
        """Load a MATLAB .mat file."""
        from scipy.io import savemat

        data = np.array([10.0, 20.0, 30.0])
        savemat(str(tmp_path / "test.mat"), {"signal": data})

        monkeypatch.setattr(
            "predictive_maintenance_mcp.signal_acquisition.loaders.DATA_DIR", tmp_path
        )
        signal = load_signal_data("test.mat")
        assert signal is not None
        np.testing.assert_array_almost_equal(signal, data)

    def test_load_wav(self, tmp_path, monkeypatch):
        """Load a WAV audio file."""
        from scipy.io import wavfile

        fs = 16000
        data = (np.sin(np.linspace(0, 2 * np.pi * 440, fs)) * 32767).astype(np.int16)
        wavfile.write(str(tmp_path / "test.wav"), fs, data)

        monkeypatch.setattr(
            "predictive_maintenance_mcp.signal_acquisition.loaders.DATA_DIR", tmp_path
        )
        signal = load_signal_data("test.wav")
        assert signal is not None
        assert len(signal) == len(data)
        assert signal.dtype == np.float64
        # Int16 signals are normalized to [-1, 1]
        assert np.max(np.abs(signal)) <= 1.0 + 1e-6

    @pytest.mark.skipif(
        not any(
            __import__("importlib").util.find_spec(e)
            for e in ("pyarrow", "fastparquet")
        ),
        reason="pyarrow/fastparquet not installed",
    )
    def test_load_parquet(self, tmp_path, monkeypatch):
        """Load a Parquet file."""
        data = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        pd.DataFrame({"signal": data}).to_parquet(tmp_path / "test.parquet")

        monkeypatch.setattr(
            "predictive_maintenance_mcp.signal_acquisition.loaders.DATA_DIR", tmp_path
        )
        signal = load_signal_data("test.parquet")
        assert signal is not None
        np.testing.assert_array_almost_equal(signal, data)

    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "predictive_maintenance_mcp.signal_acquisition.loaders.DATA_DIR", tmp_path
        )
        assert load_signal_data("nonexistent.csv") is None

    def test_unsupported_extension_returns_none(self, tmp_path, monkeypatch):
        (tmp_path / "test.xyz").write_text("data")
        monkeypatch.setattr(
            "predictive_maintenance_mcp.signal_acquisition.loaders.DATA_DIR", tmp_path
        )
        assert load_signal_data("test.xyz") is None

    def test_wav_stereo_uses_first_channel(self, tmp_path, monkeypatch):
        """Stereo WAV should use first channel only."""
        from scipy.io import wavfile

        fs = 8000
        ch1 = (np.sin(np.linspace(0, 2 * np.pi * 300, fs)) * 32767).astype(np.int16)
        ch2 = (np.sin(np.linspace(0, 2 * np.pi * 600, fs)) * 32767).astype(np.int16)
        stereo = np.column_stack([ch1, ch2])
        wavfile.write(str(tmp_path / "stereo.wav"), fs, stereo)

        monkeypatch.setattr(
            "predictive_maintenance_mcp.signal_acquisition.loaders.DATA_DIR", tmp_path
        )
        signal = load_signal_data("stereo.wav")
        assert signal is not None
        assert signal.ndim == 1
        assert len(signal) == fs

    def test_load_real_baseline_if_available(self):
        """Smoke test: load a real baseline CSV from the repo."""
        baseline = DATA_DIR / "real_train" / "baseline_1.csv"
        if not baseline.exists():
            pytest.skip("Real data not in repo")
        signal = load_signal_data("real_train/baseline_1.csv")
        assert signal is not None
        assert len(signal) > 1000


# ── extract_segment ────────────────────────────────────────────────────────


class TestExtractSegment:

    def test_seed_reproducibility(self):
        signal = np.arange(10000, dtype=float)
        seg1 = extract_segment(signal, 0.5, 1000, seed=42)
        seg2 = extract_segment(signal, 0.5, 1000, seed=42)
        np.testing.assert_array_equal(seg1, seg2)

    def test_different_seeds_differ(self):
        signal = np.arange(10000, dtype=float)
        seg1 = extract_segment(signal, 0.5, 1000, seed=42)
        seg2 = extract_segment(signal, 0.5, 1000, seed=99)
        assert not np.array_equal(seg1, seg2)

    def test_correct_length(self):
        signal = np.arange(10000, dtype=float)
        seg = extract_segment(signal, 1.0, 1000, seed=0)
        assert len(seg) == 1000

    def test_duration_exceeds_signal_returns_full(self):
        signal = np.arange(100, dtype=float)
        seg = extract_segment(signal, 2.0, 100, seed=0)
        np.testing.assert_array_equal(seg, signal)

    def test_duration_equals_signal_returns_full(self):
        signal = np.arange(100, dtype=float)
        seg = extract_segment(signal, 1.0, 100, seed=0)
        np.testing.assert_array_equal(seg, signal)

    def test_segment_is_contiguous_slice(self):
        signal = np.arange(10000, dtype=float)
        seg = extract_segment(signal, 0.2, 1000, seed=7)
        # Contiguous: consecutive integer values
        diffs = np.diff(seg)
        assert np.all(diffs == 1.0)

    def test_no_seed_is_random(self):
        signal = np.arange(100000, dtype=float)
        seg1 = extract_segment(signal, 0.1, 1000, seed=None)
        seg2 = extract_segment(signal, 0.1, 1000, seed=None)
        # Very unlikely to get the same random start (but could, so not strict assert)
        # Just verify the function runs without error
        assert len(seg1) == 100
        assert len(seg2) == 100


# ── get_metadata_path ──────────────────────────────────────────────────────


class TestGetMetadataPath:

    def test_csv_extension(self):
        path = get_metadata_path("signal.csv")
        assert path.name == "signal_metadata.json"

    def test_npy_extension(self):
        path = get_metadata_path("signal.npy")
        assert path.name == "signal_metadata.json"

    def test_mat_extension(self):
        path = get_metadata_path("signal.mat")
        assert path.name == "signal_metadata.json"

    def test_wav_extension(self):
        path = get_metadata_path("signal.wav")
        assert path.name == "signal_metadata.json"

    def test_parquet_extension(self):
        path = get_metadata_path("signal.parquet")
        assert path.name == "signal_metadata.json"

    def test_subdirectory_preserved(self):
        path = get_metadata_path("real_train/baseline_1.csv")
        assert "real_train" in str(path)
        assert path.name == "baseline_1_metadata.json"

    def test_from_dir_variant(self, tmp_path):
        path = get_metadata_path_from_dir(tmp_path, "test_signal.csv")
        assert path.parent == tmp_path
        assert path.name == "test_signal_metadata.json"

    def test_traversal_input_rejected(self):
        """F9: get_metadata_path is routed through safe_resolve, so a
        traversal filename that would escape DATA_DIR now raises instead of
        silently pointing outside it (defense-in-depth, matches
        load_signal_data)."""
        with pytest.raises(ValueError, match="escapes base directory"):
            get_metadata_path("../../../../etc/passwd.csv")

    def test_valid_input_stays_inside_data_dir(self):
        """Behavior identical for valid in-DATA_DIR inputs — the resolved
        metadata path is contained in DATA_DIR."""
        path = get_metadata_path("real_train/baseline_1.csv")
        assert path.is_relative_to(DATA_DIR.resolve())
        assert path.name == "baseline_1_metadata.json"


# ── raw extensions ─────────────────────────────────────────────────────────


class TestRawExtensions:

    def test_bin_raw_dat_in_supported_extensions(self):
        assert ".bin" in SUPPORTED_EXTENSIONS
        assert ".raw" in SUPPORTED_EXTENSIONS
        assert ".dat" in SUPPORTED_EXTENSIONS

    def test_raw_extensions_constant(self):
        assert RAW_EXTENSIONS == {".bin", ".raw", ".dat"}

    def test_existing_formats_still_supported(self):
        for ext in (".csv", ".txt", ".npy", ".mat", ".wav", ".parquet"):
            assert ext in SUPPORTED_EXTENSIONS

    def test_supported_is_composed_from_both_classes(self):
        """SUPPORTED_EXTENSIONS is composed, not hand-listed — a raw
        extension can never be raw-eligible but unlisted (the inverse of
        the .dat bug), and the two classes partition the supported set."""
        assert isinstance(SUPPORTED_EXTENSIONS, list)
        assert set(SUPPORTED_EXTENSIONS) == (
            set(SELF_DESCRIBING_EXTENSIONS) | set(RAW_EXTENSIONS)
        )
        assert not set(SELF_DESCRIBING_EXTENSIONS) & set(RAW_EXTENSIONS)

    def test_decoder_signature_defaults_match_raw_param_defaults(self):
        """RAW_PARAM_DEFAULTS is the single source of truth for the
        optional raw-parameter defaults — the decoder's literal keyword
        defaults are pinned to it so the two can never drift."""
        sig = inspect.signature(load_raw_binary)
        assert {
            name: sig.parameters[name].default for name in RAW_PARAM_DEFAULTS
        } == RAW_PARAM_DEFAULTS


# ── get_max_signal_size ────────────────────────────────────────────────────


class TestGetMaxSignalSize:

    def test_default_is_500_mb(self, monkeypatch):
        monkeypatch.delenv("PMM_MAX_SIGNAL_SIZE", raising=False)
        assert get_max_signal_size() == 500_000_000

    def test_env_override_read_at_each_call(self, monkeypatch):
        """The env var is read at call time, not frozen at import time."""
        monkeypatch.setenv("PMM_MAX_SIGNAL_SIZE", "123")
        assert get_max_signal_size() == 123
        monkeypatch.setenv("PMM_MAX_SIGNAL_SIZE", "456")
        assert get_max_signal_size() == 456


# ── load_raw_binary ────────────────────────────────────────────────────────


class TestLoadRawBinaryHappyPath:
    """Round-trips: known values written raw come back exactly, as float64."""

    def test_roundtrip_float32_le(self, tmp_path):
        values = [1.5, -2.25, 0.0, 3.75, -0.5]
        f = tmp_path / "sig.bin"
        write_raw_file(f, values, "<f4")
        out = load_raw_binary(f, sample_format="float32")
        assert out.dtype == np.float64
        assert out.shape == (5,)
        np.testing.assert_array_equal(out, np.array(values, dtype=np.float64))

    def test_roundtrip_float64_le(self, tmp_path):
        values = [1.1, -2.2, 3.3, 0.0]
        f = tmp_path / "sig.raw"
        write_raw_file(f, values, "<f8")
        out = load_raw_binary(f, sample_format="float64")
        assert out.dtype == np.float64
        np.testing.assert_array_equal(out, np.array(values, dtype=np.float64))

    def test_roundtrip_int16_le(self, tmp_path):
        values = [100, -200, 32767, -32768, 7]
        f = tmp_path / "sig.bin"
        write_raw_file(f, values, "<i2")
        out = load_raw_binary(f, sample_format="int16")
        assert out.dtype == np.float64
        np.testing.assert_array_equal(out, np.array(values, dtype=np.float64))

    def test_roundtrip_int32_le(self, tmp_path):
        values = [100_000, -2_000_000, 2_147_483_647, -2_147_483_648]
        f = tmp_path / "sig.dat"
        write_raw_file(f, values, "<i4")
        out = load_raw_binary(f, sample_format="int32")
        assert out.dtype == np.float64
        np.testing.assert_array_equal(out, np.array(values, dtype=np.float64))

    def test_roundtrip_float32_big_endian(self, tmp_path):
        values = [1.5, -2.25, 42.0]
        f = tmp_path / "sig.bin"
        write_raw_file(f, values, ">f4")
        out = load_raw_binary(f, sample_format="float32", byte_order="big")
        np.testing.assert_array_equal(out, np.array(values, dtype=np.float64))

    def test_reference_shape_exact_sample_count(self, tmp_path):
        """Reference shape: a 5,398,528-byte file is exactly 1,349,632 float32."""
        f = tmp_path / "vendor_capture.bin"
        np.zeros(1_349_632, dtype="<f4").tofile(f)
        assert f.stat().st_size == 5_398_528
        out = load_raw_binary(f, sample_format="float32")
        assert out.shape == (1_349_632,)

    def test_header_offset_skips_header_and_recovers_sine(self, tmp_path):
        fs, freq, n = 1000, 50.0, 1000
        t = np.arange(n) / fs
        sine = np.sin(2 * np.pi * freq * t).astype("<f4")
        header = b"FAKEHDR!" * 2  # 16 bytes of non-sample junk
        f = tmp_path / "with_header.raw"
        f.write_bytes(header + sine.tobytes())
        out = load_raw_binary(f, sample_format="float32", header_offset=16)
        assert out.shape == (n,)
        np.testing.assert_array_equal(out, sine.astype(np.float64))

    def test_two_channel_interleaved_recovers_each_frequency(self, tmp_path):
        fs, n = 1000, 2000  # 0.5 Hz bins: 50 and 200 Hz fall on exact bins
        t = np.arange(n) / fs
        ch0 = np.sin(2 * np.pi * 50.0 * t)
        ch1 = np.sin(2 * np.pi * 200.0 * t)
        interleaved = np.empty(2 * n, dtype="<f4")
        interleaved[0::2] = ch0
        interleaved[1::2] = ch1
        f = tmp_path / "stereo.bin"
        interleaved.tofile(f)

        def dominant_freq(x: np.ndarray) -> float:
            spectrum = np.abs(np.fft.rfft(x))
            spectrum[0] = 0.0  # ignore DC
            return float(np.fft.rfftfreq(len(x), 1 / fs)[int(np.argmax(spectrum))])

        out0 = load_raw_binary(
            f, sample_format="float32", n_channels=2, channel_index=0
        )
        out1 = load_raw_binary(
            f, sample_format="float32", n_channels=2, channel_index=1
        )
        assert out0.shape == (n,)
        assert out1.shape == (n,)
        assert abs(dominant_freq(out0) - 50.0) < 1.0
        assert abs(dominant_freq(out1) - 200.0) < 1.0

    def test_scale_factor_int16_multiplies_exactly(self, tmp_path):
        f = tmp_path / "counts.bin"
        write_raw_file(f, [100, -200, 300], "<i2")
        out = load_raw_binary(f, sample_format="int16", scale_factor=0.5)
        np.testing.assert_array_equal(out, np.array([50.0, -100.0, 150.0]))

    def test_int16_without_scale_keeps_raw_counts(self, tmp_path):
        """No WAV-style implicit normalization: raw counts stay raw."""
        f = tmp_path / "counts.bin"
        write_raw_file(f, [100, -32768, 32767], "<i2")
        out = load_raw_binary(f, sample_format="int16")
        assert out.dtype == np.float64
        np.testing.assert_array_equal(out, np.array([100.0, -32768.0, 32767.0]))


class TestLoadRawBinaryRefusals:
    """Typed refusals: ValueError with 'problem — remedy' messages."""

    def test_invalid_sample_format_lists_vocabulary(self, tmp_path):
        f = tmp_path / "sig.bin"
        write_raw_file(f, [1.0], "<f4")
        with pytest.raises(ValueError) as exc:
            load_raw_binary(f, sample_format="Float32")
        msg = str(exc.value)
        for fmt in ("float32", "float64", "int16", "int32"):
            assert fmt in msg

    def test_invalid_byte_order_lists_vocabulary(self, tmp_path):
        f = tmp_path / "sig.bin"
        write_raw_file(f, [1.0], "<f4")
        with pytest.raises(ValueError) as exc:
            load_raw_binary(f, sample_format="float32", byte_order="middle")
        msg = str(exc.value)
        assert "little" in msg
        assert "big" in msg

    def test_n_channels_zero_refused_never_zerodivision(self, tmp_path):
        f = tmp_path / "sig.bin"
        write_raw_file(f, [1.0, 2.0], "<f4")
        with pytest.raises(ValueError, match="n_channels"):
            load_raw_binary(f, sample_format="float32", n_channels=0)

    def test_n_channels_negative_refused(self, tmp_path):
        f = tmp_path / "sig.bin"
        write_raw_file(f, [1.0, 2.0], "<f4")
        with pytest.raises(ValueError, match="n_channels"):
            load_raw_binary(f, sample_format="float32", n_channels=-3)

    def test_channel_index_negative_refused(self, tmp_path):
        f = tmp_path / "sig.bin"
        write_raw_file(f, [1.0, 2.0], "<f4")
        with pytest.raises(ValueError, match="channel_index"):
            load_raw_binary(f, sample_format="float32", channel_index=-1)

    def test_channel_index_out_of_range_names_valid_range(self, tmp_path):
        f = tmp_path / "sig.bin"
        write_raw_file(f, [1.0, 2.0, 3.0, 4.0], "<f4")
        with pytest.raises(ValueError) as exc:
            load_raw_binary(f, sample_format="float32", n_channels=2, channel_index=2)
        assert "0..1" in str(exc.value)

    def test_param_validation_precedes_file_access(self, tmp_path):
        """Parameter checks come before existence/stat — a bad declaration
        on a missing file still reports the declaration problem."""
        missing = tmp_path / "missing.bin"
        with pytest.raises(ValueError, match="n_channels"):
            load_raw_binary(missing, sample_format="float32", n_channels=0)

    def test_negative_header_offset_refused(self, tmp_path):
        f = tmp_path / "sig.bin"
        write_raw_file(f, [1.0, 2.0], "<f4")
        with pytest.raises(ValueError, match="header_offset"):
            load_raw_binary(f, sample_format="float32", header_offset=-1)

    def test_header_offset_equal_to_file_size_refused(self, tmp_path):
        f = tmp_path / "sig.bin"
        f.write_bytes(b"\x00" * 64)
        with pytest.raises(ValueError) as exc:
            load_raw_binary(f, sample_format="float32", header_offset=64)
        msg = str(exc.value)
        assert "64" in msg

    def test_header_offset_beyond_file_size_refused(self, tmp_path):
        f = tmp_path / "sig.bin"
        f.write_bytes(b"\x00" * 64)
        with pytest.raises(ValueError):
            load_raw_binary(f, sample_format="float32", header_offset=100)

    def test_empty_file_refused(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        with pytest.raises(ValueError) as exc:
            load_raw_binary(f, sample_format="float32")
        assert "0" in str(exc.value)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        """The pure decoder raises a bare FileNotFoundError; the actionable
        list_signals remedy is the repository layer's business (it checks
        existence on every route and owns the canonical message)."""
        with pytest.raises(FileNotFoundError):
            load_raw_binary(tmp_path / "nope.bin", sample_format="float32")

    def test_size_off_by_one_larger_refused_with_arithmetic(self, tmp_path):
        f = tmp_path / "sig.bin"
        f.write_bytes(b"\x00" * 401)  # (100 x 4) + 1
        with pytest.raises(ValueError) as exc:
            load_raw_binary(f, sample_format="float32")
        msg = str(exc.value)
        assert "401" in msg  # payload / file size
        assert "header_offset 0" in msg
        assert "4 bytes" in msg  # dtype size
        assert "1 channel" in msg
        assert "remainder 1" in msg

    def test_size_off_by_one_smaller_refused_with_arithmetic(self, tmp_path):
        f = tmp_path / "sig.bin"
        f.write_bytes(b"\x00" * 399)  # (100 x 4) - 1
        with pytest.raises(ValueError) as exc:
            load_raw_binary(f, sample_format="float32")
        msg = str(exc.value)
        assert "399" in msg
        assert "remainder 3" in msg

    def test_divisible_by_dtype_not_by_frame_refused(self, tmp_path):
        # 12 bytes = 3 float32 samples, but not a whole number of
        # 2-channel frames (8 bytes each): remainder 4.
        f = tmp_path / "sig.bin"
        f.write_bytes(b"\x00" * 12)
        with pytest.raises(ValueError) as exc:
            load_raw_binary(f, sample_format="float32", n_channels=2)
        msg = str(exc.value)
        assert "12" in msg
        assert "2 channel" in msg
        assert "4 bytes" in msg
        assert "remainder 4" in msg

    def test_nonfinite_float_payload_refused_with_count(self, tmp_path):
        """Bytes written with the wrong endianness/dtype decode to NaN/Inf
        under the declared little-endian float32 — refusal carries the
        non-finite count and points at the declaration."""
        good = struct.pack("<f", 1.0)
        nan_le = b"\x00\x00\xc0\x7f"  # float32 NaN (little-endian)
        inf_le = b"\x00\x00\x80\x7f"  # float32 +Inf (little-endian)
        f = tmp_path / "swapped.bin"
        f.write_bytes(good * 3 + nan_le * 2 + inf_le + good * 2)
        with pytest.raises(ValueError) as exc:
            load_raw_binary(f, sample_format="float32")
        msg = str(exc.value)
        assert "3 non-finite" in msg
        assert "byte_order" in msg or "endian" in msg.lower()
        assert "sample_format" in msg

    def test_nonfinite_check_applies_to_selected_channel_only(self, tmp_path):
        """A NaN confined to channel 1 must not block loading channel 0."""
        good = struct.pack("<f", 2.0)
        nan_le = b"\x00\x00\xc0\x7f"
        # Interleaved frames: (ch0, ch1) = (good, nan), (good, good)
        f = tmp_path / "stereo_nan.bin"
        f.write_bytes(good + nan_le + good + good)
        out = load_raw_binary(f, sample_format="float32", n_channels=2, channel_index=0)
        np.testing.assert_array_equal(out, np.array([2.0, 2.0]))
        with pytest.raises(ValueError, match="non-finite"):
            load_raw_binary(f, sample_format="float32", n_channels=2, channel_index=1)

    def test_file_over_size_cap_refused_naming_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PMM_MAX_SIGNAL_SIZE", "16")
        f = tmp_path / "big.bin"
        write_raw_file(f, [1.0, 2.0, 3.0, 4.0, 5.0], "<f4")  # 20 bytes > 16
        with pytest.raises(ValueError, match="PMM_MAX_SIGNAL_SIZE"):
            load_raw_binary(f, sample_format="float32")

    def test_file_at_exact_size_cap_loads(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PMM_MAX_SIGNAL_SIZE", "16")
        f = tmp_path / "ok.bin"
        write_raw_file(f, [1.0, 2.0, 3.0, 4.0], "<f4")  # exactly 16 bytes
        out = load_raw_binary(f, sample_format="float32")
        assert out.shape == (4,)
