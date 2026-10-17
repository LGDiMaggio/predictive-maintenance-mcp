"""Diagnostic invariants of the shipped tools.

Characterization tests that pin what the diagnostic chain must keep true,
organized by invariant class: signal units, sampling-frequency assumptions,
FFT and envelope calculations, bearing fault frequencies, severity echoes,
output provenance, and invalid-input handling.

Rules this module follows:

- Every expected value is computed HERE from closed-form physics (velocity
  from acceleration, Parseval with the Hamming power gain, rolling-element
  kinematics), never copied from what the code under test returns.
- Every random source is seeded, so each run sees identical samples.
- Signals are written into the sandboxed ``data/signals`` directory and
  loaded by absolute path with an explicit ``signal_id``, ``sampling_rate``
  and DECLARED ``signal_unit`` (units are never guessed), as in
  ``tests/_golden_signals.py``. Every loaded id is cleared afterwards.
- Nothing under ``src/`` is changed to make a test pass. An invariant the
  shipped code violates is a ``pytest.mark.xfail(strict=True, raises=...)``
  tripwire whose reason, built by :func:`known_defect`, starts with
  ``audit-known-defect:`` so the weekly audit report lists it as a
  diagnostic-correctness finding. The
  ``raises=`` class is the one the CURRENT defect produces: ``AssertionError``
  where the code returns a wrong result or accepts what it should refuse,
  the library's own exception where library text reaches the caller. When a
  defect is fixed the tripwire XPASSes, which fails the strict run and asks
  for the marker to be removed.

What counts as a project-authored refusal: since issue #73 every registered
tool is wrapped by ``refusal_boundary``, which turns any ``ValueError`` into a
``ToolValueError``, so the exception type says nothing about who wrote the
message. These tests call the module-level functions (no boundary) and judge
a ``ValueError`` by where it was raised, the same criterion the boundary uses
for its log level: innermost frame inside ``src/`` means the project wrote
the message for the model; anywhere else means a library did. Passing
refusal tests additionally pin the project phrase the message carries.

Severity claims follow ISO 20816-3:2022 (evaluation band 10-1000 Hz, 2 Hz
lower edge below 600 r/min) with zone boundary values from ISO 10816-3:2009;
the boundary table itself is pinned in ``tests/test_iso20816.py`` and is not
repeated here. Bearing kinematics follow Randall & Antoni (2011), "Rolling
element bearing diagnostics - A tutorial", MSSP 25(2), 485-520.
"""

import json
import math
import traceback
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pydantic
import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer

from predictive_maintenance_mcp.document_reader import calculate_bearing_frequencies
from predictive_maintenance_mcp.mcp_tools import register_all
from predictive_maintenance_mcp.mcp_tools.acquisition_tools import load_signal
from predictive_maintenance_mcp.mcp_tools.analysis_tools import (
    analyze_envelope,
    analyze_fft,
    compute_power_spectral_density,
    compute_spectrogram_stft,
)
from predictive_maintenance_mcp.mcp_tools.diagnostics_tools import (
    assess_severity,
    calculate_bearing_characteristic_frequencies,
    check_bearing_faults,
)
from predictive_maintenance_mcp.signal_acquisition.repository import get_repository
from tools.audit.report import KNOWN_DEFECT_PREFIX

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = (REPO_ROOT / "src").resolve()
CATALOG_FILE = (
    REPO_ROOT / "resources" / "bearing_catalogs" / "common_bearings_catalog.json"
)

#: Sampling rate of most fixture signals (Hz).
FS = 10000.0

#: Standard gravity, m/s^2 per g (ISO 80000-3), defined independently of src.
STANDARD_GRAVITY = 9.80665

#: 6205 deep-groove ball bearing as documented by the CWRU Bearing Data
#: Center (drive end, SKF 6205-2RS JEM): Z, Bd (mm), Pd (mm).
Z_6205, BD_6205, PD_6205 = 9, 7.94, 39.04

#: CWRU's published cage (FTF) factor for that bearing, x shaft speed.
CWRU_FTF_FACTOR_6205 = 0.39828


# ---------------------------------------------------------------------------
# Closed-form references and helpers
# ---------------------------------------------------------------------------


def known_defect(short_id: str, text: str) -> str:
    """The xfail reason of a tripwire: ``audit-known-defect: <id>: <text>``.

    The audit report reads ``<id>`` as the finding's subject (its stable
    fingerprint) and ``<text>`` as its title.
    """
    return f"{KNOWN_DEFECT_PREFIX} {short_id}: {text}"


def _sine(amplitude: float, freq_hz: float, fs: float = FS, seconds: float = 1.0):
    """``amplitude * sin(2 pi f t)`` sampled at ``fs`` (t = n / fs)."""
    t = np.arange(int(round(fs * seconds))) / fs
    return amplitude * np.sin(2.0 * np.pi * freq_hz * t)


def _velocity_rms_mm_s(accel_amplitude_m_s2: float, freq_hz: float) -> float:
    """RMS velocity (mm/s) of a sinusoidal acceleration of amplitude A (m/s^2).

    v(t) = integral of A sin(wt) = -A/w cos(wt): amplitude A / (2 pi f),
    RMS amplitude / sqrt(2), times 1000 for m/s -> mm/s.
    """
    return accel_amplitude_m_s2 / (2.0 * math.pi * freq_hz) / math.sqrt(2.0) * 1000.0


def _bearing_kinematics(
    z: int, bd_mm: float, pd_mm: float, alpha_deg: float, rpm: float
) -> dict[str, float]:
    """Rolling-element defect frequencies (Hz), Randall & Antoni (2011).

    fr = rpm/60, c = (Bd/Pd) cos(alpha):
    BPFO = Z/2 fr (1 - c), BPFI = Z/2 fr (1 + c),
    BSF = Pd/(2 Bd) fr (1 - c^2), FTF = fr/2 (1 - c).
    """
    fr = rpm / 60.0
    c = bd_mm / pd_mm * math.cos(math.radians(alpha_deg))
    return {
        "BPFO": z / 2.0 * fr * (1.0 - c),
        "BPFI": z / 2.0 * fr * (1.0 + c),
        "BSF": pd_mm / (2.0 * bd_mm) * fr * (1.0 - c**2),
        "FTF": fr / 2.0 * (1.0 - c),
    }


def _raised_in_project(exc: BaseException) -> bool:
    """True when the innermost frame of *exc* lies inside ``src/``."""
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return False
    try:
        return Path(frames[-1].filename).resolve().is_relative_to(SRC_DIR)
    except (OSError, ValueError):
        return False


async def _outcome(awaitable):
    """Await a tool call: ``(result, None)`` or ``(None, project_refusal)``.

    A ``ValueError`` raised outside ``src/`` is library text reaching the
    caller; it propagates unchanged so a tripwire can name its class.
    """
    try:
        return await awaitable, None
    except ValueError as exc:
        if _raised_in_project(exc):
            return None, exc
        raise


def _peak_near(peaks, freq_hz: float, tol_hz: float):
    """The strongest peak within ``tol_hz`` of ``freq_hz``, or None."""
    near = [p for p in peaks if abs(p.frequency_hz - freq_hz) <= tol_hz]
    return max(near, key=lambda p: p.magnitude) if near else None


def _loaded_ids() -> set[str]:
    return {s["signal_id"] for s in get_repository().list_signals()}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ctx():
    return AsyncMock()


@pytest.fixture
def load_csv(sandbox_data_dir, mock_ctx):
    """Write a CSV into the sandbox and load it with declared parameters.

    Every id is registered for cleanup BEFORE the load, so a load that
    stores the signal and then fails still leaves the repository clean.
    """
    repo = get_repository()
    ids: list[str] = []

    async def _load(signal_id, values, *, fs=FS, unit="g"):
        path = sandbox_data_dir / f"{signal_id}.csv"
        pd.DataFrame(np.asarray(values, dtype=float)).to_csv(
            path, index=False, header=False
        )
        ids.append(signal_id)
        return await load_signal(
            mock_ctx,
            str(path),
            signal_id=signal_id,
            sampling_rate=fs,
            signal_unit=unit,
            overwrite=True,
        )

    yield _load
    for sid in ids:
        repo.clear_signal(sid)


@pytest.fixture(scope="module")
def mcp():
    server = MCPServer("test-diagnostic-invariants")
    register_all(server)
    return server


# ===========================================================================
# Units: declared unit -> ISO 20816-3 broadband RMS velocity (mm/s)
# ===========================================================================


class TestUnits:
    """The declared unit drives the conversion to ISO 20816-3 velocity."""

    @pytest.mark.asyncio
    async def test_g_acceleration_gives_closed_form_velocity(self, load_csv, mock_ctx):
        await load_csv("inv_units_g", _sine(0.5, 50.0), unit="g")
        result = await assess_severity(ctx=mock_ctx, signal_id="inv_units_g")

        expected = _velocity_rms_mm_s(0.5 * STANDARD_GRAVITY, 50.0)  # ~11.04 mm/s
        assert result.rms_velocity_mm_s == pytest.approx(expected, rel=0.02)

    @pytest.mark.asyncio
    async def test_m_s2_acceleration_gives_the_same_velocity(self, load_csv, mock_ctx):
        await load_csv("inv_units_ms2", _sine(4.903325, 50.0), unit="m/s2")
        result = await assess_severity(ctx=mock_ctx, signal_id="inv_units_ms2")

        expected = _velocity_rms_mm_s(0.5 * STANDARD_GRAVITY, 50.0)
        assert result.rms_velocity_mm_s == pytest.approx(expected, rel=0.02)

    @pytest.mark.asyncio
    async def test_m_s_velocity_scales_to_mm_s(self, load_csv, mock_ctx):
        await load_csv("inv_units_ms", _sine(0.002, 50.0), unit="m/s")
        result = await assess_severity(ctx=mock_ctx, signal_id="inv_units_ms")

        expected = 0.002 / math.sqrt(2.0) * 1000.0  # ~1.414 mm/s
        assert result.rms_velocity_mm_s == pytest.approx(expected, rel=0.02)

    @pytest.mark.asyncio
    async def test_equal_acceleration_velocity_scales_inversely_with_frequency(
        self, load_csv, mock_ctx
    ):
        """v = a / (2 pi f): the same acceleration at 25 Hz and at 100 Hz
        gives a 4:1 velocity ratio (both inside the 10-1000 Hz band)."""
        await load_csv("inv_units_25", _sine(1.0, 25.0), unit="m/s2")
        await load_csv("inv_units_100", _sine(1.0, 100.0), unit="m/s2")
        low = await assess_severity(ctx=mock_ctx, signal_id="inv_units_25")
        high = await assess_severity(ctx=mock_ctx, signal_id="inv_units_100")

        assert low.rms_velocity_mm_s / high.rms_velocity_mm_s == pytest.approx(
            100.0 / 25.0, rel=0.05
        )


# ===========================================================================
# Sampling frequency: the declared fs defines every frequency axis
# ===========================================================================


class TestSamplingFrequency:
    """Resolution and frequency limits follow from the declared fs."""

    FS_DAQ = 12800.0  # a common DAQ rate, so fs != 10 kHz is exercised

    @pytest.mark.asyncio
    @pytest.mark.parametrize("segment_duration", [1.0, 0.25, None])
    async def test_fft_resolution_is_fs_over_analyzed_samples(
        self, load_csv, mock_ctx, segment_duration
    ):
        """``analyze_fft`` echoes fs, the analyzed sample count and
        ``fs / N`` for that segment, and its frequency axis starts there."""
        rng = np.random.default_rng(11)
        signal = _sine(1.0, 400.0, fs=self.FS_DAQ, seconds=1.5)
        signal = signal + 0.01 * rng.standard_normal(signal.size)
        await load_csv("inv_fs_fft", signal, fs=self.FS_DAQ)

        result = await analyze_fft(
            ctx=mock_ctx, signal_id="inv_fs_fft", segment_duration=segment_duration
        )

        expected_n = (
            signal.size
            if segment_duration is None
            else int(round(segment_duration * self.FS_DAQ))
        )
        resolution = self.FS_DAQ / expected_n
        assert result.sampling_rate == self.FS_DAQ
        assert result.num_samples == expected_n
        assert result.frequency_resolution == pytest.approx(resolution, rel=1e-12)
        assert result.freq_range_hz[0] == pytest.approx(resolution, abs=1e-3)
        assert result.freq_range_hz[1] < self.FS_DAQ / 2.0
        assert abs(result.peak_frequency - 400.0) <= resolution

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "fs, band",
        [
            (10000.0, (500.0, 5000.0)),
            (8000.0, (500.0, 3900.0)),
            (25600.0, (500.0, 12000.0)),
        ],
    )
    async def test_envelope_peaks_never_exceed_nyquist(
        self, load_csv, mock_ctx, fs, band
    ):
        rng = np.random.default_rng(int(fs))
        await load_csv("inv_fs_env", rng.standard_normal(int(fs)), fs=fs)

        result = await analyze_envelope(
            ctx=mock_ctx,
            signal_id="inv_fs_env",
            filter_low=band[0],
            filter_high=band[1],
            num_peaks=50,
        )

        assert result.top_peaks
        assert all(0.0 < p.frequency_hz <= fs / 2.0 for p in result.top_peaks)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "fs, nperseg",
        [(10000.0, 256), (10000.0, 1000), (25600.0, 1024), (12800.0, 300)],
    )
    async def test_psd_resolution_is_fs_over_nperseg(
        self, load_csv, mock_ctx, fs, nperseg
    ):
        rng = np.random.default_rng(nperseg)
        await load_csv("inv_fs_psd", rng.standard_normal(int(fs)), fs=fs)

        result = await compute_power_spectral_density(
            ctx=mock_ctx, signal_id="inv_fs_psd", nperseg=nperseg, noverlap=nperseg // 2
        )

        assert result.nperseg == nperseg
        assert result.frequency_resolution == pytest.approx(fs / nperseg, abs=1e-4)
        assert result.freq_range_hz[1] == pytest.approx(fs / 2.0, abs=1e-3)

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=known_defect(
            "psd-nperseg-echo",
            "compute_power_spectral_density clamps nperseg to the signal length "
            "but echoes the requested value",
        ),
    )
    async def test_psd_echo_matches_the_segment_actually_used(self, load_csv, mock_ctx):
        """nperseg larger than the signal: the echoed nperseg must be the one
        whose ``fs / nperseg`` is the reported resolution."""
        rng = np.random.default_rng(5)
        await load_csv("inv_fs_psd_short", rng.standard_normal(1000))

        result = await compute_power_spectral_density(
            ctx=mock_ctx, signal_id="inv_fs_psd_short", nperseg=4096, noverlap=2048
        )

        assert result.frequency_resolution == pytest.approx(
            result.sampling_rate / result.nperseg, abs=1e-4
        ), (
            f"echoed nperseg={result.nperseg} but resolution "
            f"{result.frequency_resolution} Hz = fs/{FS / result.frequency_resolution:g}"
        )

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        strict=True,
        raises=pydantic.ValidationError,
        reason=known_defect(
            "stft-result-schema",
            "compute_spectrogram_stft always fails STFTResult validation "
            "(energy_per_band band labels are strings)",
        ),
    )
    async def test_stft_frequency_axis_ends_at_nyquist(self, load_csv, mock_ctx):
        rng = np.random.default_rng(6)
        await load_csv("inv_fs_stft", rng.standard_normal(int(FS)))

        result, refusal = await _outcome(
            compute_spectrogram_stft(ctx=mock_ctx, signal_id="inv_fs_stft")
        )

        assert refusal is None, f"STFT refused a valid 1 s signal: {refusal}"
        assert result.freq_range_hz[1] == pytest.approx(FS / 2.0, abs=1e-3)
        assert 0.0 <= result.max_power_freq_hz <= FS / 2.0


# ===========================================================================
# FFT
# ===========================================================================


class TestFFT:
    """``analyze_fft``: Hamming window, single-sided 2|X|/N amplitudes."""

    @pytest.mark.asyncio
    async def test_two_tones_are_found_and_ranked_by_amplitude(
        self, load_csv, mock_ctx
    ):
        """120 Hz (amplitude 1.0) and 1.3 kHz (0.35) on exact 1 Hz bins: both
        appear among the top peaks within one bin, the larger first; the
        window's coherent gain cancels in the amplitude ratio."""
        rng = np.random.default_rng(120)
        signal = (
            _sine(1.0, 120.0, seconds=2.0)
            + _sine(0.35, 1300.0, seconds=2.0)
            + 0.02 * rng.standard_normal(int(2 * FS))
        )
        await load_csv("inv_fft_two_tone", signal)

        result = await analyze_fft(ctx=mock_ctx, signal_id="inv_fft_two_tone")

        bin_hz = FS / int(FS * 1.0)  # leading 1.0 s segment by default
        low = _peak_near(result.top_peaks, 120.0, bin_hz)
        high = _peak_near(result.top_peaks, 1300.0, bin_hz)
        assert low is not None and high is not None
        assert abs(result.peak_frequency - 120.0) <= bin_hz
        assert low.magnitude > high.magnitude
        assert low.magnitude / high.magnitude == pytest.approx(1.0 / 0.35, rel=0.05)

    @pytest.mark.asyncio
    async def test_white_noise_energy_matches_after_window_power_correction(
        self, load_csv, mock_ctx
    ):
        """Parseval for a Hamming-windowed spectrum of white noise.

        Hamming: w[n] = 0.54 - 0.46 cos(2 pi n / (N - 1)). Its power gain is
        U = mean(w^2) = 0.54^2 + 0.46^2 / 2 (the cross term averages to zero,
        cos^2 to one half). Parseval: sum_k |X_w[k]|^2 = N sum_n (x w)^2,
        about N^2 sigma^2 U for white noise; the positive bins carry half.
        With one-sided amplitudes A_k = 2|X_w[k]|/N the spectral power is
        sum A_k^2 / 2 = sigma^2 U, so dividing by U recovers the time-domain
        mean square. ``rms_spectral^2 * total_bins`` is sum A_k^2.
        """
        rng = np.random.default_rng(2024)
        signal = 2.0 * rng.standard_normal(int(FS))
        await load_csv("inv_fft_noise", signal)

        result = await analyze_fft(
            ctx=mock_ctx, signal_id="inv_fft_noise", segment_duration=None
        )

        hamming_power_gain = 0.54**2 + 0.46**2 / 2.0
        spectral_power = result.rms_spectral**2 * result.total_bins / 2.0
        assert result.total_bins == (signal.size - 1) // 2
        assert spectral_power / hamming_power_gain == pytest.approx(
            float(np.mean(signal**2)), rel=0.05
        )


# ===========================================================================
# Envelope
# ===========================================================================


class TestEnvelope:
    @pytest.mark.asyncio
    async def test_am_rate_is_the_strongest_envelope_peak(self, load_csv, mock_ctx):
        """A 3 kHz carrier amplitude-modulated at 97 Hz demodulates to a
        97 Hz envelope line, within one bin (fs / analyzed samples)."""
        rng = np.random.default_rng(97)
        t = np.arange(int(FS)) / FS
        signal = (1.0 + 0.5 * np.sin(2 * np.pi * 97.0 * t)) * np.sin(
            2 * np.pi * 3000.0 * t
        ) + 0.05 * rng.standard_normal(t.size)
        await load_csv("inv_env_am", signal)

        result = await analyze_envelope(ctx=mock_ctx, signal_id="inv_env_am")

        bin_hz = FS / int(FS * 1.0)
        strongest = max(result.top_peaks, key=lambda p: p.magnitude)
        assert abs(strongest.frequency_hz - 97.0) <= bin_hz


# ===========================================================================
# Bearing fault frequencies (engine kinematics + tool wiring)
# ===========================================================================


class TestBearingFrequencies:
    """``document_reader.calculate_bearing_frequencies`` against closed form."""

    @pytest.mark.asyncio
    async def test_6205_at_1797_rpm_matches_closed_form(self):
        expected = _bearing_kinematics(Z_6205, BD_6205, PD_6205, 0.0, 1797.0)
        # Self-check of the reference itself: BSF about 70.6 Hz, and FTF
        # agrees with CWRU's published cage factor.
        assert expected["BSF"] == pytest.approx(70.6, abs=0.1)
        assert expected["FTF"] == pytest.approx(
            CWRU_FTF_FACTOR_6205 * 1797.0 / 60.0, rel=0.005
        )

        engine = calculate_bearing_frequencies(
            num_balls=Z_6205,
            ball_diameter_mm=BD_6205,
            pitch_diameter_mm=PD_6205,
            contact_angle_deg=0.0,
            shaft_speed_rpm=1797.0,
        )
        tool = await calculate_bearing_characteristic_frequencies(
            num_balls=Z_6205,
            ball_diameter_mm=BD_6205,
            pitch_diameter_mm=PD_6205,
            rpm=1797.0,
        )
        for freqs in (engine, tool):
            assert freqs["BSF"] == pytest.approx(expected["BSF"], rel=0.005)
            assert freqs["FTF"] == pytest.approx(expected["FTF"], rel=0.005)

    def test_15_degree_contact_angle_matches_cos_alpha_formulas(self):
        expected = _bearing_kinematics(16, 12.7, 71.5, 15.0, 2000.0)
        freqs = calculate_bearing_frequencies(
            num_balls=16,
            ball_diameter_mm=12.7,
            pitch_diameter_mm=71.5,
            contact_angle_deg=15.0,
            shaft_speed_rpm=2000.0,
        )
        for key in ("BPFO", "BPFI", "BSF", "FTF"):
            assert freqs[key] == pytest.approx(expected[key], rel=0.005), key

    @pytest.mark.parametrize("alpha_deg", [0.0, 15.0, 25.0, 40.0])
    @pytest.mark.parametrize("rpm", [600.0, 1797.0, 3600.0])
    @pytest.mark.parametrize(
        "z, bd_mm, pd_mm", [(Z_6205, BD_6205, PD_6205), (16, 12.7, 71.5)]
    )
    def test_kinematic_identities(self, z, bd_mm, pd_mm, alpha_deg, rpm):
        """BPFO + BPFI = Z fr and BPFO = Z FTF, up to the 0.01 Hz rounding
        of the reported values (each off by at most 0.005 Hz)."""
        freqs = calculate_bearing_frequencies(
            num_balls=z,
            ball_diameter_mm=bd_mm,
            pitch_diameter_mm=pd_mm,
            contact_angle_deg=alpha_deg,
            shaft_speed_rpm=rpm,
        )
        fr = rpm / 60.0
        assert freqs["BPFO"] + freqs["BPFI"] == pytest.approx(z * fr, abs=0.01 + 1e-9)
        assert freqs["BPFO"] == pytest.approx(
            z * freqs["FTF"], abs=(z + 1) * 0.005 + 1e-9
        )

    @pytest.mark.asyncio
    async def test_explicit_geometry_route_refuses_zero_rpm(self, load_csv, mock_ctx):
        await load_csv("inv_brg_geom0", _sine(0.1, 50.0))

        result, refusal = await _outcome(
            check_bearing_faults(
                ctx=mock_ctx,
                signal_id="inv_brg_geom0",
                rpm=0.0,
                num_balls=Z_6205,
                ball_diameter_mm=BD_6205,
                pitch_diameter_mm=PD_6205,
            )
        )

        # The speed is refused, not the all-zero frequencies derived from it
        # (issue #83): the caller is told which value it supplied is wrong.
        assert result is None
        assert "rpm must be positive and finite" in str(refusal)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "rpm",
        [0.0, -1797.0, math.nan, math.inf, -math.inf],
        ids=["zero", "negative", "nan", "inf", "-inf"],
    )
    async def test_catalog_route_never_calls_a_zero_speed_bearing_healthy(
        self, load_csv, mock_ctx, rpm
    ):
        """With no positive finite shaft speed there are no defect frequencies
        to look for: the call is refused by project code, naming the speed.
        Before issue #83 a zero, negative or NaN speed read 'healthy' and an
        infinite one read as a fault."""
        await load_csv("inv_brg_cat0", _sine(0.1, 50.0))

        result, refusal = await _outcome(
            check_bearing_faults(
                ctx=mock_ctx, signal_id="inv_brg_cat0", rpm=rpm, bearing_id="6205"
            )
        )

        assert result is None, (
            f"rpm={rpm} accepted: expected frequencies "
            f"{result.bearing_frequencies}, verdict {result.overall_assessment!r}"
        )
        assert "rpm must be positive and finite" in str(refusal)


# ===========================================================================
# Severity echoes (ISO 20816-3:2022, boundaries from ISO 10816-3:2009)
# ===========================================================================


class TestSeverityEchoes:
    """A severity result says which unit it converted from, whether it
    converted, which band it integrated, and where its thresholds come from."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "unit, amplitude, converted",
        [
            ("g", 0.1, True),
            ("m/s2", 1.0, True),
            ("m/s", 0.002, True),
            ("mm/s", 2.0, False),
        ],
    )
    async def test_signal_route_echoes_unit_conversion_and_provenance(
        self, load_csv, mock_ctx, unit, amplitude, converted
    ):
        await load_csv("inv_sev_unit", _sine(amplitude, 50.0), unit=unit)

        result = await assess_severity(ctx=mock_ctx, signal_id="inv_sev_unit")

        assert result.original_unit == unit
        assert result.unit_conversion_performed is converted
        assert "ISO 10816-3:2009" in result.threshold_provenance
        assert "ISO 20816-3:2022" in result.threshold_provenance

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rpm, band", [(None, "10-1000 Hz"), (400.0, "2-1000 Hz")])
    async def test_signal_route_echoes_the_iso_band_for_the_speed(
        self, load_csv, mock_ctx, rpm, band
    ):
        """ISO 20816-3: 10-1000 Hz, with a 2 Hz lower edge below 600 r/min."""
        await load_csv("inv_sev_band", _sine(0.1, 50.0), unit="g")

        result = await assess_severity(ctx=mock_ctx, signal_id="inv_sev_band", rpm=rpm)

        assert result.frequency_range.startswith(band)
        assert result.operating_speed_rpm == rpm

    @pytest.mark.asyncio
    async def test_direct_route_echoes_mm_s_without_conversion(self, mock_ctx):
        result = await assess_severity(ctx=mock_ctx, rms_velocity_mm_s=3.0)

        assert result.signal_id is None
        assert result.original_unit == "mm/s"
        assert result.unit_conversion_performed is False
        assert result.rms_velocity_mm_s == pytest.approx(3.0)
        assert "ISO 10816-3:2009" in result.threshold_provenance

    @pytest.mark.asyncio
    async def test_custom_thresholds_replace_the_iso_provenance(self, mock_ctx):
        result = await assess_severity(
            ctx=mock_ctx,
            rms_velocity_mm_s=3.0,
            thresholds={"warning": 1.0, "alarm": 2.0, "danger": 5.0},
        )

        assert result.zone == "C"  # 2.0 < 3.0 <= 5.0
        assert "custom" in result.threshold_provenance.lower()
        assert "NOT applied" in result.threshold_provenance


# ===========================================================================
# Provenance: each output echoes the parameters it used
# ===========================================================================


class TestProvenanceEchoes:
    """FFT echoes (fs, sample count, resolution) are pinned in
    ``TestSamplingFrequency``; severity echoes in ``TestSeverityEchoes``."""

    @pytest.mark.asyncio
    async def test_envelope_echoes_signal_id_and_requested_band(
        self, load_csv, mock_ctx
    ):
        rng = np.random.default_rng(800)
        await load_csv("inv_prov_env", rng.standard_normal(int(2 * FS)))

        result = await analyze_envelope(
            ctx=mock_ctx, signal_id="inv_prov_env", filter_low=800.0, filter_high=4200.0
        )

        assert result.signal_id == "inv_prov_env"
        assert tuple(result.filter_band) == (800.0, 4200.0)
        assert result.sampling_rate == FS
        assert result.num_samples == int(FS * 1.0)  # leading 1.0 s segment
        assert "800-4200 Hz" in result.diagnosis

    @pytest.mark.asyncio
    async def test_catalog_route_echoes_speed_frequencies_and_citation(
        self, load_csv, mock_ctx
    ):
        await load_csv("inv_prov_cat", _sine(0.1, 50.0))
        catalog = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
        citation = catalog["bearings"]["6205"]["source"]

        result = await check_bearing_faults(
            ctx=mock_ctx, signal_id="inv_prov_cat", rpm=1797.0, bearing_id="6205"
        )

        expected = _bearing_kinematics(Z_6205, BD_6205, PD_6205, 0.0, 1797.0)
        assert result.signal_id == "inv_prov_cat"
        assert result.bearing_id == "6205"
        assert result.rpm == 1797.0
        assert result.shaft_frequency_hz == pytest.approx(1797.0 / 60.0, abs=0.005)
        assert result.source == citation
        for key, hz in expected.items():
            assert result.bearing_frequencies[key] == pytest.approx(hz, rel=0.005)
        checked = {c.fault_type: c.expected_frequency_hz for c in result.fault_checks}
        assert checked == pytest.approx(
            {k: result.bearing_frequencies[k] for k in expected}, abs=0.005
        )

    @pytest.mark.asyncio
    async def test_explicit_routes_name_the_user_as_source(self, load_csv, mock_ctx):
        await load_csv("inv_prov_user", _sine(0.1, 50.0))

        geometry = await check_bearing_faults(
            ctx=mock_ctx,
            signal_id="inv_prov_user",
            rpm=1500.0,
            num_balls=Z_6205,
            ball_diameter_mm=BD_6205,
            pitch_diameter_mm=PD_6205,
        )
        explicit = await check_bearing_faults(
            ctx=mock_ctx,
            signal_id="inv_prov_user",
            rpm=1500.0,
            frequencies={"GMF": 350.0},
        )

        assert geometry.source == "user-provided geometry"
        assert geometry.bearing_id is None
        assert geometry.shaft_frequency_hz == pytest.approx(25.0)
        assert explicit.source == "user-provided frequencies"
        assert explicit.bearing_frequencies["GMF"] == 350.0
        assert explicit.shaft_frequency_hz == pytest.approx(25.0)


# ===========================================================================
# Invalid inputs
# ===========================================================================


class TestInvalidInputs:
    """Bad input ends in a project-authored refusal or a correct result."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool, good, bad",
        [
            (
                "check_bearing_faults",
                {"signal_id": "inv_wire", "rpm": 1797.0, "bearing_id": "6205"},
                {"signal_id": "inv_wire", "rpm": "fast", "bearing_id": "6205"},
            ),
            ("analyze_fft", {"signal_id": "inv_wire"}, {"signal_id": 123}),
        ],
    )
    async def test_wrong_json_types_over_the_wire_are_errors(
        self, mcp, load_csv, tool, good, bad
    ):
        """Control first: the well-typed call on the same loaded signal
        succeeds, so the error comes from the argument type alone."""
        await load_csv("inv_wire", _sine(0.1, 50.0))

        async with Client(mcp, mode="legacy") as client:
            ok = await client.call_tool(tool, good, read_timeout_seconds=30)
            wrong = await client.call_tool(tool, bad, read_timeout_seconds=30)

        assert not ok.is_error
        assert wrong.is_error

    @pytest.mark.asyncio
    async def test_empty_csv_is_refused_and_nothing_is_stored(
        self, sandbox_data_dir, mock_ctx
    ):
        path = sandbox_data_dir / "inv_empty.csv"
        path.write_text("", encoding="utf-8")

        stored, refusal = await _outcome(
            load_signal(
                mock_ctx,
                str(path),
                signal_id="inv_empty",
                sampling_rate=FS,
                signal_unit="g",
            )
        )
        assert stored is None
        assert "Unable to load signal" in str(refusal)
        assert "inv_empty.csv" in str(refusal)

        result, refusal = await _outcome(
            analyze_fft(ctx=mock_ctx, signal_id="inv_empty")
        )
        assert result is None
        assert "not in the in-memory repository" in str(refusal)

    @pytest.mark.asyncio
    async def test_one_sample_csv_loads_and_fft_refuses(self, load_csv, mock_ctx):
        info = await load_csv("inv_one", [0.5])
        assert info.num_samples == 1

        result, refusal = await _outcome(analyze_fft(ctx=mock_ctx, signal_id="inv_one"))

        assert result is None
        assert "at least 3 samples" in str(refusal)

    @pytest.mark.asyncio
    async def test_all_zero_signal_fft_reports_zero_energy(self, load_csv, mock_ctx):
        await load_csv("inv_zero_fft", np.zeros(int(FS)))

        result = await analyze_fft(ctx=mock_ctx, signal_id="inv_zero_fft")

        assert result.peak_magnitude == 0.0
        assert result.rms_spectral == 0.0
        assert all(p.magnitude == 0.0 for p in result.top_peaks)
        values = [result.peak_frequency, result.frequency_resolution] + [
            v for p in result.top_peaks for v in (p.frequency_hz, p.magnitude_db)
        ]
        assert all(math.isfinite(v) for v in values)

    @pytest.mark.asyncio
    async def test_all_zero_signal_severity_is_zone_a_with_zero_rms(
        self, load_csv, mock_ctx
    ):
        await load_csv("inv_zero_sev", np.zeros(int(FS)), unit="g")

        result = await assess_severity(ctx=mock_ctx, signal_id="inv_zero_sev")

        assert result.rms_velocity_mm_s == 0.0
        assert result.zone == "A"

    @pytest.mark.asyncio
    async def test_nan_sample_is_refused_or_yields_a_finite_severity(
        self, load_csv, mock_ctx
    ):
        signal = _sine(0.1, 50.0)
        signal[100] = np.nan

        async def load_then_assess():
            await load_csv("inv_nan", signal, unit="g")
            return await assess_severity(ctx=mock_ctx, signal_id="inv_nan")

        result, refusal = await _outcome(load_then_assess())

        if refusal is None:
            assert math.isfinite(result.rms_velocity_mm_s), (
                f"NaN sample accepted: zone {result.zone} with "
                f"rms_velocity_mm_s={result.rms_velocity_mm_s}"
            )
        else:
            assert "1 non-finite" in str(refusal)

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        strict=True,
        raises=ValueError,
        reason=known_defect(
            "envelope-short-signal",
            "analyze_envelope on a 20-sample signal surfaces scipy's padlen "
            "ValueError, not a project refusal",
        ),
    )
    async def test_envelope_on_a_too_short_signal_is_a_project_refusal(
        self, load_csv, mock_ctx
    ):
        await load_csv("inv_env_short", _sine(0.1, 50.0)[:20])

        await _outcome(analyze_envelope(ctx=mock_ctx, signal_id="inv_env_short"))

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        strict=True,
        raises=ValueError,
        reason=known_defect(
            "fft-max-frequency-below-first-bin",
            "analyze_fft surfaces numpy's empty-sequence argmax ValueError",
        ),
    )
    async def test_fft_max_frequency_below_first_bin_is_a_project_refusal(
        self, load_csv, mock_ctx
    ):
        await load_csv("inv_fft_maxf", _sine(0.1, 50.0))  # 1 s: first bin 1 Hz

        await _outcome(
            analyze_fft(ctx=mock_ctx, signal_id="inv_fft_maxf", max_frequency=0.5)
        )

    @pytest.mark.asyncio
    async def test_infinite_sampling_rate_is_refused(self, load_csv):
        info, refusal = await _outcome(
            load_csv("inv_fs_inf", _sine(0.1, 50.0), fs=math.inf)
        )

        assert refusal is not None, (
            f"sampling_rate=inf accepted: stored sampling_rate={info.sampling_rate}, "
            f"duration_s={info.duration_s}"
        )
        assert "sampling rate must be positive" in str(refusal)
        assert "inv_fs_inf" not in _loaded_ids()

    @pytest.mark.asyncio
    async def test_nan_sampling_rate_is_a_project_refusal(self, load_csv):
        _, refusal = await _outcome(
            load_csv("inv_fs_nan", _sine(0.1, 50.0), fs=math.nan)
        )

        assert refusal is not None
        assert "sampling rate must be positive" in str(refusal)
        assert "inv_fs_nan" not in _loaded_ids()
