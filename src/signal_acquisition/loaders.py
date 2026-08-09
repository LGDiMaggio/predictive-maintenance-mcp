"""
Signal loading and segmentation utilities.

Funzioni pure senza dipendenze da MCP o dal filesystem globale:
- load_signal_data: carica un file segnale in qualsiasi formato supportato
- load_raw_binary: decodifica file binari raw headerless (.bin/.raw/.dat)
- extract_segment: estrae un segmento casuale/deterministico da un segnale
- get_metadata_path / get_metadata_path_from_dir: derivano il path del file metadata

Queste funzioni sono testabili in isolamento senza avviare il server MCP.

Error contract: the historical functions in this module return ``Optional``
(``None`` on failure). ``load_raw_binary`` deliberately deviates and RAISES
typed errors ("problem — remedy" ``ValueError``s, plus a bare
``FileNotFoundError``): the caller must be able to relay exactly which
declared parameter contradicts the file, which a bare ``None`` cannot
express. See its docstring.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import DATA_DIR, get_max_signal_size
from ..path_safety import safe_resolve

logger = logging.getLogger(__name__)

# Extensions eligible for raw-binary decoding (headerless files that need an
# explicit declaration of sample_format/byte_order/... to be loadable).
RAW_EXTENSIONS = frozenset({".bin", ".raw", ".dat"})

#: Formats whose header/structure declares how to decode them — no raw
#: declaration is needed (or allowed) for these.
SELF_DESCRIBING_EXTENSIONS = [".csv", ".txt", ".npy", ".mat", ".wav", ".parquet"]

#: Everything list_signals(scope='disk') surfaces. Composed from the two
#: classes above so a future raw extension can never be raw-eligible but
#: unlisted (the inverse of the .dat bug this composition replaced).
SUPPORTED_EXTENSIONS = [*SELF_DESCRIBING_EXTENSIONS, *sorted(RAW_EXTENSIONS)]

# Closed vocabularies for the raw decoder declaration. Keys are the accepted
# parameter values; values are the numpy dtype building blocks.
_RAW_DTYPE_CODES = {
    "float32": "f4",
    "float64": "f8",
    "int16": "i2",
    "int32": "i4",
}
_RAW_BYTE_ORDER_PREFIXES = {"little": "<", "big": ">"}

#: Public closed vocabularies of the raw declaration — the SAME sets the
#: decoder enforces, exported so the repository layer can validate
#: companion-metadata values (json.load applies no Literal) with errors that
#: name the valid vocabulary, mirroring VALID_SIGNAL_UNITS for units.
VALID_SAMPLE_FORMATS: tuple[str, ...] = tuple(sorted(_RAW_DTYPE_CODES))
VALID_BYTE_ORDERS: tuple[str, ...] = tuple(sorted(_RAW_BYTE_ORDER_PREFIXES))

#: Effective defaults for the OPTIONAL raw decode parameters — the single
#: source of truth for the decoder signature's literal defaults (pinned by
#: test) and the repository's explicit > companion > default merge.
#: ``sample_format`` (like ``sampling_rate``) is deliberately absent: for a
#: raw file it is a REQUIRED declaration with no default.
RAW_PARAM_DEFAULTS: dict[str, object] = {
    "byte_order": "little",
    "n_channels": 1,
    "channel_index": 0,
    "header_offset": 0,
    "scale_factor": None,
}


def load_signal_data(filename: str) -> Optional[np.ndarray]:
    """
    Load signal data from file.

    Supported formats:
        - .csv, .txt: Comma/tab-separated values (first column used)
        - .npy: NumPy binary array
        - .mat: MATLAB files (first numeric variable used)
        - .wav: WAV audio files (first channel, normalized to [-1, 1])
        - .parquet: Apache Parquet files (first column used)

    Args:
        filename: File name relative to data/signals/

    Returns:
        Numpy array with data or None if error
    """
    try:
        # Contain the user-supplied filename inside DATA_DIR before any stat/read
        # so a traversal (e.g. '../../secret.csv') cannot exfiltrate file contents.
        # safe_resolve raises on escape; the surrounding except maps it to None,
        # preserving this function's "return None on failure" contract.
        file_path = safe_resolve(DATA_DIR, filename)

        if not file_path.exists():
            return None

        if file_path.suffix == ".npy":
            return np.load(file_path)

        elif file_path.suffix in [".csv", ".txt"]:
            df = pd.read_csv(file_path, header=None)
            return df.iloc[:, 0].values

        elif file_path.suffix == ".mat":
            from scipy.io import loadmat

            mat_data = loadmat(str(file_path))
            for key, value in mat_data.items():
                if key.startswith("__"):
                    continue
                if isinstance(value, np.ndarray) and value.dtype.kind in (
                    "f",
                    "i",
                    "u",
                ):
                    data = value.flatten()
                    if len(data) > 0:
                        return data.astype(np.float64)
            logger.warning(f"No numeric data found in MAT file: {filename}")
            return None

        elif file_path.suffix == ".wav":
            from scipy.io import wavfile

            sample_rate, data = wavfile.read(str(file_path))
            if data.ndim > 1:
                data = data[:, 0]
            if data.dtype.kind == "i":
                data = data.astype(np.float64) / np.iinfo(data.dtype).max
            return data.astype(np.float64)

        elif file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
            return df.iloc[:, 0].values.astype(np.float64)

        return None

    except Exception as e:
        logger.error(f"Error loading signal {filename}: {e}")
        return None


def load_raw_binary(
    path: Path,
    *,
    sample_format: str,
    byte_order: str = "little",
    n_channels: int = 1,
    channel_index: int = 0,
    header_offset: int = 0,
    scale_factor: Optional[float] = None,
) -> np.ndarray:
    """Decode a headerless raw binary signal file into a float64 array.

    Pure decoder for raw waveform files (see ``RAW_EXTENSIONS``): validates
    the caller's declaration against the file's actual size, reads the whole
    payload eagerly with ``np.fromfile``, de-interleaves the selected channel
    and returns float64 samples. Integer sample formats are returned as raw
    counts — there is deliberately NO WAV-style implicit normalization;
    declare a ``scale_factor`` to convert ADC counts to physical units (the
    multiply is performed in float64).

    Deviation from the module convention: unlike the ``Optional``-returning
    loaders in this module, this function RAISES typed errors with
    actionable "problem — remedy" messages, because the caller must relay
    exactly which declared parameter contradicts the file.

    ``path`` must already be resolved and contained by the caller
    (``safe_resolve`` at the existing call sites); no containment is applied
    here.

    Args:
        path: Already-resolved filesystem path of the raw binary file.
        sample_format: Declared sample dtype: 'float32', 'float64', 'int16'
            or 'int32'.
        byte_order: Declared endianness: 'little' (default) or 'big'.
        n_channels: Number of interleaved channels in the file (>= 1).
        channel_index: 0-based channel to extract (< n_channels).
        header_offset: Bytes to skip before the first sample (>= 0).
        scale_factor: Optional multiplier applied after decoding (e.g. ADC
            counts -> physical unit). Default: no scaling.

    Returns:
        1-D float64 numpy array with the selected channel's samples.

    Raises:
        ValueError: If a declared parameter is outside its vocabulary or
            range, the file exceeds the PMM_MAX_SIGNAL_SIZE cap, the payload
            size contradicts the declaration (the message shows the
            arithmetic), or a float payload decodes to NaN/Inf samples.
        FileNotFoundError: If ``path`` does not exist.
    """
    if sample_format not in _RAW_DTYPE_CODES:
        raise ValueError(
            f"Invalid sample_format '{sample_format}' — declare one of "
            f"{sorted(_RAW_DTYPE_CODES)}."
        )
    if byte_order not in _RAW_BYTE_ORDER_PREFIXES:
        raise ValueError(
            f"Invalid byte_order '{byte_order}' — declare one of "
            f"{sorted(_RAW_BYTE_ORDER_PREFIXES)} ('little' is the default)."
        )
    # Bounds checks come BEFORE n_channels is used as a divisor below, so a
    # bad declaration can never surface as a raw ZeroDivisionError.
    if n_channels < 1:
        raise ValueError(
            f"n_channels must be >= 1, got {n_channels} — declare how many "
            f"interleaved channels the file contains (1 for a single-channel "
            f"recording)."
        )
    if channel_index < 0:
        raise ValueError(
            f"channel_index must be >= 0, got {channel_index} — channels "
            f"are 0-indexed."
        )
    if channel_index >= n_channels:
        raise ValueError(
            f"channel_index {channel_index} is out of range for "
            f"n_channels={n_channels} — valid values are "
            f"0..{n_channels - 1}."
        )
    if header_offset < 0:
        raise ValueError(
            f"header_offset must be >= 0, got {header_offset} — declare the "
            f"header size in bytes (0 for a headerless file)."
        )

    # Pre-read size guard: stat() only, so an oversized file is refused
    # before a single payload byte is read into memory. A missing file is a
    # bare FileNotFoundError — the actionable "use list_signals" remedy is
    # the repository layer's business (it checks existence on every route
    # and owns the canonical message).
    try:
        file_size = path.stat().st_size
    except FileNotFoundError:
        raise FileNotFoundError(str(path)) from None
    max_size = get_max_signal_size()
    if file_size > max_size:
        raise ValueError(
            f"File {path} is {file_size} bytes, over the {max_size}-byte "
            f"cap — raise the cap via the PMM_MAX_SIGNAL_SIZE environment "
            f"variable (bytes) if this size is intentional."
        )

    dtype = np.dtype(
        _RAW_BYTE_ORDER_PREFIXES[byte_order] + _RAW_DTYPE_CODES[sample_format]
    )
    dtype_size = dtype.itemsize
    frame_size = dtype_size * n_channels
    payload = file_size - header_offset

    if payload <= 0:
        raise ValueError(
            f"No payload to decode: file size {file_size} bytes minus "
            f"header_offset {header_offset} leaves {payload} bytes — check "
            f"header_offset against the actual file size (an empty or "
            f"truncated file has nothing to decode)."
        )
    if payload % frame_size != 0:
        raise ValueError(
            f"Payload of {payload} bytes (file size {file_size} - "
            f"header_offset {header_offset}) is not a whole number of "
            f"{frame_size}-byte frames ({sample_format} = {dtype_size} bytes "
            f"x {n_channels} channel(s)): remainder {payload % frame_size} "
            f"byte(s) — the declared sample_format/n_channels/header_offset "
            f"do not fit this file; re-check the declaration."
        )

    # Eager full read with the declared dtype (explicit endianness code).
    data = np.fromfile(path, dtype=dtype, offset=header_offset)
    if n_channels > 1:
        data = data[channel_index::n_channels]

    if dtype.kind == "f":
        nonfinite = data.size - int(np.count_nonzero(np.isfinite(data)))
        if nonfinite:
            raise ValueError(
                f"Decoded payload contains {nonfinite} non-finite sample(s) "
                f"(NaN/Inf) out of {data.size} — the declared sample_format "
                f"'{sample_format}' or byte_order '{byte_order}' probably "
                f"does not match how the file was written; re-check the "
                f"declaration."
            )

    result: np.ndarray
    if scale_factor is not None:
        # Convert + scale in a single float64 pass (one temporary).
        result = np.multiply(data, scale_factor, dtype=np.float64)
    else:
        # No copy when the payload is already native float64.
        result = data.astype(np.float64, copy=False)
    return result


def extract_segment(
    signal: np.ndarray,
    duration_s: float,
    sampling_rate: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Estrae un segmento da un segnale, con posizione di partenza casuale.

    Usa un generatore locale (numpy.random.default_rng) per evitare
    la mutazione dello stato globale di numpy.

    Args:
        signal: Array del segnale completo
        duration_s: Durata del segmento in secondi
        sampling_rate: Frequenza di campionamento in Hz
        seed: Seme per riproducibilita' (None = casuale)

    Returns:
        Segmento estratto come numpy array
    """
    segment_samples = int(duration_s * sampling_rate)
    signal_length = len(signal)

    if segment_samples >= signal_length:
        return signal

    max_start = signal_length - segment_samples
    rng = np.random.default_rng(seed)
    start_idx = int(rng.integers(0, max_start + 1))

    return signal[start_idx : start_idx + segment_samples]


def get_metadata_path(signal_filename: str) -> Path:
    """
    Deriva il path del file metadata JSON da un filename di segnale.

    Funziona con tutte le estensioni supportate: .csv, .txt, .npy, .mat, .wav, .parquet

    Args:
        signal_filename: Nome del file segnale (relativo a DATA_DIR)

    Returns:
        Path al corrispondente file _metadata.json (contenuto dentro DATA_DIR)

    Raises:
        ValueError: If signal_filename escapes DATA_DIR (path traversal).
    """
    # Route the derived metadata path through safe_resolve so containment
    # can't be forgotten here, mirroring load_signal_data. Behavior is
    # identical for valid in-DATA_DIR inputs; a traversal now raises instead
    # of silently pointing outside DATA_DIR.
    p = Path(signal_filename)
    rel_meta = p.parent / f"{p.stem}_metadata.json"
    return safe_resolve(DATA_DIR, str(rel_meta))


def get_metadata_path_from_dir(data_dir: Path, signal_filename: str) -> Path:
    """Deriva il path metadata a partire da una directory specifica."""
    stem = Path(signal_filename).stem
    return data_dir / f"{stem}_metadata.json"
