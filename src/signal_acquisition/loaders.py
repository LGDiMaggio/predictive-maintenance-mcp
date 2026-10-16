"""
Signal loading and segmentation utilities.

Funzioni pure senza dipendenze da MCP o dal filesystem globale:
- load_signal_data: carica un file segnale in qualsiasi formato supportato
- load_raw_binary: decodifica file binari raw headerless (.bin/.raw/.dat)
- check_text_blank_rows: refuses a .csv/.txt file with an empty line between
  two data lines (a missing sample); its callers run it before decoding
- extract_segment: estrae un segmento casuale/deterministico da un segnale
- get_metadata_path / get_metadata_path_from_dir: derivano il path del file metadata

Queste funzioni sono testabili in isolamento senza avviare il server MCP.

Error contract: the historical functions in this module return ``Optional``
(``None`` on failure). ``load_raw_binary`` and ``check_text_blank_rows``
deliberately deviate and RAISE typed errors ("problem — remedy"
``ValueError``s, plus a bare ``FileNotFoundError`` from the raw decoder): the
caller must be able to relay exactly what is wrong with the file, which a
bare ``None`` cannot express. See their docstrings.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import DATA_DIR, get_max_signal_size
from ..path_safety import safe_resolve
from ..signal_validation import BAD_SAMPLE_REMEDY, validate_finite_samples

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

#: Remedy for non-finite samples decoded from a raw file, for the callers
#: that validate after :func:`load_raw_binary`. The decoder already refuses a
#: non-finite float payload before scaling, so what is left is a declaration
#: that does not fit the file, most often the scale_factor.
RAW_SAMPLES_REMEDY = (
    "re-check the raw declaration: a sample_format or byte_order that does "
    "not match how the file was written, or a scale_factor that is NaN, "
    "infinite or large enough to overflow, decodes to non-finite samples; if "
    "the declaration is right, " + BAD_SAMPLE_REMEDY
)

# Blank-row scan (check_text_blank_rows). A run of \r followed by \n is ONE
# line break (Python's csv module on Windows writes \r\r\n), and a lone \n or
# \r is one break each. A chunk whose every \r sits right before a \n is
# scanned as is, with \r\n read as one break; any other chunk is first
# normalized so that every break is a single \n.
_LINE_BREAK = re.compile(rb"\r*\n|\r")
# Any byte that is not ASCII whitespace makes its line a data line.
_DATA_BYTE = re.compile(rb"[^ \t\r\n\x0b\x0c]")
# A whole blank line: its opening break, whitespace, its closing break.
_BLANK_LINE = re.compile(rb"\n[ \t\x0b\x0c]*\r?\n")
# The rest of a line that began blank in the previous chunk, up to its break.
_BLANK_LINE_REST = re.compile(rb"[ \t\x0b\x0c]*\r?\n")
# Read size of the scan: memory stays bounded whatever the file size.
_SCAN_CHUNK_BYTES = 1 << 20


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

        return load_self_describing(file_path)

    except Exception as e:
        logger.error(f"Error loading signal {filename}: {e}")
        return None


def load_self_describing(path: Path) -> Optional[np.ndarray]:
    """Decode an already-resolved self-describing signal file.

    The format dispatch of :func:`load_signal_data`, on a path the caller
    has already contained (``safe_resolve``) or deliberately accepts as
    absolute (the asset ledger re-reads a recorded location only after
    verifying the file's content hash against the ledger). No containment
    and no existence check are applied here, and nothing is swallowed:
    ``load_signal_data`` keeps its "None on failure" contract by catching
    what this function raises.

    Args:
        path: Resolved filesystem path of a ``SELF_DESCRIBING_EXTENSIONS``
            file.

    Returns:
        The first column / channel as a 1-D array (WAV integer PCM
        normalized to [-1, 1] as before); None for an unsupported suffix or
        a MAT file without numeric data.

    Raises:
        OSError, ValueError: From the decoders (missing or malformed file).
    """
    if path.suffix == ".npy":
        return np.load(path)

    if path.suffix in (".csv", ".txt"):
        df = pd.read_csv(path, header=None)
        return df.iloc[:, 0].values

    if path.suffix == ".mat":
        from scipy.io import loadmat

        mat_data = loadmat(str(path))
        for key, value in mat_data.items():
            if key.startswith("__"):
                continue
            if isinstance(value, np.ndarray) and value.dtype.kind in ("f", "i", "u"):
                data = value.flatten()
                if len(data) > 0:
                    return data.astype(np.float64)
        logger.warning(f"No numeric data found in MAT file: {path.name}")
        return None

    if path.suffix == ".wav":
        from scipy.io import wavfile

        sample_rate, data = wavfile.read(str(path))
        if data.ndim > 1:
            data = data[:, 0]
        if data.dtype.kind == "i":
            data = data.astype(np.float64) / np.iinfo(data.dtype).max
        return data.astype(np.float64)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        return df.iloc[:, 0].values.astype(np.float64)

    return None


def check_text_blank_rows(path: Path) -> None:
    r"""Refuse a text signal file that has an empty line between data lines.

    The CSV reader skips blank lines, so an empty or whitespace-only line in
    the middle of a ``.csv``/``.txt`` capture would silently drop a sample
    and shift every later sample in time. This scan refuses the file at the
    first such line instead. Blank lines before the first data line and
    after the last one (a leading empty line, trailing newlines) pass.

    The file is read in binary, in fixed-size chunks, so the scan never
    raises a decoding error (a cp1252 header, for example) and its memory
    stays bounded whatever the file size. A line is blank when it is empty
    or holds only ASCII whitespace. A run of ``\r`` immediately followed by
    ``\n`` ends ONE line (Python's ``csv`` module on Windows writes
    ``\r\r\n``); a lone ``\n`` or ``\r`` ends one line each.

    This is a separate call, not part of :func:`load_self_describing`,
    because the signal store's main route swallows every exception raised
    inside that decoder. The signal store and the asset ledger call it on
    ``.csv`` and ``.txt`` files before decoding them.

    Args:
        path: Resolved filesystem path of a ``.csv`` or ``.txt`` file.

    Raises:
        ValueError: If an empty or whitespace-only line has a data line both
            before and after it; the message names the file and the 1-based
            line number.
        OSError: If the file cannot be opened or read.
    """
    seen_data = False  # a data line has started
    blank_line: Optional[int] = None  # first blank line after the first data line
    line_is_blank = False  # the line in progress holds only whitespace so far
    breaks = 0  # line breaks before ``pos``, all chunks included
    carry = b""
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_SCAN_CHUNK_BYTES)
            if chunk:
                # A trailing run of \r may end in a \n that the next read brings.
                data = carry + chunk
                body = data.rstrip(b"\r")
                carry = data[len(body) :]
            else:
                body, carry = carry, b""
            carriage_returns = body.count(b"\r")
            if carriage_returns and carriage_returns != body.count(b"\r\n"):
                # Not plain \r\n breaks: make every break a single \n. Two
                # plain replaces cover \r\n and \r\r\n (a regex substitution
                # per line costs ~10x more). A \r still right before a \n
                # means a longer run, so the regex takes the chunk; with
                # none, every \r left is a lone break.
                body = body.replace(b"\r\n", b"\n").replace(b"\r\n", b"\n")
                if b"\r\n" in body:
                    body = _LINE_BREAK.sub(b"\n", body)
                else:
                    body = body.replace(b"\r", b"\n")

            pos = 0
            if not seen_data:
                first = _DATA_BYTE.search(body)
                if first is None:
                    breaks += body.count(b"\n")
                else:
                    breaks += body.count(b"\n", 0, first.start())
                    pos = first.start()
                    seen_data = True

            if seen_data and blank_line is None:
                start = pos
                blank = _BLANK_LINE_REST.match(body, pos) if line_is_blank else None
                if blank is None:
                    blank = _BLANK_LINE.search(body, pos)
                    if blank is not None:
                        start = blank.start() + 1
                if blank is not None:
                    blank_line = breaks + body.count(b"\n", pos, start) + 1
                    pos = blank.end()
                else:
                    breaks += body.count(b"\n", pos)
                    last_break = body.rfind(b"\n", pos)
                    if last_break >= 0:
                        rest = _DATA_BYTE.search(body, last_break + 1)
                        line_is_blank = rest is None
                    elif _DATA_BYTE.search(body, pos) is not None:
                        line_is_blank = False

            # Only a data line AFTER the blank line makes it a missing sample.
            if blank_line is not None and _DATA_BYTE.search(body, pos) is not None:
                raise ValueError(
                    f"File '{path.name}' has an empty line at line {blank_line} "
                    f"between data lines, which is a missing sample — "
                    f"{BAD_SAMPLE_REMEDY}."
                )
            if not chunk:
                return


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

    # Before scaling, on the float payload only (integer counts are always
    # finite): here a NaN or infinity means the declaration does not fit the
    # bytes. Callers check again after decoding, with RAW_SAMPLES_REMEDY.
    if dtype.kind == "f":
        validate_finite_samples(
            data,
            source="Decoded payload",
            remedy=(
                f"the declared sample_format '{sample_format}' or byte_order "
                f"'{byte_order}' probably does not match how the file was "
                f"written; re-check the declaration, and if it is right, "
                + BAD_SAMPLE_REMEDY
            ),
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

    Funziona con tutte le estensioni in ``SUPPORTED_EXTENSIONS`` (il path
    metadata è derivato dallo stem, quindi è indipendente dal formato).

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
