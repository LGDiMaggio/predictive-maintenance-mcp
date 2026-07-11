"""
Signal loading and segmentation utilities.

Funzioni pure senza dipendenze da MCP o dal filesystem globale:
- load_signal_data: carica un file segnale in qualsiasi formato supportato
- extract_segment: estrae un segmento casuale/deterministico da un segnale
- get_metadata_path / get_metadata_path_from_dir: derivano il path del file metadata

Queste funzioni sono testabili in isolamento senza avviare il server MCP.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import DATA_DIR
from ..path_safety import safe_resolve

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = [".csv", ".txt", ".npy", ".dat", ".mat", ".wav", ".parquet"]


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
                if key.startswith('__'):
                    continue
                if isinstance(value, np.ndarray) and value.dtype.kind in ('f', 'i', 'u'):
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
            if data.dtype.kind == 'i':
                data = data.astype(np.float64) / np.iinfo(data.dtype).max
            return data.astype(np.float64)

        elif file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
            return df.iloc[:, 0].values.astype(np.float64)

        return None

    except Exception as e:
        logger.error(f"Error loading signal {filename}: {e}")
        return None


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

    return signal[start_idx:start_idx + segment_samples]


def get_metadata_path(signal_filename: str) -> Path:
    """
    Deriva il path del file metadata JSON da un filename di segnale.

    Funziona con tutte le estensioni supportate: .csv, .txt, .npy, .mat, .wav, .parquet

    Args:
        signal_filename: Nome del file segnale (relativo a DATA_DIR)

    Returns:
        Path al corrispondente file _metadata.json
    """
    p = Path(signal_filename)
    return DATA_DIR / p.parent / f"{p.stem}_metadata.json"


def get_metadata_path_from_dir(data_dir: Path, signal_filename: str) -> Path:
    """Deriva il path metadata a partire da una directory specifica."""
    stem = Path(signal_filename).stem
    return data_dir / f"{stem}_metadata.json"
