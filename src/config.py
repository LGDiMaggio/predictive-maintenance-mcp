"""
Configurazione centralizzata dei path per il server MCP.

Questo modulo e' l'unico punto di verita' per tutti i path del progetto.
Gli altri moduli importano da qui invece di calcolare path propri.

Nessun side-effect a import-time: nessun mkdir, nessun logging.
"""

import os
import re
from pathlib import Path


def resolve_project_root() -> Path:
    """
    Trova la directory root del progetto.

    Priorita':
    1. Variabile d'ambiente PDM_PROJECT_DIR (configurazione esplicita)
    2. CWD se contiene data/signals/ (repo clonato, esecuzione diretta)
    3. Relativo a __file__ (installazione pip, src/ -> parent e' il root)
    4. Fallback a CWD (l'utente deve creare data/ manualmente)
    """
    env_dir = os.environ.get("PDM_PROJECT_DIR")
    if env_dir and Path(env_dir).is_dir():
        return Path(env_dir)

    cwd = Path.cwd()
    if (cwd / "data" / "signals").is_dir():
        return cwd

    file_based = Path(__file__).parent.parent
    if (file_based / "data" / "signals").is_dir():
        return file_based

    return cwd


def get_max_signal_size() -> int:
    """Maximum signal file size in bytes accepted by the raw decoder.

    Reads ``PMM_MAX_SIGNAL_SIZE`` from the environment at EACH call —
    mirroring the call-time read of ``PMM_SIGNAL_CACHE_GB`` in
    ``get_repository()`` — instead of freezing it at import time, so tests
    and operators can override it with a plain env change (monkeypatch)
    without reimporting this module.

    Returns:
        Cap in bytes. Default: 500_000_000 (500 MB) when the variable is
        unset.
    """
    return int(os.environ.get("PMM_MAX_SIGNAL_SIZE", "500000000"))


def get_ledger_dir() -> Path:
    """Directory of the append-only asset ledger (one JSONL file per asset).

    Reads ``PMM_LEDGER_DIR`` from the environment at EACH call (the
    ``get_max_signal_size`` pattern) so tests and operators can redirect it
    with a plain env change; a set-but-blank value counts as unset. Nothing
    is created here: the ledger store creates the directory on first append
    and ``server._setup_environment`` creates it at startup (warning when it
    is not writable or lies under a cloud-synced folder).

    Returns:
        The configured directory (not resolved, not created). Default:
        ``PROJECT_ROOT / "data" / "ledger"``.
    """
    configured = (os.environ.get("PMM_LEDGER_DIR") or "").strip()
    if configured:
        return Path(configured)
    return PROJECT_ROOT / "data" / "ledger"


def get_import_dir() -> Path:
    """Directory of the signal import's record store.

    Reads ``PMM_IMPORT_DIR`` at EACH call (the ``get_ledger_dir`` pattern);
    a set-but-blank value counts as unset. Nothing is created here: the
    record store creates the directory on its first write and
    ``server._setup_environment`` creates it at startup (warning when it is
    not writable, cannot hold hard links, or lies under a cloud-synced
    folder).

    Returns:
        The configured directory (not resolved, not created). Default:
        ``PROJECT_ROOT / "data" / "imports"``.
    """
    configured = (os.environ.get("PMM_IMPORT_DIR") or "").strip()
    if configured:
        return Path(configured)
    return PROJECT_ROOT / "data" / "imports"


def get_import_roots() -> tuple[Path, ...]:
    """Folders the signal import may read: the data directory, then the
    entries of ``PMM_IMPORT_ROOTS``.

    Reads the variable at EACH call (the ``get_ledger_dir`` pattern); the
    entries are separated by ``os.pathsep`` (``;`` on Windows, ``:``
    elsewhere), and a set-but-blank value counts as unset. ``DATA_DIR`` is
    read at call time too, so a redirected data directory is honored.

    Returns:
        ``(DATA_DIR, *entries)`` as given (not resolved, not created).

    Raises:
        ValueError: For an empty entry, a relative entry, a root-relative
            Windows entry without a drive letter, or a UNC entry
            (``\\\\server\\share``, ``//server/share``): a network share is
            listed through a mapped drive letter, never as a UNC path.
    """
    configured = os.environ.get("PMM_IMPORT_ROOTS") or ""
    if not configured.strip():
        return (DATA_DIR,)
    roots = [DATA_DIR]
    for entry in configured.split(os.pathsep):
        problem = None
        if not entry.strip():
            problem = "has an empty entry"
        elif entry.startswith(("\\\\", "//")):
            problem = f"lists the UNC path {entry!r}"
        elif os.name == "nt" and not re.match(r"[A-Za-z]:[\\/]", entry):
            problem = f"lists {entry!r}, which is not an absolute path with a drive"
        elif not os.path.isabs(entry):
            problem = f"lists the relative path {entry!r}"
        if problem is not None:
            raise ValueError(
                f"PMM_IMPORT_ROOTS {problem}. List absolute folder paths "
                f"separated by {os.pathsep!r}; reach a network share through a "
                f"mapped drive letter."
            )
        roots.append(Path(entry))
    return tuple(roots)


PROJECT_ROOT = resolve_project_root()

DATA_DIR = PROJECT_ROOT / "data" / "signals"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESOURCES_DIR = PROJECT_ROOT / "resources"
CACHE_DIR = RESOURCES_DIR / "cache"
