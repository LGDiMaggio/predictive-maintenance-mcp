"""
Configurazione centralizzata dei path per il server MCP.

Questo modulo e' l'unico punto di verita' per tutti i path del progetto.
Gli altri moduli importano da qui invece di calcolare path propri.

Nessun side-effect a import-time: nessun mkdir, nessun logging.
"""

import os
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


PROJECT_ROOT = resolve_project_root()

DATA_DIR = PROJECT_ROOT / "data" / "signals"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESOURCES_DIR = PROJECT_ROOT / "resources"
CACHE_DIR = RESOURCES_DIR / "cache"
