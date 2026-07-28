"""Filesystem anchors for the LIMA project.

Every path in this project is derived from PROJECT_ROOT, which is computed from
*this file's* location on disk. 
"""

from pathlib import Path

# __file__            -> .../LIMA/lima/paths.py
# .resolve()          -> absolute, symlinks expanded
# .parent             -> .../LIMA/lima
# .parent.parent      -> .../LIMA          <- the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

DEFAULT_CONFIG = CONFIGS_DIR / "qwen.yaml"
