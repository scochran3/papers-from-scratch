"""Config loading for the LIMA project."""

from pathlib import Path
from typing import Any

import yaml

from lima.paths import DEFAULT_CONFIG


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config.

    Args:
        path: Config file to read. Defaults to configs/qwen.yaml. A relative
            path is resolved against the current working directory, which is
            what you want for a value typed on the command line.

    Returns:
        The parsed config as a dict.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"No config at {config_path}")
    with config_path.open() as f:
        return yaml.safe_load(f)
