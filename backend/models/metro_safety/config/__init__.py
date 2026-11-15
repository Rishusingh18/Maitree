"""Configuration loader and schema validator."""
from pathlib import Path
import yaml
from typing import Dict, Any, Optional

def _find_default_config() -> Path:
    candidates = [
        Path("config.yaml"),
        Path(__file__).resolve().parent.parent.parent / "config.yaml",
        Path(__file__).resolve().parent.parent / "config.yaml",
        Path(__file__).resolve().parents[2] / "config.yaml" if len(Path(__file__).resolve().parents) > 2 else None,
        Path(__file__).resolve().parents[3] / "config.yaml" if len(Path(__file__).resolve().parents) > 3 else None,
        Path(__file__).resolve().parents[4] / "config.yaml" if len(Path(__file__).resolve().parents) > 4 else None,
        Path("backend/models/config.yaml"),
    ]
    for c in candidates:
        if c and c.exists():
            return c.resolve()
    return (Path(__file__).resolve().parent.parent.parent / "config.yaml").resolve()

DEFAULT_CONFIG_PATH = _find_default_config()

def load_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Load system configuration from YAML file."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
