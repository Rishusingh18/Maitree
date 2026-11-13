"""Configuration loader and schema validator."""
from pathlib import Path
import yaml
from typing import Dict, Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

def load_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Load system configuration from YAML file."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
