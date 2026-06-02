import yaml
from pathlib import Path
from typing import Any, Dict

def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML content from a file into a Python dictionary.

    Args:
        path: Path or string representing the YAML file location.

    Returns:
        A dictionary containing the parsed YAML content. Returns an empty
        dictionary if the file contains no data.

    Raises:
        FileNotFoundError: If the given path does not exist.
        yaml.YAMLError: If the YAML content cannot be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f) or {}