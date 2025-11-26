import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f) or {}
    
@dataclass
class RAGConfig:
    embedding_model_name: str
    chunk_size: int
    reader_model_name: str
    vectorstore_path: Path
    prompt_file: Path

def load_rag_config(
    base_path: str | Path = PROJECT_ROOT / "conf/base/rag_config.yml",
) -> RAGConfig:
    """
    Load the RAG configuration from YAML to dataclass RAGConfig to configure pipeline. 

    - base_path: Path(conf/base/rag_config.yml)
    """
    base_path = Path(base_path)
    base_cfg = _load_yaml(base_path)

    return RAGConfig(**base_cfg)

def load_prompt_template(
    prompt_file: str | Path,
    prompt_key: str = "default_prompt",
) -> List[Dict[str, Any]]:
    """
    Load a chat template (list of {role, content} dicts) from a YAML file.

    Example YAML structure:

    default_prompt:
      - role: system
        content: "..."
      - role: user
        content: "..."
    """
    prompt_file = PROJECT_ROOT / prompt_file
    path = Path(prompt_file)
    data = _load_yaml(path)

    if prompt_key not in data:
        available = ", ".join(sorted(data.keys()))
        raise KeyError(
            f"Prompt key '{prompt_key}' not found in {prompt_file}. "
            f"Available keys: {available}"
        )

    template = data[prompt_key]
    if not isinstance(template, list):
        raise ValueError(
            f"Prompt '{prompt_key}' in {prompt_file} is not a list of messages."
        )

    return template
