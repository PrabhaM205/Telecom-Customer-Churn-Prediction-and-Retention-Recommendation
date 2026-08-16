# src/config.py
"""
Centralized configuration loader.

Reads config.yaml (non-secret, environment-independent thresholds/paths)
and resolves relative paths against the PROJECT ROOT -- not the current
working directory. This is what fixes the bug where genai/rag/retriever.py
and vector_store.py only worked when a script happened to be run from the
repo root: any other caller (Streamlit, the agent, a future API) would
silently fail to find the FAISS index.

Secrets (API keys) stay in .env / os.environ -- never in config.yaml,
since config.yaml is meant to be safely committed to version control.
"""

from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {CONFIG_PATH}. "
            f"This file holds non-secret thresholds/paths and should be committed to the repo."
        )
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}


_CONFIG = _load_config()


def get(*keys, default=None):
    """
    Safe nested lookup, e.g. get("agent", "max_retries").
    Returns `default` if any key in the path is missing rather than
    raising -- keeps future config.yaml additions backward-compatible.
    """
    node = _CONFIG
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def resolve_path(relative_path: str) -> Path:
    """
    Resolves a config.yaml path against the PROJECT ROOT, regardless of
    the current working directory the caller was launched from.
    """
    return PROJECT_ROOT / relative_path