"""Strategy Loader — reads system prompts and RAG strategies from files.

System prompts: strategies/prompts/system/{key}.txt
Fatigue prompts: strategies/prompts/fatigue/{NN_name}.txt (sorted by filename)
RAG strategies: strategies/rag/{slug}.json
"""

import json
from pathlib import Path

from ..config import log

_BASE_DIR = Path(__file__).resolve().parent
_PROMPTS_SYSTEM = _BASE_DIR / "prompts" / "system"
_PROMPTS_FATIGUE = _BASE_DIR / "prompts" / "fatigue"
_RAG_DIR = _BASE_DIR / "rag"

_RAG_REQUIRED_KEYS = {"strategy_name", "concept_family", "entry_rules"}


def load_system_prompt(key: str) -> str | None:
    """Load a system prompt by key (filename without .txt)."""
    safe_key = Path(key).name
    path = _PROMPTS_SYSTEM / f"{safe_key}.txt"
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as e:
        log.warning(f"[STRATEGIES] Failed to read prompt {path.name}: {e}")
        return None


def load_rag_strategies() -> list[dict]:
    """Load all RAG strategies from rag/*.json.

    Each JSON file must contain at minimum: strategy_name, concept_family, entry_rules.
    Invalid files are logged and skipped.
    """
    strategies = []
    if not _RAG_DIR.exists():
        return strategies
    for f in sorted(_RAG_DIR.glob("*.json")):
        strat = _load_single_rag(f)
        if strat:
            strategies.append(strat)
    return strategies


def _load_single_rag(path: Path) -> dict | None:
    """Load and validate a single RAG strategy JSON file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log.warning(f"[STRATEGIES] Invalid RAG file {path.name}: {e}")
        return None
    if not isinstance(data, dict):
        log.warning(f"[STRATEGIES] RAG file {path.name}: expected dict, got {type(data).__name__}")
        return None
    missing = _RAG_REQUIRED_KEYS - data.keys()
    if missing:
        log.warning(f"[STRATEGIES] RAG file {path.name}: missing keys {missing}")
        return None
    return data
