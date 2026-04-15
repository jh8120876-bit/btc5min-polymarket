"""
ConfigManager — Hot-reload fail-safe for dynamic_rules.json.

Reads the rules file with mtime-based caching. If the file is malformed,
logs the error and silently retains the last known-good configuration.
The engine NEVER crashes due to a JSON typo.
"""

import json
import os
import threading
from pathlib import Path
from copy import deepcopy

from .config import log

_RULES_PATH = str(Path(__file__).resolve().parent.parent / "dynamic_rules.json")

# ── Default values and expected types for critical fields ──────────
# Structure: {section: {key: (expected_type, default_value)}}
# Used as fallback when dynamic_rules.json has invalid types.
_FIELD_SCHEMA: dict[str, dict[str, tuple[type, object]]] = {
    "risk": {
        "max_bet_pct":              ((int, float), 0.05),
        "min_confidence":           ((int, float), 45),
        "high_risk_min_conf":       ((int, float), 55),
        "daily_loss_limit_pct":     ((int, float), 0.10),
        "max_drawdown_halt_pct":    ((int, float), 10),
        "max_drawdown_reduce_pct":  ((int, float), 5),
        "paper_bet_amount":         ((int, float), 1.0),
        "low_quality_cap_pct":      ((int, float), 0.001),
        "kelly_fraction":           ((int, float), 0.5),
        "losing_streak_threshold":  ((int, float), 3),
        "losing_streak_reduction":  ((int, float), 0.5),
    },
    "trust": {
        "default":          ((int, float), 50),
        "win_bonus":        ((int, float), 4),
        "loss_penalty":     ((int, float), 5),
        "high_conf_bonus":  ((int, float), 2),
        "min":              ((int, float), 10),
        "max":              ((int, float), 95),
    },
    "sniper": {
        "eval_interval_sec":       ((int, float), 15),
        "eval_cutoff_sec":         ((int, float), 235),
        "first_eval_delay_sec":    ((int, float), 25),
        "min_consensus_age_sec":   ((int, float), 20),
        "consensus_score_fire":    ((int, float), 2),
        "consensus_score_flip":    ((int, float), -2),
        "confidence_decay_per_min":((int, float), 2.0),
        "decay_floor_conf":        ((int, float), 51),
        "ev_early_fire_threshold": ((int, float), 1.30),
        "last_minute_rescue_conf": ((int, float), 60),
        "last_minute_rescue_ev":   ((int, float), 1.05),
    },
    "second_opinion": {
        "enabled":             (bool, True),
        "cache_ttl_sec":       ((int, float), 60),
        "variant_d_launch_sec":((int, float), 120),
        "variant_d_enabled":   (bool, True),
        "variant_b_enabled":   (bool, True),
        "variant_c_enabled":   (bool, True),
        "variant_a_enabled":   (bool, True),
        "timeout_sec":         ((int, float), 3.0),
        "max_tokens":          ((int, float), 80),
    },
}


class ConfigManager:
    """Singleton manager for dynamic_rules.json with mtime cache."""

    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._lock = threading.Lock()
        self._last_mtime: float = 0.0
        self._rules: dict = {}
        self._path = _RULES_PATH
        self._initialized = True
        # Initial load
        self._force_load()

    def _force_load(self):
        """Read JSON from disk unconditionally. On error, retain previous config."""
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = f.read()
            parsed = json.loads(raw)
            self._validate_and_fix(parsed)
            with self._lock:
                self._rules = parsed
                self._last_mtime = os.path.getmtime(self._path)
            log.info(f"ConfigManager: dynamic_rules.json loaded OK "
                     f"(v{parsed.get('_meta', {}).get('version', '?')})")
        except FileNotFoundError:
            log.warning(f"ConfigManager: {self._path} not found — using defaults")
        except json.JSONDecodeError as e:
            log.error(f"ConfigManager: JSON SYNTAX ERROR in dynamic_rules.json "
                      f"line {e.lineno} col {e.colno}: {e.msg}. "
                      f"RETAINING last valid config.")
        except Exception as e:
            log.error(f"ConfigManager: unexpected error reading rules: {e}. "
                      f"RETAINING last valid config.")

    @staticmethod
    def _validate_and_fix(parsed: dict):
        """Validate critical field types in-place. Replace bad values with defaults."""
        for section, fields in _FIELD_SCHEMA.items():
            section_data = parsed.get(section)
            if not isinstance(section_data, dict):
                continue
            for key, (expected_types, default) in fields.items():
                if key not in section_data:
                    continue
                val = section_data[key]
                if val is None or not isinstance(val, expected_types):
                    log.warning(
                        f"ConfigManager: [{section}].{key} has invalid value "
                        f"{val!r} (expected {expected_types}) — "
                        f"using default {default}"
                    )
                    section_data[key] = default

    def _maybe_reload(self):
        """Check mtime and reload only if the file changed on disk."""
        try:
            current_mtime = os.path.getmtime(self._path)
        except (FileNotFoundError, OSError):
            return
        if current_mtime != self._last_mtime:
            log.info("ConfigManager: file change detected, reloading...")
            self._force_load()

    def force_reload(self) -> bool:
        """
        Force an immediate reload (called from /api/reload_rules).
        Returns True if load succeeded, False if JSON was malformed.
        """
        old_rules = self.get_all()
        self._force_load()
        new_rules = self.get_all()
        return new_rules != old_rules or bool(new_rules)

    def get_all(self) -> dict:
        """Return a deep copy of the full rules dict."""
        self._maybe_reload()
        with self._lock:
            return deepcopy(self._rules)

    def get(self, section: str, key: str, default=None):
        """Get a specific value: get('risk', 'max_bet_pct', 0.05)."""
        self._maybe_reload()
        with self._lock:
            return self._rules.get(section, {}).get(key, default)

    def get_section(self, section: str) -> dict:
        """Get an entire section as a dict copy."""
        self._maybe_reload()
        with self._lock:
            return deepcopy(self._rules.get(section, {}))

    def get_prompt(self, key: str) -> str:
        """Get a prompt by key. Priority: dynamic_rules.json > strategies/ files."""
        prompt = self.get("prompts", key, "")
        if prompt:
            return prompt
        # Fallback: load from strategies/prompts/system/{key}.txt
        from .strategies import load_system_prompt
        return load_system_prompt(key) or ""


# Module-level singleton for easy import
rules = ConfigManager()
