"""
Local LLM utilities — stateless helpers for detecting LM Studio / Ollama
servers and listing their available models.

The system is now fully provider-agnostic: local models are configured as
either the Primary Agent or a Swarm Agent via `dynamic_rules.json` with
`api_type: "local"`. There is no longer a parallel "IA LOCAL" prediction
path in the engine — the prediction is dispatched through
`swarm._API_CALLERS["local"]` like any other provider.

This module exists solely to power the UI dropdowns that let the user pick
a model when configuring a local agent.
"""

import time
from typing import Optional

import requests

from ..config import LOCAL_LLM_URL, log


class LocalLLM:
    """Stateless helper for probing a local OpenAI-compatible LLM server."""

    def __init__(self, url: Optional[str] = None):
        self.url: str = url or LOCAL_LLM_URL
        self._available_cache: bool = False
        self._available_cache_time: float = 0
        self._available_cache_ttl: float = 30.0

    def is_available(self, url: Optional[str] = None) -> bool:
        """Check whether a local LLM server is reachable.

        Caches the positive/negative result for 30s to avoid hammering the
        endpoint from the UI or the live observer.
        """
        target = url or self.url
        now = time.time()
        if now - self._available_cache_time < self._available_cache_ttl:
            return self._available_cache
        try:
            r = requests.get(f"{target}/models", timeout=3)
            self._available_cache = r.status_code == 200
        except Exception:
            self._available_cache = False
        self._available_cache_time = now
        return self._available_cache

    def list_models(self, url: Optional[str] = None) -> list[dict]:
        """List models exposed by a local OpenAI-compatible server."""
        target = url or self.url
        try:
            r = requests.get(f"{target}/models", timeout=5)
            r.raise_for_status()
            models = r.json().get("data", [])
            return [{"id": m["id"], "object": m.get("object", "model")}
                    for m in models]
        except Exception as e:
            log.error(f"Error listing local LLM models: {e}")
            return []
