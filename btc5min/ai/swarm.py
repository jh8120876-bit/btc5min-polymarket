"""
Multi-Agent Swarm — parallel AI predictions via different API providers.

Each agent runs in its own thread, calls its API, and returns an AISignal.
The engine collects results and places independent shadow bets per agent.
Uses only `requests` — no provider-specific SDKs required.
"""

import json
import os
import threading
import time
from typing import Optional

import requests

from ..utils import extract_json

# All HTTP timeouts MUST be strictly less than SWARM_JOIN_TIMEOUT
# to guarantee the network cuts before the engine abandons the thread.
# Tuned for max 3 agents (1 primary + 2 swarm): 30s is enough for any
# single API call, and all run in parallel so wall-clock = slowest one.
SWARM_JOIN_TIMEOUT = 30
_HTTP_TIMEOUT = 25  # seconds — hard ceiling for all API calls

from ..config import log
from ..config_manager import rules
from ..models import AISignal

# Will be set by engine.py at startup to share the global rate-limit semaphore
_api_semaphore: threading.Semaphore | None = None


# ── Agent API call implementations ────────────────────────────

def _call_deepseek(api_key: str, model: str, system_prompt: str,
                   user_prompt: str, base_url: str | None = None) -> dict:
    url = (base_url or "https://api.deepseek.com/v1") + "/chat/completions"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model or "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 600,
            "temperature": 0.15,
            "response_format": {"type": "json_object"},
        },
        timeout=_HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    body = resp.json()
    content = body["choices"][0]["message"]["content"].strip()
    result = extract_json(content)
    if result is None:
        raise ValueError(f"Could not parse JSON from DeepSeek response: {content[:200]}")
    usage = body.get("usage") or {}
    result["_usage_tokens"] = usage.get("total_tokens") or (
        (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
    ) or None
    return result


def _call_anthropic(api_key: str, model: str, system_prompt: str,
                    user_prompt: str, base_url: str | None = None) -> dict:
    url = (base_url or "https://api.anthropic.com") + "/v1/messages"
    resp = requests.post(
        url,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model or "claude-opus-4-6",
            "max_tokens": 600,
            "temperature": 0.15,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=_HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    body = resp.json()
    content = body["content"][0]["text"].strip()
    result = extract_json(content)
    if result is None:
        raise ValueError(f"Could not parse JSON from Anthropic response: {content[:200]}")
    usage = body.get("usage") or {}
    result["_usage_tokens"] = (
        (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
    ) or None
    return result


def _call_openai(api_key: str, model: str, system_prompt: str,
                 user_prompt: str, base_url: str | None = None) -> dict:
    url = (base_url or "https://api.openai.com/v1") + "/chat/completions"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model or "gpt-4o",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 600,
            "temperature": 0.15,
            "response_format": {"type": "json_object"},
        },
        timeout=_HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    body = resp.json()
    content = body["choices"][0]["message"]["content"].strip()
    result = extract_json(content)
    if result is None:
        raise ValueError(f"Could not parse JSON from OpenAI response: {content[:200]}")
    usage = body.get("usage") or {}
    result["_usage_tokens"] = usage.get("total_tokens") or (
        (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
    ) or None
    return result


def _call_local(api_key: str, model: str, system_prompt: str,
                 user_prompt: str, base_url: str | None = None) -> dict:
    """OpenAI-compatible caller for local LLMs (LM Studio / Ollama)."""
    url = (base_url or "http://localhost:1234/v1") + "/chat/completions"
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 600,
        "temperature": 0.15,
    }
    resp = requests.post(url, json=body, timeout=_HTTP_TIMEOUT)
    resp.raise_for_status()
    body_resp = resp.json()
    content = body_resp["choices"][0]["message"]["content"].strip()
    result = extract_json(content)
    if result is None:
        raise ValueError(f"Could not parse JSON from local LLM response: {content[:200]}")
    usage = body_resp.get("usage") or {}
    result["_usage_tokens"] = usage.get("total_tokens") or (
        (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
    ) or None
    return result


_API_CALLERS = {
    "deepseek": _call_deepseek,
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "local": _call_local,
}


# ── HFT 2-Phase: Fast callers (max_tokens=20, prediction-only) ──

def _call_fast_openai_compat(api_key: str, model: str, system_prompt: str,
                             user_prompt: str, base_url: str,
                             extra_headers: dict | None = None) -> dict:
    """Shared fast caller for OpenAI-compatible APIs (Phase 1: ~0.3-0.5s)."""
    url = base_url + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    fast_system = (
        "Respond ONLY with a JSON object: "
        '{"prediction":"UP" or "DOWN","confidence":integer 40-95}. '
        "No explanation, no extra keys."
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": fast_system},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 20,
        "temperature": 0.05,
    }
    # DeepSeek/OpenAI support response_format
    if "anthropic" not in base_url:
        body["response_format"] = {"type": "json_object"}
    resp = requests.post(url, headers=headers, json=body, timeout=8)
    resp.raise_for_status()
    resp_body = resp.json()
    content = resp_body["choices"][0]["message"]["content"].strip()
    result = extract_json(content)
    if result is None:
        raise ValueError(f"Fast call: could not parse JSON: {content[:100]}")
    usage = resp_body.get("usage") or {}
    result["_usage_tokens"] = usage.get("total_tokens") or (
        (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
    ) or None
    return result


def _call_fast_anthropic(api_key: str, model: str, system_prompt: str,
                         user_prompt: str, base_url: str | None = None) -> dict:
    """Fast Anthropic caller (Phase 1)."""
    url = (base_url or "https://api.anthropic.com") + "/v1/messages"
    fast_system = (
        "Respond ONLY with a JSON object: "
        '{"prediction":"UP" or "DOWN","confidence":integer 40-95}. '
        "No explanation, no extra keys."
    )
    resp = requests.post(
        url,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model or "claude-opus-4-6",
            "max_tokens": 20,
            "temperature": 0.05,
            "system": fast_system,
            "messages": [{"role": "user", "content": user_prompt}],
        },
        timeout=8,
    )
    resp.raise_for_status()
    resp_body = resp.json()
    content = resp_body["content"][0]["text"].strip()
    result = extract_json(content)
    if result is None:
        raise ValueError(f"Fast Anthropic: could not parse JSON: {content[:100]}")
    usage = resp_body.get("usage") or {}
    result["_usage_tokens"] = (
        (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
    ) or None
    return result


def call_fast_prediction(api_type: str, api_key: str, model: str,
                         system_prompt: str, user_prompt: str,
                         base_url: str | None = None) -> dict | None:
    """Phase 1 HFT: Get prediction+confidence in <0.5s. Returns dict or None."""
    try:
        if api_type == "anthropic":
            return _call_fast_anthropic(api_key, model, system_prompt,
                                        user_prompt, base_url)
        # DeepSeek, OpenAI, Local — all OpenAI-compatible
        default_urls = {
            "deepseek": "https://api.deepseek.com/v1",
            "openai": "https://api.openai.com/v1",
            "local": "http://localhost:1234/v1",
        }
        url = base_url or default_urls.get(api_type, "https://api.deepseek.com/v1")
        return _call_fast_openai_compat(api_key, model, system_prompt,
                                         user_prompt, url)
    except Exception as e:
        log.warning(f"[HFT-Phase1] Fast prediction failed ({api_type}): {e}")
        return None


def call_background_reasoning(api_type: str, api_key: str, model: str,
                              prediction: str, confidence: int,
                              user_prompt: str,
                              base_url: str | None = None) -> dict | None:
    """Phase 2 HFT: Background reasoning call. Returns full dict or None."""
    injected_prompt = (
        f"You already decided {prediction} at {confidence}% confidence.\n"
        f"Now generate your detailed 'reasoning' and 'news_impact' analysis.\n"
        f"Respond with JSON: {{\"reasoning\": \"...\", \"news_impact\": \"...\"}}\n\n"
        f"Original context:\n{user_prompt[:3000]}"
    )
    try:
        caller = _API_CALLERS.get(api_type)
        if not caller:
            return None
        return caller(api_key, model, "You are a crypto analyst.", injected_prompt,
                      base_url)
    except Exception as e:
        log.debug(f"[HFT-Phase2] Background reasoning failed ({api_type}): {e}")
        return None


# ── Public interface ──────────────────────────────────────────

def get_active_agents() -> list[dict]:
    """Return list of enabled agent configs from dynamic_rules.json."""
    agents_section = rules.get_section("agents")
    if not agents_section:
        return []
    active = []
    for agent_id, cfg in agents_section.items():
        if not isinstance(cfg, dict):
            continue
        if cfg.get("enabled", False):
            agent = dict(cfg)
            agent["agent_id"] = agent_id
            active.append(agent)
    return active


def get_primary_agent() -> dict | None:
    """Return the agent marked as is_primary=true, or None."""
    agents_section = rules.get_section("agents")
    if not agents_section:
        return None
    for agent_id, cfg in agents_section.items():
        if not isinstance(cfg, dict):
            continue
        if cfg.get("is_primary", False):
            agent = dict(cfg)
            agent["agent_id"] = agent_id
            return agent
    return None


def get_primary_agent_id() -> str:
    """Return the agent_id of the primary agent."""
    primary = get_primary_agent()
    return primary["agent_id"] if primary else "deepseek_smc"


def get_secondary_agents() -> list[dict]:
    """Return active agents excluding the primary one."""
    return [a for a in get_active_agents() if not a.get("is_primary", False)]


def get_all_agents() -> list[dict]:
    """Return all agent configs (enabled or not) for the config panel."""
    agents_section = rules.get_section("agents")
    if not agents_section:
        return []
    result = []
    for agent_id, cfg in agents_section.items():
        if not isinstance(cfg, dict):
            continue
        agent = dict(cfg)
        agent["agent_id"] = agent_id
        # Check if API key is actually set in environment
        key_env = cfg.get("api_key_env", "")
        agent["has_api_key"] = bool(os.environ.get(key_env, "")) if key_env else False
        result.append(agent)
    return result


def call_agent(agent_cfg: dict, user_prompt: str,
               ta: dict, fatigue_prompt: str = "",
               shared_system_prompt: str = "") -> Optional[AISignal]:
    """Call a single agent's API and return an AISignal or None on failure.

    This is designed to run in a thread — it's blocking and self-contained.
    If shared_system_prompt is set, it overrides the agent's own system_prompt_key.
    If fatigue_prompt is set, it's prepended to the system prompt to override strategy.
    """
    agent_id = agent_cfg.get("agent_id", "unknown")
    api_type = agent_cfg.get("api_type", "deepseek")
    model = agent_cfg.get("model", "")
    key_env = agent_cfg.get("api_key_env", "")
    api_key = os.environ.get(key_env, "") if key_env else ""
    base_url = agent_cfg.get("api_base_url") or None

    # Shared system prompt (from primary's RAG + lessons + regime) takes priority
    if shared_system_prompt:
        system_prompt = shared_system_prompt
    else:
        prompt_key = agent_cfg.get("system_prompt_key", "")
        system_prompt = rules.get_prompt(prompt_key) if prompt_key else ""

    if not api_key and api_type != "local":
        log.warning(f"[SWARM] Agent {agent_id}: no API key ({key_env}), skipping")
        return None

    if not system_prompt:
        log.warning(f"[SWARM] Agent {agent_id}: no system prompt available, skipping")
        return None

    # Fatigue override: prepend fatigue instructions to override base strategy
    if fatigue_prompt:
        system_prompt = fatigue_prompt + "\n\n---\n(Prompt base ignorado por fatiga)\n" + system_prompt
        log.info(f"[SWARM] Agent {agent_id}: FATIGUE prompt injected")

    caller = _API_CALLERS.get(api_type)
    if not caller:
        log.warning(f"[SWARM] Agent {agent_id}: unknown api_type '{api_type}'")
        return None

    # Acquire rate-limit semaphore for all remote API calls.
    # Prevents overlapping calls from swarm + primary + observer.
    acquired = False
    if _api_semaphore and api_type != "local":
        # Non-blocking try: if both slots are taken, skip this agent
        # rather than queuing (prevents swarm from stalling the window)
        acquired = _api_semaphore.acquire(timeout=5)
        if not acquired:
            log.warning(f"[SWARM] Agent {agent_id}: API semaphore busy, skipping")
            return None

    # Paid agents (anthropic, openai) get 1 retry on timeout/5xx
    max_attempts = 2 if api_type in ("anthropic", "openai") else 1
    last_error = None

    try:
        for attempt in range(max_attempts):
            try:
                tag = f" (retry)" if attempt > 0 else ""
                log.info(f"[SWARM] Agent {agent_id}: calling {api_type}/{model}...{tag}")
                _t0 = time.time()
                r = caller(api_key, model, system_prompt, user_prompt, base_url)
                _latency_ms = round((time.time() - _t0) * 1000, 1)

                confidence = max(0, min(100, int(r.get("confidence", 50))))
                raw_pred = r.get("prediction", "UP").upper().strip()
                if raw_pred not in ("UP", "DOWN"):
                    mom = ta.get("momentum", 0) if ta else 0
                    raw_pred = "UP" if mom >= 0 else "DOWN"
                    confidence = max(confidence, 51)
                if confidence == 50:
                    confidence = 51

                signal = AISignal(
                    prediction=raw_pred,
                    confidence=confidence,
                    reasoning=r.get("reasoning", "Sin analisis"),
                    news_summary=r.get("news_impact", "N/A"),
                    risk_score=r.get("risk_score", "MEDIUM").upper(),
                    suggested_bet_pct=max(0.01, min(0.05,
                        float(r.get("suggested_bet_pct", 0.02)))),
                    timestamp=time.time(),
                    original_confidence=confidence,
                    layer_alignment=r.get("layer_alignment", ""),
                )
                signal.latency_ms = _latency_ms
                signal.usage_tokens = r.get("_usage_tokens")
                log.info(f"[SWARM] Agent {agent_id}: {signal.prediction} "
                         f"({signal.confidence}%) [{_latency_ms:.0f}ms] — "
                         f"{signal.reasoning[:80]}")
                return signal

            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError) as e:
                last_error = e
                if attempt < max_attempts - 1:
                    log.warning(f"[SWARM] Agent {agent_id}: {type(e).__name__}, "
                                f"retrying in 2s...")
                    time.sleep(2)
                    continue
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else 0
                last_error = e
                if status >= 500 and attempt < max_attempts - 1:
                    log.warning(f"[SWARM] Agent {agent_id}: HTTP {status}, "
                                f"retrying in 2s...")
                    time.sleep(2)
                    continue
                log.warning(f"[SWARM] Agent {agent_id}: HTTP error {status}")
                return None
            except json.JSONDecodeError as e:
                log.warning(f"[SWARM] Agent {agent_id}: invalid JSON response — {e}")
                return None
            except Exception as e:
                log.warning(f"[SWARM] Agent {agent_id}: unexpected error — {e}")
                return None

        # All retries exhausted
        log.warning(f"[SWARM] Agent {agent_id}: failed after {max_attempts} attempts "
                    f"— {last_error}")
        return None
    finally:
        if acquired:
            _api_semaphore.release()


def run_swarm_predictions(user_prompt: str, ta: dict,
                          timeout: float = SWARM_JOIN_TIMEOUT,
                          fatigue_prompts: dict[str, str] | None = None,
                          shared_system_prompt: str = "",
                          ) -> dict[str, AISignal]:
    """Run all secondary agents in parallel threads.

    Returns {agent_id: AISignal} for agents that responded successfully.
    Primary agent is handled by the existing engine code path.
    shared_system_prompt: if set, all agents inherit this prompt instead of their own.
    fatigue_prompts: {agent_id: override_prompt} for fatigued agents.
    """
    agents = get_secondary_agents()
    if not agents:
        return {}

    fatigue_prompts = fatigue_prompts or {}
    results: dict[str, Optional[AISignal]] = {}
    lock = threading.Lock()

    def _worker(agent_cfg):
        agent_id = agent_cfg["agent_id"]
        fp = fatigue_prompts.get(agent_id, "")
        signal = call_agent(agent_cfg, user_prompt, ta, fatigue_prompt=fp,
                            shared_system_prompt=shared_system_prompt)
        with lock:
            results[agent_id] = signal

    threads = []
    for agent_cfg in agents:
        t = threading.Thread(
            target=_worker,
            args=(agent_cfg,),
            daemon=True,
            name=f"swarm-{agent_cfg['agent_id']}",
        )
        threads.append(t)
        t.start()

    # Wait for all threads (with timeout)
    for t in threads:
        t.join(timeout=timeout)

    # Filter out None results
    return {k: v for k, v in results.items() if v is not None}
