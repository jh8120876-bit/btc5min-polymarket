"""
Institutional Sentiment Engine — Zero-Cost Architecture.

Data sources (all free, no API keys):
  1. Alternative.me Fear & Greed Index (polled every 1 hour)
  2. CoinDesk / CoinTelegraph RSS feeds (polled every 3-5 minutes)

Provides get_market_context() for injection into AI prompts.
All network errors are silently handled — never crashes the engine.
"""

import re
import threading
import time
import defusedxml.ElementTree as ET
from typing import Optional

import requests

from ..config import log

# ── Fear & Greed Index ────────────────────────────────────────

_FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1&format=json"
_FEAR_GREED_INTERVAL = 3600  # 1 hour

_fg_cache: dict = {}
_fg_last_fetch: float = 0
_fg_lock = threading.Lock()


def _fetch_fear_greed() -> dict:
    """Fetch Fear & Greed Index from Alternative.me. Returns {} on failure."""
    global _fg_cache, _fg_last_fetch
    now = time.time()

    with _fg_lock:
        if _fg_cache and (now - _fg_last_fetch) < _FEAR_GREED_INTERVAL:
            return _fg_cache

    try:
        resp = requests.get(_FEAR_GREED_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        entry = data.get("data", [{}])[0]
        result = {
            "value": int(entry.get("value", 0)),
            "label": entry.get("value_classification", "N/A"),
            "timestamp": entry.get("timestamp", ""),
        }
        with _fg_lock:
            _fg_cache = result
            _fg_last_fetch = now
        log.info(f"[SENTIMENT] Fear & Greed: {result['value']} ({result['label']})")
        return result
    except Exception as e:
        log.debug(f"[SENTIMENT] Fear & Greed fetch failed: {e}")
        with _fg_lock:
            return _fg_cache  # return stale cache or {}


# ── RSS News Parser ───────────────────────────────────────────

_RSS_FEEDS = [
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("CoinTelegraph", "https://cointelegraph.com/rss"),
]
_RSS_INTERVAL = 240  # 4 minutes
_RSS_MAX_HEADLINES = 8

_rss_cache: list[dict] = []
_rss_last_fetch: float = 0
_rss_lock = threading.Lock()


def _parse_rss_xml(xml_text: str, source: str) -> list[dict]:
    """Parse RSS XML and extract headlines. Pure stdlib, no dependencies."""
    items = []
    try:
        root = ET.fromstring(xml_text)
        # Standard RSS 2.0: <rss><channel><item><title>...</title></item>
        for item in root.iter("item"):
            title_el = item.find("title")
            if title_el is not None and title_el.text:
                title = title_el.text.strip()
                # Skip very short or empty titles
                if len(title) < 10:
                    continue
                pub_date = ""
                pub_el = item.find("pubDate")
                if pub_el is not None and pub_el.text:
                    pub_date = pub_el.text.strip()
                items.append({
                    "title": title[:200],
                    "source": source,
                    "pub_date": pub_date,
                })
                if len(items) >= _RSS_MAX_HEADLINES:
                    break
    except ET.ParseError:
        pass
    return items


def _fetch_rss() -> list[dict]:
    """Fetch headlines from all RSS feeds. Returns cached data on failure."""
    global _rss_cache, _rss_last_fetch
    now = time.time()

    with _rss_lock:
        if _rss_cache and (now - _rss_last_fetch) < _RSS_INTERVAL:
            return list(_rss_cache)

    all_headlines: list[dict] = []
    for source_name, url in _RSS_FEEDS:
        try:
            resp = requests.get(
                url,
                timeout=12,
                headers={"User-Agent": "btc5min-predictor/1.0"},
            )
            resp.raise_for_status()
            headlines = _parse_rss_xml(resp.text, source_name)
            all_headlines.extend(headlines)
        except Exception as e:
            log.debug(f"[SENTIMENT] RSS {source_name} failed: {e}")

    if all_headlines:
        # Deduplicate by title (first 60 chars) and limit
        seen = set()
        unique = []
        for h in all_headlines:
            key = h["title"][:60].lower()
            if key not in seen:
                seen.add(key)
                unique.append(h)
        unique = unique[:_RSS_MAX_HEADLINES]

        with _rss_lock:
            _rss_cache = unique
            _rss_last_fetch = now
        log.info(f"[SENTIMENT] RSS: {len(unique)} headlines from "
                 f"{len(_RSS_FEEDS)} feeds")
        return unique

    # Return stale cache if all feeds failed
    with _rss_lock:
        return list(_rss_cache)


# ── Black Swan Circuit Breaker ────────────────────────────────

_BLACK_SWAN_KEYWORDS = [
    # Regulatory / legal shocks
    "ban", "prohibit", "illegal", "prohíbe", "prohíben", "ilegal",
    "sec lawsuit", "sec demanda", "indictment", "arrested",
    # Exchange collapse / insolvency
    "bankrupt", "insolvency", "quiebra", "hack", "hacked", "exploit",
    "funds frozen", "withdrawal halt", "retiros suspendidos",
    # Macro shocks
    "emergency rate", "tasa emergencia", "black monday", "flash crash",
    "circuit breaker", "trading halt", "market halt",
    # Systemic risk
    "depeg", "depegged", "bank run", "corrida bancaria",
    "contagion", "contagio", "systemic risk", "riesgo sistémico",
]

_circuit_breaker_active: bool = False
_circuit_breaker_until: float = 0
_circuit_breaker_reason: str = ""
_cb_lock = threading.Lock()


def _check_black_swan(headlines: list[dict]) -> tuple[bool, str]:
    """Scan headlines for Black Swan keywords using word-boundary regex.

    Uses \\b word boundaries to prevent false positives like
    'urban' matching 'ban' or 'banana' matching 'ban'.
    """
    for h in headlines:
        title_lower = h.get("title", "").lower()
        for kw in _BLACK_SWAN_KEYWORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', title_lower):
                return True, f"Black Swan detectado: '{h['title'][:100]}' (keyword: {kw})"
    return False, ""


def activate_circuit_breaker(reason: str, duration_minutes: int = 15):
    """Activate the circuit breaker — pauses all auto-bets for N minutes."""
    global _circuit_breaker_active, _circuit_breaker_until, _circuit_breaker_reason
    with _cb_lock:
        _circuit_breaker_active = True
        _circuit_breaker_until = time.time() + (duration_minutes * 60)
        _circuit_breaker_reason = reason
    log.warning(f"[CIRCUIT BREAKER] ACTIVADO por {duration_minutes}min — {reason}")


def is_circuit_breaker_active() -> bool:
    """Check if the circuit breaker is currently active."""
    global _circuit_breaker_active
    with _cb_lock:
        if _circuit_breaker_active and time.time() >= _circuit_breaker_until:
            _circuit_breaker_active = False
            _circuit_breaker_reason = ""
            log.info("[CIRCUIT BREAKER] Expirado — operación normal restaurada")
        return _circuit_breaker_active


def cancel_circuit_breaker() -> bool:
    """Manually cancel the circuit breaker. Returns True if it was active."""
    global _circuit_breaker_active, _circuit_breaker_reason
    with _cb_lock:
        was_active = _circuit_breaker_active
        _circuit_breaker_active = False
        _circuit_breaker_reason = ""
    if was_active:
        log.info("[CIRCUIT BREAKER] Cancelado manualmente")
    return was_active


def get_circuit_breaker_status() -> dict:
    """Get circuit breaker status for dashboard display."""
    with _cb_lock:
        remaining = max(0, _circuit_breaker_until - time.time()) if _circuit_breaker_active else 0
        return {
            "active": _circuit_breaker_active and remaining > 0,
            "reason": _circuit_breaker_reason,
            "remaining_seconds": round(remaining),
        }


# ── Public API ────────────────────────────────────────────────

def get_fear_greed() -> dict:
    """Get cached Fear & Greed data. Safe to call from any thread."""
    return _fetch_fear_greed()


def get_headlines() -> list[dict]:
    """Get cached RSS headlines. Safe to call from any thread."""
    return _fetch_rss()


def get_market_context() -> str:
    """Build formatted market context string for AI prompt injection.

    Returns a compact, AI-friendly summary of sentiment + news.
    Always returns a string (never raises).
    Triggers circuit breaker on Black Swan headlines.
    """
    parts = []

    # Fear & Greed
    fg = _fetch_fear_greed()
    if fg and fg.get("value"):
        val = fg["value"]
        label = fg["label"]
        # Institutional interpretation
        if val <= 20:
            interp = "PANICO EXTREMO — contrarian bullish, smart money acumula"
        elif val <= 35:
            interp = "MIEDO — sesgo bajista retail, posible acumulacion institucional"
        elif val <= 55:
            interp = "NEUTRAL — sin sesgo dominante"
        elif val <= 75:
            interp = "AVARICIA — sesgo alcista, cautela ante posible techo"
        else:
            interp = "AVARICIA EXTREMA — contrarian bearish, distribucion probable"
        parts.append(
            f"Fear_Greed_Index: {val}/100 ({label}) — {interp}"
        )
    else:
        parts.append("Fear_Greed_Index: Sin datos")

    # RSS Headlines
    headlines = _fetch_rss()
    if headlines:
        titles = [f"- {h['title']}" for h in headlines[:5]]
        parts.append("Titulares_recientes:\n" + "\n".join(titles))

        # Black Swan circuit breaker check
        triggered, reason = _check_black_swan(headlines)
        if triggered and not is_circuit_breaker_active():
            activate_circuit_breaker(reason, duration_minutes=15)
            parts.append(f"\n⚠️ CIRCUIT BREAKER ACTIVADO: {reason}")
    else:
        parts.append("Titulares_recientes: Sin datos de noticias")

    return "\n".join(parts)
