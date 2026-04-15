"""Shared utility functions — deduplicates logic used across modules."""

import json
import re


def trust_bet_multiplier(trust_score: float) -> float:
    """Trust score affects bet sizing: 0.0x to 1.5x.

    Used by both AIEngine (ai_engine.py) and RiskManager (risk.py).
    """
    if trust_score < 20:
        return 0.0
    if trust_score < 35:
        return 0.3
    if trust_score < 50:
        return 0.6
    if trust_score < 65:
        return 1.0
    if trust_score < 80:
        return 1.2
    return 1.5


def extract_json(text: str) -> dict | None:
    """Extract JSON from LLM response, handling markdown blocks and preamble.

    Bullet-proof extractor for models that wrap JSON in prose.
    Tries in order:
    1. Direct json.loads (clean response)
    2. Markdown code block extraction (```json ... ```)
    3. Greedy brace match with nested brace support (re.DOTALL)
    4. Flat brace match (no nesting)
    5. Keyword scraping: extract prediction/confidence from raw text
    """
    # 1) Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) Markdown code block
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3) Greedy: outermost { ... } allowing nested braces (re.DOTALL)
    greedy_match = re.search(r"\{.*\}", text, re.DOTALL)
    if greedy_match:
        try:
            return json.loads(greedy_match.group())
        except json.JSONDecodeError:
            pass

    # 4) Flat: first { ... } without nested braces
    flat_match = re.search(r"\{[^{}]*\}", text)
    if flat_match:
        try:
            return json.loads(flat_match.group())
        except json.JSONDecodeError:
            pass

    # 5) Keyword scraping: last resort for completely broken format
    pred_match = re.search(r"\b(UP|DOWN)\b", text, re.IGNORECASE)
    conf_match = re.search(r"(\d{2,3})\s*%", text)
    if pred_match:
        return {
            "prediction": pred_match.group(1).upper(),
            "confidence": int(conf_match.group(1)) if conf_match else 60,
            "reasoning": text[:200].strip(),
        }

    return None
