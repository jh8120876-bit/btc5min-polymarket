import json
import os
import time
import threading
from datetime import datetime
from typing import Optional

import requests

from ..config import log
from ..models import AISignal, PriceData
from .. import database as db
from ..config_manager import rules
from ..utils import trust_bet_multiplier


class AIEngine:
    """Provider-agnostic prediction engine with trust score and persistent memory."""

    # Trust defaults (overridden by dynamic_rules.json at runtime)
    TRUST_DEFAULT = 50
    TRUST_WIN_BONUS = 4
    TRUST_LOSS_PENALTY = 5
    TRUST_HIGH_CONF_BONUS = 2
    TRUST_MIN = 10
    TRUST_MAX = 95

    @property
    def _trust_cfg(self) -> dict:
        return rules.get_section("trust")

    def _tv(self, key: str, fallback):
        """Get trust value from dynamic rules, falling back to class constant."""
        return self._trust_cfg.get(key, fallback)

    def __init__(self):
        self.last_signal: Optional[AISignal] = None
        self.prediction_history: list[dict] = []
        self.db_accuracy: dict = {}
        self.db_patterns: str = ""
        # Trust score — persists between sessions, guarded by _trust_lock
        self._trust_lock = threading.Lock()
        self.trust_score: float = self.TRUST_DEFAULT
        self.sessions_count: int = 0
        self.learned_lessons: list[str] = []
        self.bias_corrections: dict = {}
        # Early bet threshold — learnable from DB
        self.early_bet_threshold: int = 65  # default: bet early if conf >= 65
        # Confidence calibration map: {reported_bucket: actual_accuracy}
        self.confidence_calibration: dict = {}
        # Volatility regime for bias correction
        self.volatility_regime: str = "NORMAL"  # LOW / NORMAL / HIGH
        # Trend alignment cooldown: max 1 trust boost per hour
        self._last_trend_boost_time: float = 0
        _TREND_BOOST_COOLDOWN = 3600  # seconds
        # ── Epistemic Reflection Memory ──
        self.recent_reflection_lesson: str = ""
        self._reflection_lock = threading.Lock()
        self._reflection_in_flight: bool = False
        # ── Rolling TA history for velocity/acceleration features ──
        from collections import deque
        self._ta_history: dict[str, deque] = {}  # agent_id -> deque(maxlen=3)

    # ── Persistent Memory ──────────────────────────────────

    def load_history_from_db(self):
        """Load historical accuracy, patterns, trust score and memory on startup."""
        try:
            # Load trust score
            with self._trust_lock:
                self.trust_score = db.load_ai_memory("trust_score", self.TRUST_DEFAULT)
            self.sessions_count = db.load_ai_memory("sessions_count", 0)
            self.learned_lessons = db.load_ai_memory("learned_lessons", [])
            self.bias_corrections = db.load_ai_memory("bias_corrections", {})
            self.early_bet_threshold = db.load_ai_memory("early_bet_threshold", 75)

            # Increment session count
            self.sessions_count += 1
            db.save_ai_memory("sessions_count", self.sessions_count)

            log.info(f"AI Memory: trust={self.trust_score:.1f}, "
                     f"sessions={self.sessions_count}, "
                     f"lessons={len(self.learned_lessons)}")

            # Load accuracy and patterns
            self.db_accuracy = db.get_historical_accuracy(limit=50)
            if self.db_accuracy.get("total", 0) > 0:
                self.db_patterns = self.db_accuracy.get("patterns", "")
                log.info(f"AI loaded {self.db_accuracy['total']} historical predictions "
                         f"({self.db_accuracy.get('pct', 0)}% accuracy)")

            # Auto-detect bias from history
            self._detect_bias()

            # Load confidence calibration
            self._load_confidence_calibration()

            # Restore recent prediction outcomes
            recent = db.get_recent_windows(limit=20)
            for w in reversed(recent):
                if w.get("prediction") and w.get("outcome"):
                    self.prediction_history.append({
                        "prediction": w["prediction"],
                        "actual": w["outcome"],
                        "correct": bool(w.get("correct")),
                        "ts": 0,
                    })
            if self.prediction_history:
                self.prediction_history = self.prediction_history[-20:]
                log.info(f"AI restored {len(self.prediction_history)} recent outcomes from DB")

            # Auto-learn from DB on startup (no LLM needed)
            self._auto_learn_from_db()
        except Exception as e:
            log.error(f"Error loading AI history from DB: {e}")

    def _detect_bias(self):
        """Auto-detect prediction biases from historical data."""
        if self.db_accuracy.get("total", 0) < 10:
            return
        up_pct = self.db_accuracy.get("up_pct", 50)
        dn_pct = self.db_accuracy.get("dn_pct", 50)
        up_total = self.db_accuracy.get("up_total", 0)
        dn_total = self.db_accuracy.get("dn_total", 0)

        self.bias_corrections = {}
        # If significantly more UP predictions than DOWN (or vice versa)
        total = up_total + dn_total
        if total > 0:
            up_ratio = up_total / total
            if up_ratio > 0.7:
                self.bias_corrections["over_up"] = (
                    f"Sesgo detectado: {up_ratio:.0%} de predicciones son UP. "
                    f"Considerar DOWN mas seguido."
                )
            elif up_ratio < 0.3:
                self.bias_corrections["over_down"] = (
                    f"Sesgo detectado: {1-up_ratio:.0%} de predicciones son DOWN. "
                    f"Considerar UP mas seguido."
                )
        # Accuracy imbalance
        if up_pct < 40 and up_total >= 5:
            self.bias_corrections["up_weak"] = (
                f"Predicciones UP solo {up_pct}% correctas. "
                f"Reducir confianza en senales UP."
            )
        if dn_pct < 40 and dn_total >= 5:
            self.bias_corrections["down_weak"] = (
                f"Predicciones DOWN solo {dn_pct}% correctas. "
                f"Reducir confianza en senales DOWN."
            )

        if self.bias_corrections:
            db.save_ai_memory("bias_corrections", self.bias_corrections)

    # ── Trust Score ────────────────────────────────────────

    def update_trust(self, prediction: str, actual: str, confidence: int):
        """Update trust score based on prediction outcome."""
        correct = prediction == actual
        t_min = self._tv("min", self.TRUST_MIN)
        t_max = self._tv("max", self.TRUST_MAX)
        with self._trust_lock:
            if correct:
                bonus = self._tv("win_bonus", self.TRUST_WIN_BONUS)
                if confidence >= 70:
                    bonus += self._tv("high_conf_bonus", self.TRUST_HIGH_CONF_BONUS)
                self.trust_score = min(t_max, self.trust_score + bonus)
            else:
                penalty = self._tv("loss_penalty", self.TRUST_LOSS_PENALTY)
                if confidence >= 70:
                    penalty += 2
                self.trust_score = max(t_min, self.trust_score - penalty)
            score_snapshot = self.trust_score

        db.save_ai_memory("trust_score", round(score_snapshot, 1))
        log.info(f"Trust score: {score_snapshot:.1f} "
                 f"({'correct' if correct else 'wrong'}, conf={confidence}%)")

    def get_trust_info(self) -> dict:
        """Get trust score info for display and local LLM."""
        with self._trust_lock:
            score = self.trust_score
        return {
            "trust_score": round(score, 1),
            "sessions": self.sessions_count,
            "lessons_count": len(self.learned_lessons),
            "bias": self.bias_corrections,
            "bet_allowed": score >= 20,
            "bet_multiplier": trust_bet_multiplier(score),
            "early_bet_threshold": self.early_bet_threshold,
        }

    # ── Lessons ────────────────────────────────────────────

    def add_lesson(self, lesson: str):
        """Add a learned lesson (from local LLM post-analysis)."""
        if not lesson:
            return
        self.learned_lessons.append(lesson)
        if len(self.learned_lessons) > 15:
            self.learned_lessons = self.learned_lessons[-15:]
        db.save_ai_memory("learned_lessons", self.learned_lessons)

    # ── Self-learning from DB (no LLM needed) ───────────────

    def _auto_learn_from_db(self):
        """Analyze DB outcomes and auto-generate lessons. No LLM required."""
        try:
            windows = db.get_recent_windows(limit=40)
            if len(windows) < 8:
                return

            lessons = []

            # 1. Momentum alignment: when momentum disagrees with prediction
            mom_agree = [w for w in windows if w.get("prediction") and w.get("outcome")
                         and w.get("momentum") is not None]
            if len(mom_agree) >= 5:
                # Predictions that went AGAINST momentum
                against_mom = [w for w in mom_agree
                               if (w["prediction"] == "UP" and w["momentum"] < -0.01)
                               or (w["prediction"] == "DOWN" and w["momentum"] > 0.01)]
                if len(against_mom) >= 3:
                    wrong = sum(1 for w in against_mom if not w["correct"])
                    pct_wrong = wrong / len(against_mom) * 100
                    if pct_wrong > 60:
                        lessons.append(
                            f"Predecir contra el momentum falla {pct_wrong:.0f}% "
                            f"({wrong}/{len(against_mom)}). Respetar la tendencia del precio."
                        )

            # 2. MACD alignment
            macd_data = [w for w in windows if w.get("prediction") and w.get("outcome")
                         and w.get("macd_histogram") is not None]
            if len(macd_data) >= 5:
                against_macd = [w for w in macd_data
                                if (w["prediction"] == "UP" and w["macd_histogram"] < 0)
                                or (w["prediction"] == "DOWN" and w["macd_histogram"] > 0)]
                if len(against_macd) >= 3:
                    wrong = sum(1 for w in against_macd if not w["correct"])
                    pct_wrong = wrong / len(against_macd) * 100
                    if pct_wrong > 60:
                        lessons.append(
                            f"Predecir contra MACD falla {pct_wrong:.0f}%. "
                            f"MACD+ = preferir UP, MACD- = preferir DOWN."
                        )

            # 3. Buy pressure alignment
            bp_data = [w for w in windows if w.get("prediction") and w.get("outcome")
                       and w.get("bb_pct") is not None]
            # Using bb_pct as proxy since we have it in the query

            # 4. High confidence but wrong
            high_conf_wrong = [w for w in windows
                               if w.get("confidence") and w["confidence"] >= 70
                               and w.get("correct") == 0 and w.get("outcome")]
            if len(high_conf_wrong) >= 3:
                total_high = len([w for w in windows
                                  if w.get("confidence") and w["confidence"] >= 70
                                  and w.get("outcome")])
                if total_high > 0:
                    fail_rate = len(high_conf_wrong) / total_high * 100
                    if fail_rate > 50:
                        lessons.append(
                            f"Confianza alta (>=70) falla {fail_rate:.0f}%. "
                            f"Ser mas conservador con la confianza."
                        )

            # 5. Consecutive losses in same direction
            recent = windows[:10]
            up_recent = [w for w in recent if w.get("prediction") == "UP" and w.get("outcome")]
            down_recent = [w for w in recent if w.get("prediction") == "DOWN" and w.get("outcome")]
            if len(up_recent) >= 4:
                up_wrong = sum(1 for w in up_recent if not w["correct"])
                if up_wrong / len(up_recent) > 0.7:
                    lessons.append(
                        f"UP reciente falla {up_wrong}/{len(up_recent)}. "
                        f"Considerar sesgo alcista — evaluar DOWN mas."
                    )
            if len(down_recent) >= 4:
                dn_wrong = sum(1 for w in down_recent if not w["correct"])
                if dn_wrong / len(down_recent) > 0.7:
                    lessons.append(
                        f"DOWN reciente falla {dn_wrong}/{len(down_recent)}. "
                        f"Considerar sesgo bajista — evaluar UP mas."
                    )

            # 6. RSI extreme accuracy
            rsi_extreme_up = [w for w in windows if w.get("rsi") and w["rsi"] > 70
                              and w.get("prediction") == "UP" and w.get("outcome")]
            if len(rsi_extreme_up) >= 3:
                wrong = sum(1 for w in rsi_extreme_up if not w["correct"])
                if wrong / len(rsi_extreme_up) > 0.6:
                    lessons.append(
                        f"UP con RSI>70 falla {wrong}/{len(rsi_extreme_up)}. "
                        f"RSI sobrecompra sugiere reversal, no continuar UP."
                    )

            rsi_extreme_dn = [w for w in windows if w.get("rsi") and w["rsi"] < 30
                              and w.get("prediction") == "DOWN" and w.get("outcome")]
            if len(rsi_extreme_dn) >= 3:
                wrong = sum(1 for w in rsi_extreme_dn if not w["correct"])
                if wrong / len(rsi_extreme_dn) > 0.6:
                    lessons.append(
                        f"DOWN con RSI<30 falla {wrong}/{len(rsi_extreme_dn)}. "
                        f"RSI sobreventa sugiere rebote, no continuar DOWN."
                    )

            # 7. Learn early bet threshold from actual performance
            try:
                ev = db.get_early_vs_delayed_stats()
                early = ev["early"]
                delayed = ev["delayed"]
                if early["total"] >= 5 and delayed["total"] >= 5:
                    early_wr = early["wins"] / early["total"] * 100
                    delayed_wr = delayed["wins"] / delayed["total"] * 100
                    if early_wr > delayed_wr + 10:
                        # Early bets doing better — lower threshold to bet early more
                        new_thresh = max(60, self.early_bet_threshold - 3)
                        if new_thresh != self.early_bet_threshold:
                            lessons.append(
                                f"Apuestas tempranas ({early_wr:.0f}% WR) superan "
                                f"las tardias ({delayed_wr:.0f}%). "
                                f"Umbral bajado a {new_thresh}%."
                            )
                            self.early_bet_threshold = new_thresh
                            db.save_ai_memory("early_bet_threshold", new_thresh)
                    elif delayed_wr > early_wr + 10:
                        # Delayed bets doing better — raise threshold
                        new_thresh = min(90, self.early_bet_threshold + 3)
                        if new_thresh != self.early_bet_threshold:
                            lessons.append(
                                f"Apuestas tardias ({delayed_wr:.0f}% WR) superan "
                                f"las tempranas ({early_wr:.0f}%). "
                                f"Umbral subido a {new_thresh}%."
                            )
                            self.early_bet_threshold = new_thresh
                            db.save_ai_memory("early_bet_threshold", new_thresh)
            except Exception:
                pass

            # Save only new lessons (avoid duplicates)
            existing = set(self.learned_lessons)
            new_lessons = [l for l in lessons if l not in existing]
            for lesson in new_lessons[:3]:  # Max 3 new lessons per cycle
                self.add_lesson(lesson)

            if new_lessons:
                log.info(f"Auto-learned {len(new_lessons)} lessons from DB analysis")

        except Exception as e:
            log.error(f"Auto-learn error: {e}")

    # ── Trend Alignment Trust Boost ──────────────────────────

    def apply_trend_alignment(self, signal: AISignal,
                              trend_direction: str | None) -> AISignal:
        """
        If multi-timeframe trends (15m + 1h) align with prediction,
        boost trust by 20%. If they contradict, penalize.
        """
        if trend_direction is None:
            return signal

        if trend_direction == signal.prediction:
            # Aligned — boost trust (max 1 boost per hour to prevent inflation)
            now = time.time()
            with self._trust_lock:
                old_trust = self.trust_score
                if now - self._last_trend_boost_time >= 3600:
                    boost = min(self.TRUST_MAX - self.trust_score,
                                self.trust_score * 0.20)
                    self.trust_score = min(self.TRUST_MAX,
                                           self.trust_score + boost)
                    self._last_trend_boost_time = now
                    new_trust = self.trust_score
                else:
                    new_trust = self.trust_score
                    boost = 0
            signal.confidence = min(90, signal.confidence + 5)
            if boost > 0:
                signal.reasoning += (
                    f" | TREND_ALIGN: 15m+1h={trend_direction} confirma {signal.prediction}, "
                    f"trust {old_trust:.0f}->{new_trust:.0f} (+20%)"
                )
                log.info(f"Trend alignment boost: trust {old_trust:.0f}->"
                         f"{new_trust:.0f}, conf +5")
            else:
                signal.reasoning += (
                    f" | TREND_ALIGN: 15m+1h={trend_direction} confirma {signal.prediction}, "
                    f"trust boost on cooldown (conf +5 only)"
                )
                log.info(f"Trend alignment: trust boost on cooldown, conf +5 only")
        elif ((trend_direction == "UP" and signal.prediction == "DOWN") or
              (trend_direction == "DOWN" and signal.prediction == "UP")):
            # Contradicts — penalize confidence
            penalty = 8
            old_conf = signal.confidence
            signal.confidence = max(30, signal.confidence - penalty)
            signal.reasoning += (
                f" | TREND_CONTRA: 15m+1h={trend_direction} contradice "
                f"{signal.prediction}, conf {old_conf}->{signal.confidence}"
            )
            log.info(f"Trend contradiction penalty: conf {old_conf}->"
                     f"{signal.confidence}")

        return signal

    # ── Liquidation Reversal Check ───────────────────────────

    def check_liquidation_reversal(self, signal: AISignal,
                                   liq_summary: dict) -> tuple[AISignal, bool]:
        """
        Check if liquidation spike signals imminent reversal.

        Logic:
        - If spike of BUY liquidations (shorts liquidated) and we predict DOWN → reversal risk
        - If spike of SELL liquidations (longs liquidated) and we predict UP → reversal risk
        - If spike ratio > 5x → too chaotic, SKIP entirely

        Returns:
            (signal, should_skip: bool)
            should_skip=True means don't bet at all this window.
        """
        if not liq_summary.get("spike"):
            return signal, False

        spike_ratio = liq_summary.get("spike_ratio", 0)
        net_side = liq_summary.get("net_side", "NEUTRAL")
        total_vol = liq_summary.get("total_vol", 0)

        # Extreme chaos — skip entirely
        liq_chaos_thresh = float(
            (rules.get_section("risk") or {}).get("liq_chaos_threshold", 5.0)
        )
        if spike_ratio > liq_chaos_thresh:
            signal.signal_quality = "LOW_QUALITY"
            signal.reasoning += (
                f" | LIQ_CHAOS: spike {spike_ratio:.1f}x normal "
                f"(${total_vol:,.0f}), mercado demasiado caotico — SKIP"
            )
            log.warning(f"Liquidation CHAOS: {spike_ratio:.1f}x spike "
                        f"(${total_vol:,.0f}) — skipping bet")
            return signal, True

        # Reversal detection
        # BUY liquidations = shorts squeezed = bullish pressure
        # SELL liquidations = longs squeezed = bearish pressure
        liq_implies = "UP" if net_side == "BUY" else "DOWN" if net_side == "SELL" else None

        if liq_implies and liq_implies != signal.prediction:
            old_conf = signal.confidence
            penalty = min(20, int(spike_ratio * 5))
            signal.confidence = max(25, signal.confidence - penalty)
            signal.reasoning += (
                f" | LIQ_REVERSAL: {net_side} liquidations spike "
                f"{spike_ratio:.1f}x (${total_vol:,.0f}) implies {liq_implies}, "
                f"contradicts {signal.prediction}. conf {old_conf}->{signal.confidence}"
            )
            log.warning(f"Liquidation reversal alert: {net_side} spike implies "
                        f"{liq_implies}, prediction is {signal.prediction}, "
                        f"conf -{penalty}")
        elif liq_implies == signal.prediction:
            # Liquidations confirm our direction — slight boost
            boost = min(5, int(spike_ratio))
            signal.confidence = min(90, signal.confidence + boost)
            signal.reasoning += (
                f" | LIQ_CONFIRM: {net_side} liquidations confirm {signal.prediction}"
            )

        return signal, False

    # ── Signal Quality (ATR gate) ─────────────────────────────

    def mark_signal_quality(self, signal: AISignal,
                            quality: str, reason: str) -> AISignal:
        """Mark signal quality based on ATR quality gate."""
        if quality == "LOW_QUALITY":
            signal.signal_quality = "LOW_QUALITY"
            signal.reasoning += f" | ATR_LOW_QUALITY: {reason}"
            log.info(f"Signal marked LOW_QUALITY: {reason}")
        elif quality == "HIGH_QUALITY":
            signal.signal_quality = "HIGH_QUALITY"
            signal.reasoning += f" | ATR_HIGH_QUALITY: {reason}"
            log.info(f"Signal marked HIGH_QUALITY: {reason}")
        return signal

    # ── Confidence Calibration ─────────────────────────────────

    def _load_confidence_calibration(self):
        """Load actual accuracy per confidence bucket from DB."""
        try:
            self.confidence_calibration = db.get_confidence_calibration(limit=50)
            if self.confidence_calibration:
                log.info(f"Confidence calibration loaded: {self.confidence_calibration}")
        except Exception as e:
            log.debug(f"Calibration load error: {e}")

    def calibrate_confidence(self, raw_confidence: int) -> int:
        """
        Adjust reported confidence based on historical accuracy.
        If the model says 70% but historically 70s only hit 55%,
        we return 55 as the calibrated confidence.
        """
        if not self.confidence_calibration:
            return raw_confidence

        bucket = (raw_confidence // 10) * 10
        actual_accuracy = self.confidence_calibration.get(bucket)

        if actual_accuracy is not None:
            # Blend: 60% historical accuracy + 40% reported confidence
            calibrated = round(actual_accuracy * 0.6 + raw_confidence * 0.4)
            calibrated = max(20, min(95, calibrated))
            if calibrated != raw_confidence:
                log.info(f"Calibration: {raw_confidence}% -> {calibrated}% "
                         f"(bucket {bucket}: {actual_accuracy}% real)")
            return calibrated
        return raw_confidence

    # ── Volatility Regime Bias Correction ──────────────────────

    def update_volatility_regime(self, ta: dict, hmm_regime: dict | None = None):
        """Classify current volatility regime.

        Priority:
            1. HMM Markov regime snapshot (if provided and posterior >= 0.60)
            2. Legacy ATR/volatility thresholds (retrocompatible fallback)
        """
        # ── HMM-driven regime (preferred when confident enough) ──
        if hmm_regime and float(hmm_regime.get("confidence", 0.0)) >= 0.60:
            label = hmm_regime.get("label", "")
            if label in ("TREND_BULL", "TREND_BEAR", "HIGH_VOL_SHOCK"):
                self.volatility_regime = "HIGH"
            elif label == "LOW_VOL_DRIFT":
                self.volatility_regime = "LOW"
            elif label == "RANGE_CHOP":
                self.volatility_regime = "NORMAL"
            else:
                self.volatility_regime = "NORMAL"
            return

        # ── Legacy ATR/volatility fallback ──
        vol = ta.get("volatility", 0)
        atr_pct = ta.get("atr_pct", 0)

        if vol > 0.05 or atr_pct > 0.08:
            self.volatility_regime = "HIGH"
        elif vol < 0.01 and atr_pct < 0.02:
            self.volatility_regime = "LOW"
        else:
            self.volatility_regime = "NORMAL"

    def apply_bias_correction(self, signal: AISignal, ta: dict,
                              hmm_regime: dict | None = None) -> AISignal:
        """
        Penalize constant UP predictions in bearish volatility regimes.
        If market is in HIGH volatility + bearish indicators and AI says UP,
        reduce confidence. When a HMM regime snapshot is provided it takes
        precedence over the ATR-based volatility classifier.
        """
        self.update_volatility_regime(ta, hmm_regime=hmm_regime)

        if signal.prediction != "UP":
            return signal

        # Only correct in HIGH volatility with bearish indicators
        if self.volatility_regime != "HIGH":
            return signal

        bearish_count = 0
        if ta.get("momentum", 0) < -0.005:
            bearish_count += 1
        if ta.get("macd_histogram", 0) < 0:
            bearish_count += 1
        if ta.get("ema_cross") == "BEARISH":
            bearish_count += 1
        if ta.get("trend") == "BEARISH":
            bearish_count += 1

        if bearish_count >= 3:
            penalty = min(15, bearish_count * 4)
            old_conf = signal.confidence
            signal.confidence = max(30, signal.confidence - penalty)
            signal.reasoning += (
                f" | BIAS_CORR: volatilidad alta + {bearish_count} ind. bajistas, "
                f"UP penalizado {old_conf}->{signal.confidence}"
            )
            log.info(f"Bias correction: UP penalty -{penalty} "
                     f"(vol_regime={self.volatility_regime}, "
                     f"bearish_indicators={bearish_count})")

        return signal

    # ── Market signal check (for early betting) ────────────────

    def _count_indicator_alignment(self, pred: str, ta: dict) -> tuple:
        """Count how many indicators confirm vs contradict a prediction.
        Returns (confirms, contradicts)."""
        confirms = 0
        contradicts = 0

        # Momentum
        mom = ta.get("momentum", 0)
        if mom > 0.003:
            confirms += 1 if pred == "UP" else 0
            contradicts += 1 if pred == "DOWN" else 0
        elif mom < -0.003:
            confirms += 1 if pred == "DOWN" else 0
            contradicts += 1 if pred == "UP" else 0

        # MACD histogram
        macd = ta.get("macd_histogram", 0)
        if macd > 0:
            confirms += 1 if pred == "UP" else 0
            contradicts += 1 if pred == "DOWN" else 0
        elif macd < 0:
            confirms += 1 if pred == "DOWN" else 0
            contradicts += 1 if pred == "UP" else 0

        # Buy pressure
        bp = ta.get("tick_buy_pressure", 50)
        if bp > 55:
            confirms += 1 if pred == "UP" else 0
            contradicts += 1 if pred == "DOWN" else 0
        elif bp < 45:
            confirms += 1 if pred == "DOWN" else 0
            contradicts += 1 if pred == "UP" else 0

        # Trend
        trend = ta.get("trend", "")
        if "BULLISH" in trend:
            confirms += 1 if pred == "UP" else 0
            contradicts += 1 if pred == "DOWN" else 0
        elif "BEARISH" in trend:
            confirms += 1 if pred == "DOWN" else 0
            contradicts += 1 if pred == "UP" else 0

        # EMA cross
        ema = ta.get("ema_cross", "")
        if ema == "BULLISH":
            confirms += 1 if pred == "UP" else 0
            contradicts += 1 if pred == "DOWN" else 0
        elif ema == "BEARISH":
            confirms += 1 if pred == "DOWN" else 0
            contradicts += 1 if pred == "UP" else 0

        return confirms, contradicts

    # ── Re-evaluate signal for betting decision ───────────────

    def reevaluate_for_bet(self, signal, current_price: float,
                           open_price: float, ta: dict,
                           force_decision: bool = False) -> tuple:
        """
        Re-evaluate prediction at ~2 min remaining.
        Returns (should_bet: bool, adjusted_signal: AISignal, reason: str)
        If force_decision=True, will almost always bet (flip if needed).
        """
        if open_price <= 0:
            return False, signal, "Sin precio de apertura"

        price_move_pct = (current_price - open_price) / open_price * 100
        pred = signal.prediction
        conf = signal.confidence

        price_confirms = (
            (pred == "UP" and price_move_pct > 0.003) or
            (pred == "DOWN" and price_move_pct < -0.003)
        )
        price_contradicts = (
            (pred == "UP" and price_move_pct < -0.01) or
            (pred == "DOWN" and price_move_pct > 0.01)
        )

        confirms, contradicts = self._count_indicator_alignment(pred, ta)

        # === Case 1: Strong contradiction — FLIP ===
        if price_contradicts and contradicts >= 2:
            new_pred = "DOWN" if pred == "UP" else "UP"
            new_conf = max(48, min(70, 50 + contradicts * 5))
            signal.prediction = new_pred
            signal.confidence = new_conf
            signal.reasoning += (
                f" | FLIP@2min: precio {price_move_pct:+.3f}% contradice {pred}, "
                f"{contradicts} ind. contra. Cambiado a {new_pred}."
            )
            log.info(f"REEVAL: FLIP {pred}->{new_pred} "
                     f"(price {price_move_pct:+.3f}%, {contradicts} contra)")
            return True, signal, f"Flipped {pred}->{new_pred}"

        # === Case 2: Confirmed ===
        if price_confirms and confirms >= 1:
            boost = min(15, confirms * 3 + int(abs(price_move_pct) * 100))
            signal.confidence = min(90, conf + boost)
            signal.reasoning += (
                f" | CONF@2min: precio {price_move_pct:+.3f}% confirma {pred}, "
                f"{confirms} ind. a favor."
            )
            log.info(f"REEVAL: CONFIRMED {pred} "
                     f"(price {price_move_pct:+.3f}%, {confirms} favor, "
                     f"conf {conf}->{signal.confidence})")
            return True, signal, f"Confirmed {pred}"

        # === Case 3: Force decision (2-min deadline) ===
        if force_decision:
            # Flat market or weak signals — still bet but with adjusted confidence
            if abs(price_move_pct) < 0.003:
                # Nearly flat — follow prediction with slightly reduced conf
                signal.confidence = max(45, conf - 5)
                signal.reasoning += (
                    f" | FLAT@2min: mercado plano ({price_move_pct:+.4f}%), "
                    f"manteniendo {pred} con conf ajustada."
                )
                log.info(f"REEVAL: FLAT — betting {pred} conf={signal.confidence}")
                return True, signal, f"Flat market, keeping {pred}"
            else:
                # Price moved but weakly — follow the price direction
                actual_dir = "UP" if price_move_pct > 0 else "DOWN"
                if actual_dir != pred and contradicts > confirms:
                    signal.prediction = actual_dir
                    signal.confidence = max(45, 50 + int(abs(price_move_pct) * 50))
                    signal.reasoning += (
                        f" | FORCE@2min: precio va {actual_dir} ({price_move_pct:+.3f}%), "
                        f"ajustando prediccion."
                    )
                    log.info(f"REEVAL: FORCE flip to {actual_dir}")
                else:
                    signal.confidence = max(45, conf - 5)
                    signal.reasoning += f" | FORCE@2min: decidiendo {pred}."
                    log.info(f"REEVAL: FORCE keeping {pred}")
                return True, signal, f"Forced decision: {signal.prediction}"

        # === Case 4: No force, mixed — skip ===
        log.info(f"REEVAL: SKIP mixed (price {price_move_pct:+.3f}%, "
                 f"{confirms} favor, {contradicts} contra)")
        return False, signal, "Senales mixtas"

    # ── Quick Second Opinion (tactical LLM consult) ──────────

    def quick_second_opinion_react(self, current_price: float, open_price: float,
                                    ta: dict, elapsed_sec: float,
                                    time_left_sec: float,
                                    poly_quote: dict | None,
                                    source: str = "?",
                                    max_rounds: int = 3) -> dict:
        """ReAct second-opinion: LLM can request specific tools mid-reasoning.

        Vibe-Trading inspired — instead of dumping full TA+CVD+GEX payload,
        we send a thin market skeleton + list of available tools. LLM responds
        with either:
            {"tool": "<name>", "args": {...}}   → we run it, append observation, loop
            {"direction": "UP|DOWN", "confidence": 10-95, "veto": bool, "reason": "..."} → done

        Capped at ``max_rounds`` tool calls to bound latency and tokens.
        Falls back to legacy quick_second_opinion if tool_broker is missing.
        """
        from . import swarm as _swarm

        result = {
            "direction": "", "confidence": 0, "veto": False,
            "reasoning": "", "latency_ms": 0.0, "error": "",
            "source": f"{source}:react", "tool_calls": 0, "tools_used": [],
        }
        broker = getattr(self, "tool_broker", None)
        if not broker:
            # No broker wired — degrade gracefully to legacy path.
            legacy = self.quick_second_opinion(
                current_price, open_price, ta, elapsed_sec,
                time_left_sec, poly_quote, source=source,
            )
            legacy["source"] = f"{source}:react-fallback"
            return legacy

        pcfg = self._get_primary_config()
        api_type = pcfg.get("api_type", "")
        if not pcfg.get("api_key") and api_type != "local":
            result["error"] = "no_api_key"
            return result

        move_pct = ((current_price - open_price) / open_price * 100) if open_price else 0.0
        poly_line = ""
        if isinstance(poly_quote, dict):
            up_c = poly_quote.get("up_price_cents")
            dn_c = poly_quote.get("down_price_cents")
            if up_c is not None and dn_c is not None:
                poly_line = f"Polymarket: UP={up_c:.0f}c DOWN={dn_c:.0f}c"

        base_pred = self.last_signal.prediction if self.last_signal else "?"
        base_conf = self.last_signal.confidence if self.last_signal else 0

        tool_list = ", ".join(broker.keys())
        system_prompt = (
            "Eres evaluador tactico BTC/USD 5min (Polymarket UP/DOWN) con "
            "tool-calling. Cada turno responde SOLO JSON. Para pedir datos: "
            '{"tool":"<name>","args":{...}}. Para decidir (final): '
            '{"direction":"UP"|"DOWN","confidence":10-95,"veto":false,'
            '"reason":"..."}. '
            f"Tools disponibles: {tool_list}. "
            f"Maximo {max_rounds} llamadas a tools antes del JSON final."
        )

        messages = [
            f"[SNAPSHOT {source}] precio={current_price:.2f} open={open_price:.2f} "
            f"move={move_pct:+.3f}% elapsed={elapsed_sec:.0f}s left={time_left_sec:.0f}s",
            f"RSI={ta.get('rsi','?')} MACD_h={ta.get('macd_histogram','?')} "
            f"ATR%={ta.get('atr_pct','?')} Trend={ta.get('trend','?')}",
            poly_line,
            f"Primary: {base_pred} @ {base_conf}%",
            "Decide: pide tools si dudas, o responde JSON final.",
        ]
        transcript = "\n".join(m for m in messages if m)

        t0 = time.time()
        sem = getattr(_swarm, "_api_semaphore", None)
        acquired = False
        raw: dict | None = None
        try:
            if sem and api_type != "local":
                acquired = sem.acquire(timeout=2)
                if not acquired:
                    result["error"] = "semaphore_busy"
                    result["latency_ms"] = round((time.time() - t0) * 1000, 1)
                    return result

            for round_idx in range(max_rounds + 1):
                try:
                    raw = self._call_primary_api(system_prompt, transcript)
                except Exception as e:
                    result["error"] = f"api_fail:{e}"
                    break

                # Final decision?
                d = (raw.get("direction") or raw.get("prediction") or "").upper().strip()
                if d in ("UP", "DOWN"):
                    result["direction"] = d
                    try:
                        result["confidence"] = max(10, min(95, int(raw.get("confidence", 51))))
                    except (TypeError, ValueError):
                        result["confidence"] = 51
                    result["veto"] = bool(raw.get("veto", False))
                    result["reasoning"] = (raw.get("reason") or raw.get("reasoning") or "")[:200]
                    break

                # Tool request?
                tool = raw.get("tool") or raw.get("action")
                if tool and round_idx < max_rounds:
                    args = raw.get("args") or raw.get("arguments") or {}
                    fn = broker.get(tool)
                    if not fn:
                        transcript += f"\n[TOOL {tool}] ERROR: unknown tool"
                    else:
                        try:
                            obs = fn(**args) if isinstance(args, dict) else fn()
                            obs_str = str(obs)[:500]
                            transcript += f"\n[TOOL {tool}({args})] => {obs_str}"
                            result["tool_calls"] += 1
                            result["tools_used"].append(tool)
                        except Exception as e:
                            transcript += f"\n[TOOL {tool}] ERROR: {e}"
                    continue

                # No tool, no direction — force final
                transcript += "\n[SYSTEM] Turno final. Responde JSON con direction+confidence+veto."
        finally:
            if acquired and sem:
                sem.release()

        result["latency_ms"] = round((time.time() - t0) * 1000, 1)
        if not result["direction"] and not result["error"]:
            # Exhausted rounds without a decision — momentum fallback
            mom = ta.get("momentum", 0) if ta else 0
            result["direction"] = "UP" if mom >= 0 else "DOWN"
            result["confidence"] = 50
            result["veto"] = True
            result["reasoning"] = "exhausted_rounds_no_decision"

        log.info(f"[2nd-OP:{source}:react] {result['direction']} "
                 f"{result['confidence']}% veto={result['veto']} "
                 f"tools={result['tool_calls']}{result['tools_used']} "
                 f"({result['latency_ms']}ms)")
        return result

    def quick_second_opinion(self, current_price: float, open_price: float,
                              ta: dict, elapsed_sec: float,
                              time_left_sec: float,
                              poly_quote: dict | None,
                              source: str = "?",
                              context: dict | None = None) -> dict:
        """Fire a fast tactical consult to the primary LLM.

        Returns a dict with keys:
            direction (str "UP"/"DOWN" or ""),
            confidence (int 10-95 or 0 on error),
            veto (bool),
            reasoning (str),
            latency_ms (float),
            error (str, "" on success),
            source (str, same as input).

        Never raises. Uses call_fast_prediction first, falls back to
        _call_primary_api. Rate-limited via the swarm _api_semaphore so it
        shares the same 2-slot budget as primary + swarm calls.

        Hot-delegation to ReAct tool-calling variant when
        ``dynamic_rules.json:second_opinion.react_enabled`` is true and the
        engine has registered a ``tool_broker`` on this AIEngine instance.
        """
        if (rules.get("second_opinion", "react_enabled", False)
                and getattr(self, "tool_broker", None)):
            return self.quick_second_opinion_react(
                current_price=current_price, open_price=open_price,
                ta=ta, elapsed_sec=elapsed_sec, time_left_sec=time_left_sec,
                poly_quote=poly_quote, source=source,
                max_rounds=int(rules.get("second_opinion", "react_max_rounds", 3)),
            )
        from . import swarm as _swarm

        result = {
            "direction": "",
            "confidence": 0,
            "veto": False,
            "reasoning": "",
            "latency_ms": 0.0,
            "error": "",
            "source": source,
        }

        pcfg = self._get_primary_config()
        api_type = pcfg.get("api_type", "")
        if not pcfg.get("api_key") and api_type != "local":
            result["error"] = "no_api_key"
            return result

        # Compact TA snapshot for the prompt (defensive gets)
        def _g(k, default="?"):
            v = ta.get(k, default) if ta else default
            return v
            
        ctx = context or {}
        def _c(k, default="?"):
            v = ctx.get(k, default) if ctx else default
            return v

        move_pct = 0.0
        if open_price and open_price > 0:
            move_pct = (current_price - open_price) / open_price * 100

        poly_text = ""
        if isinstance(poly_quote, dict):
            up_cents = poly_quote.get("up_price_cents")
            dn_cents = poly_quote.get("down_price_cents")
            if up_cents is not None and dn_cents is not None:
                poly_text = (f"\nPolymarket odds: UP={up_cents:.0f}c "
                             f"DOWN={dn_cents:.0f}c")

        base_pred = self.last_signal.prediction if self.last_signal else "?"
        base_conf = self.last_signal.confidence if self.last_signal else 0

        user_prompt = (
            f"SNAPSHOT BTC/USD 5min window [{source}]\n"
            f"Precio actual: {current_price:.2f} | Apertura: {open_price:.2f} "
            f"| Move: {move_pct:+.3f}%\n"
            f"Elapsed: {elapsed_sec:.0f}s | TimeLeft: {time_left_sec:.0f}s\n"
            f"RSI={_g('rsi')} MACD_hist={_g('macd_histogram')} "
            f"ATR%={_g('atr_pct')} BB%={_g('bb_pct')}\n"
            f"Momentum={_g('momentum')} Trend={_g('trend')} "
            f"BuyPressure={_g('tick_buy_pressure')}%\n"
            f"Funding={_c('funding_rate')} OB_imb={_c('order_book_imbalance')} "
            f"CVD_imb={_c('cvd_imbalance_pct')}%"
            f"{poly_text}\n"
            f"Sistema actual predice: {base_pred} @ {base_conf}%\n\n"
            f"Tu segunda opinion binaria UP/DOWN con confidence 10-95 y "
            f"veto opcional. Responde SOLO el JSON definido por el system."
        )

        system_prompt = rules.get_prompt("second_opinion_system") or (
            "Eres un evaluador tactico binario BTC/USD 5min. Responde SOLO "
            "JSON: {\"direction\":\"UP\",\"confidence\":70,\"veto\":false,"
            "\"reason\":\"...\"}"
        )

        t0 = time.time()
        raw: dict | None = None
        sem = getattr(_swarm, "_api_semaphore", None)
        acquired = False
        try:
            if sem and api_type != "local":
                acquired = sem.acquire(timeout=2)
                if not acquired:
                    result["error"] = "semaphore_busy"
                    result["latency_ms"] = round((time.time() - t0) * 1000, 1)
                    log.info(f"[2nd-OP:{source}] semaphore busy — skip")
                    return result

            # Phase 1 fast path
            try:
                raw = _swarm.call_fast_prediction(
                    api_type, pcfg.get("api_key", ""),
                    pcfg["model"], system_prompt, user_prompt,
                    pcfg.get("base_url"),
                )
            except Exception as e:
                log.debug(f"[2nd-OP:{source}] fast_prediction raised: {e}")
                raw = None

            # Fallback to full traditional call if fast path is empty
            if not raw or not raw.get("prediction"):
                try:
                    raw = self._call_primary_api(system_prompt, user_prompt)
                except Exception as e:
                    result["error"] = f"api_fail:{e}"
                    result["latency_ms"] = round(
                        (time.time() - t0) * 1000, 1)
                    log.warning(
                        f"[2nd-OP:{source}] primary API failed: {e}")
                    return result
        finally:
            if acquired and sem:
                sem.release()

        # call_fast_prediction returns {"prediction", "confidence", ...}
        # _call_primary_api also returns that shape. Our system prompt
        # asks for {"direction",...}, so accept both field names.
        direction = (raw.get("direction") or raw.get("prediction") or "").upper().strip()
        if direction not in ("UP", "DOWN"):
            # Tie-breaker on momentum
            mom = ta.get("momentum", 0) if ta else 0
            direction = "UP" if mom >= 0 else "DOWN"

        conf_raw = raw.get("confidence", 51)
        try:
            confidence = int(conf_raw)
        except (TypeError, ValueError):
            confidence = 51
        confidence = max(10, min(95, confidence))

        veto = bool(raw.get("veto", False))
        reasoning = (raw.get("reason") or raw.get("reasoning") or "").strip()

        result.update({
            "direction": direction,
            "confidence": confidence,
            "veto": veto,
            "reasoning": reasoning[:200],
            "latency_ms": round((time.time() - t0) * 1000, 1),
        })
        log.info(f"[2nd-OP:{source}] {direction} {confidence}% "
                 f"veto={veto} ({result['latency_ms']}ms) — {reasoning[:80]}")
        return result

    # ── Outcomes ───────────────────────────────────────────

    def record_outcome(self, prediction: str, actual: str, confidence: int = 0):
        self.prediction_history.append({
            "prediction": prediction, "actual": actual,
            "correct": prediction == actual, "ts": time.time(),
        })
        if len(self.prediction_history) > 20:
            self.prediction_history = self.prediction_history[-20:]

        # Update trust score
        self.update_trust(prediction, actual, confidence)

        # Refresh DB accuracy and calibration
        try:
            self.db_accuracy = db.get_historical_accuracy(limit=50)
            self.db_patterns = self.db_accuracy.get("patterns", "")
            self._detect_bias()
            self._load_confidence_calibration()
        except Exception:
            pass

        # Auto-learn from accumulated data (independent of local LLM)
        self._auto_learn_from_db()

    # ── Epistemic Reflection (async post-loss analysis) ──────

    def trigger_reflection(self, prediction: str, actual: str,
                           confidence: int, ta_snapshot: dict,
                           cvd_summary: dict | None = None):
        """
        Launch an async reflection after a LOST window.
        Calls a fast LLM (DeepSeek) asking: why did we lose?
        Extracts a short causal lesson and stores it for the next prompt.

        This is NON-BLOCKING — runs in a daemon thread.
        """
        if prediction == actual:
            return  # Only reflect on losses

        if self._reflection_in_flight:
            return  # One reflection at a time

        self._reflection_in_flight = True
        threading.Thread(
            target=self._run_reflection,
            args=(prediction, actual, confidence, ta_snapshot, cvd_summary),
            daemon=True,
            name="epistemic-reflection",
        ).start()

    def _run_reflection(self, prediction: str, actual: str,
                        confidence: int, ta: dict,
                        cvd_summary: dict | None):
        """Background worker: ask LLM for causal analysis of the loss."""
        try:
            pcfg = self._get_primary_config()
            if not pcfg.get("api_key") and pcfg.get("api_type") != "local":
                return

            # Build a compact context for reflection
            cvd_text = ""
            if cvd_summary:
                cvd_imb = cvd_summary.get("cvd_imbalance_pct", 0)
                cvd_net = cvd_summary.get("cvd_net", 0)
                cvd_text = (f"CVD_imbalance: {cvd_imb:+.1f}%, "
                            f"CVD_net: ${cvd_net:,.0f}")

            reflection_prompt = (
                f"Acabamos de PERDER una predicción de BTC/USD a 5 minutos.\n"
                f"Predijimos: {prediction} (confianza {confidence}%)\n"
                f"Resultado real: {actual}\n\n"
                f"Datos al momento de la predicción:\n"
                f"RSI={ta.get('rsi', 'N/A'):.1f}, "
                f"MACD_hist={ta.get('macd_histogram', 'N/A'):+.2f}, "
                f"Momentum={ta.get('momentum', 'N/A'):+.4f}%, "
                f"BB%={ta.get('bb_pct', 'N/A'):.1f}, "
                f"Trend={ta.get('trend', 'N/A')}, "
                f"Buy_pressure={ta.get('tick_buy_pressure', 'N/A')}%\n"
                f"{f'Order Flow: {cvd_text}' if cvd_text else ''}\n\n"
                f"Responde en MÁXIMO 25 palabras:\n"
                f"1. ¿Cuál fue la causa probable? (ej: Absorción Pasiva, "
                f"Divergencia CVD, Trampa de Liquidez, Momentum Exhaustion)\n"
                f"2. ¿Qué patrón debemos vigilar para no repetirlo?\n\n"
                f"Formato: CAUSA: [etiqueta]. REGLA: [acción concreta]."
            )

            system_msg = ("Eres un analista cuantitativo de post-mortem. "
                          "Diagnostica la causa raíz de cada pérdida en "
                          "máximo 25 palabras. Sé técnico y directo.")

            r = self._call_primary_api(system_msg, reflection_prompt)
            # _call_primary_api returns parsed dict, but reflection expects raw text
            # If the response is a dict (structured), extract reasoning; otherwise use raw
            if isinstance(r, dict):
                lesson = r.get("reasoning", r.get("causa", str(r)))[:200]
            else:
                lesson = str(r)[:200]

            # Store the reflection lesson (thread-safe)
            with self._reflection_lock:
                self.recent_reflection_lesson = lesson

            # Also add to persistent lessons
            self.add_lesson(f"[REFLECTION] {lesson}")
            log.info(f"[REFLECTION] Post-loss lesson: {lesson}")

        except Exception as e:
            log.debug(f"[REFLECTION] Failed (non-fatal): {e}")
        finally:
            self._reflection_in_flight = False

    def get_and_clear_reflection(self) -> str:
        """Get the latest reflection lesson and clear it (consumed once)."""
        with self._reflection_lock:
            lesson = self.recent_reflection_lesson
            self.recent_reflection_lesson = ""
        return lesson

    def get_accuracy_stats(self) -> dict:
        if not self.prediction_history:
            if self.db_accuracy.get("total", 0) > 0:
                return {
                    "total": self.db_accuracy["total"],
                    "correct": self.db_accuracy.get("correct", 0),
                    "pct": self.db_accuracy.get("pct", 0),
                    "up_acc": self.db_accuracy.get("up_pct", 0),
                    "down_acc": self.db_accuracy.get("dn_pct", 0),
                    "last5": self.db_accuracy.get("last10", "")[:5],
                }
            return {"total": 0, "correct": 0, "pct": 0,
                    "up_acc": 0, "down_acc": 0, "last5": ""}
        total = len(self.prediction_history)
        correct = sum(1 for p in self.prediction_history if p["correct"])
        up_preds = [p for p in self.prediction_history if p["prediction"] == "UP"]
        down_preds = [p for p in self.prediction_history if p["prediction"] == "DOWN"]
        up_acc = (sum(1 for p in up_preds if p["correct"]) / len(up_preds) * 100) if up_preds else 0
        down_acc = (sum(1 for p in down_preds if p["correct"]) / len(down_preds) * 100) if down_preds else 0
        last5 = "".join("W" if p["correct"] else "L" for p in self.prediction_history[-5:])
        db_total = self.db_accuracy.get("total", 0)
        if db_total > total:
            total = db_total
            correct = self.db_accuracy.get("correct", correct)
        return {
            "total": total, "correct": correct,
            "pct": round(correct / total * 100, 1) if total else 0,
            "up_acc": round(up_acc, 1), "down_acc": round(down_acc, 1),
            "last5": last5,
        }

    def update_confidence(self, signal: AISignal, current_price: float,
                          open_price: float, ta: dict, time_remaining: int) -> AISignal:
        """Mid-window confidence adjustment without API call."""
        if open_price <= 0:
            return signal
        price_move_pct = (current_price - open_price) / open_price * 100
        adjustment = 0
        reasons = []

        # Price movement alignment
        if signal.prediction == "UP":
            if price_move_pct > 0.02:
                adjustment += min(10, int(price_move_pct * 200))
                reasons.append(f"precio +{price_move_pct:.3f}% confirma UP")
            elif price_move_pct < -0.02:
                adjustment -= min(15, int(abs(price_move_pct) * 300))
                reasons.append(f"precio {price_move_pct:.3f}% contradice UP")
        else:
            if price_move_pct < -0.02:
                adjustment += min(10, int(abs(price_move_pct) * 200))
                reasons.append(f"precio {price_move_pct:.3f}% confirma DOWN")
            elif price_move_pct > 0.02:
                adjustment -= min(15, int(price_move_pct * 300))
                reasons.append(f"precio +{price_move_pct:.3f}% contradice DOWN")

        # RSI shift
        if ta["rsi"] > 75 and signal.prediction == "UP":
            adjustment -= 5; reasons.append("RSI sobrecompra")
        elif ta["rsi"] < 25 and signal.prediction == "DOWN":
            adjustment -= 5; reasons.append("RSI sobreventa")

        # MACD histogram direction
        if ta["macd_histogram"] > 0 and signal.prediction == "UP":
            adjustment += 3
        elif ta["macd_histogram"] < 0 and signal.prediction == "DOWN":
            adjustment += 3
        elif ta["macd_histogram"] > 0 and signal.prediction == "DOWN":
            adjustment -= 3
        elif ta["macd_histogram"] < 0 and signal.prediction == "UP":
            adjustment -= 3

        # Time factor
        if time_remaining < 60 and abs(price_move_pct) > 0.01:
            if (price_move_pct > 0 and signal.prediction == "UP") or \
               (price_move_pct < 0 and signal.prediction == "DOWN"):
                adjustment += 5
                reasons.append("poco tiempo, tendencia clara")

        new_confidence = max(10, min(95, signal.confidence + adjustment))
        if reasons:
            signal.reasoning += f" | MidUpdate: {'; '.join(reasons)}"
        signal.confidence = new_confidence
        signal.updated_at = time.time()
        signal.update_count += 1
        return signal

    # ── Provider-agnostic API call ──────────────────────────────

    @staticmethod
    def _get_primary_config() -> dict:
        """Read primary agent config from dynamic_rules.json."""
        from . import swarm as _swarm
        primary = _swarm.get_primary_agent()
        if not primary:
            return {}
        api_key_env = primary.get("api_key_env", "")
        return {
            "api_type": primary.get("api_type", "deepseek"),
            "model": primary.get("model", ""),
            "api_key": os.environ.get(api_key_env, "") if api_key_env else "",
            "base_url": primary.get("api_base_url"),
            "system_prompt_key": primary.get("system_prompt_key", ""),
            "agent_id": primary.get("agent_id", "primary"),
        }

    def bull_bear_debate(self, bull_args: list[dict], bear_args: list[dict],
                         market_context: str = "") -> dict:
        """Bull-vs-Bear debate judged by the primary LLM.

        Inspired by Vibe-Trading ``investment_committee``. When the swarm is
        NOT unanimous, we collect UP-side reasoning as the bull case and
        DOWN-side reasoning as the bear case, then ask the primary to rule.

        Args:
            bull_args: list of {display_name, confidence, reasoning} dicts for UP
            bear_args: same shape for DOWN
            market_context: compact TA/CVD/GEX snapshot (optional)

        Returns dict: {direction: "UP"|"DOWN"|"SKIP", confidence: int,
                       reason: str, veto: bool}. On any error returns {"veto": True}.
        """
        if not bull_args and not bear_args:
            return {"veto": True, "reason": "no_args"}

        def _fmt(args: list[dict]) -> str:
            if not args:
                return "(ninguno)"
            return "\n".join(
                f"- {a.get('display_name','?')} ({a.get('confidence',0)}%): "
                f"{(a.get('reasoning') or '')[:280]}"
                for a in args
            )

        system = (
            "Eres juez imparcial de un comite de trading BTC/USD 5-min "
            "(Polymarket UP/DOWN). Recibes argumentos BULL (UP) y BEAR (DOWN) "
            "de agentes disidentes. Evalua cual caso es mas robusto DADO el "
            "contexto de mercado. Responde SOLO JSON: "
            '{"direction":"UP"|"DOWN"|"SKIP","confidence":0-95,'
            '"reason":"...","veto":false}. '
            "Si ambos casos son debiles o se cancelan → direction=SKIP, veto=true. "
            "Confidence refleja conviccion del JUEZ, no promedio de agentes."
        )
        user = (
            f"[CONTEXTO MERCADO]\n{market_context or 'N/A'}\n\n"
            f"[CASO BULL — argumentan UP]\n{_fmt(bull_args)}\n\n"
            f"[CASO BEAR — argumentan DOWN]\n{_fmt(bear_args)}\n\n"
            "Decide y responde JSON."
        )
        try:
            resp = self._call_primary_api(system, user)
            direction = str(resp.get("direction", "SKIP")).upper()
            if direction not in ("UP", "DOWN", "SKIP"):
                direction = "SKIP"
            conf = int(float(resp.get("confidence", 0)))
            conf = max(0, min(95, conf))
            return {
                "direction": direction,
                "confidence": conf,
                "reason": str(resp.get("reason", ""))[:400],
                "veto": bool(resp.get("veto", False)) or direction == "SKIP",
            }
        except Exception as e:
            log.warning(f"[bull_bear_debate] failed: {e}")
            return {"veto": True, "reason": f"error:{e}"}

    def _call_primary_api(self, system_prompt: str, user_prompt: str) -> dict:
        """Call the configured primary AI provider. Returns parsed JSON dict."""
        from . import swarm as _swarm
        pcfg = self._get_primary_config()
        if not pcfg.get("api_key") and pcfg.get("api_type") != "local":
            raise ValueError("Primary agent has no API key configured")

        caller = _swarm._API_CALLERS.get(pcfg["api_type"])
        if not caller:
            raise ValueError(f"Unknown api_type: {pcfg['api_type']}")

        return caller(
            pcfg["api_key"], pcfg["model"],
            system_prompt, user_prompt,
            pcfg.get("base_url"),
        )

    def push_ta_snapshot(self, agent_id: str, ta: dict):
        """Push a TA snapshot into the rolling history for velocity computation."""
        from collections import deque
        if agent_id not in self._ta_history:
            self._ta_history[agent_id] = deque(maxlen=3)
        self._ta_history[agent_id].append({
            "rsi": ta.get("rsi", 0) or 0,
            "momentum": ta.get("momentum", 0) or 0,
            "tick_buy_pressure": ta.get("tick_buy_pressure", 0) or 0,
        })

    def compute_ta_velocities(self, agent_id: str) -> dict:
        """Compute velocity/acceleration from rolling TA history.

        Returns dict with rsi_velocity, rsi_acceleration,
        momentum_velocity, cvd_velocity.  All None if <2 snapshots.
        """
        buf = self._ta_history.get(agent_id)
        result = {
            "rsi_velocity": None,
            "rsi_acceleration": None,
            "momentum_velocity": None,
            "cvd_velocity": None,
        }
        if not buf or len(buf) < 2:
            return result
        t0 = buf[-1]
        tm1 = buf[-2]
        result["rsi_velocity"] = round(t0["rsi"] - tm1["rsi"], 4)
        result["momentum_velocity"] = round(t0["momentum"] - tm1["momentum"], 6)
        result["cvd_velocity"] = round(
            t0["tick_buy_pressure"] - tm1["tick_buy_pressure"], 4
        )
        if len(buf) >= 3:
            tm2 = buf[-3]
            vel_tm1 = tm1["rsi"] - tm2["rsi"]
            result["rsi_acceleration"] = round(
                result["rsi_velocity"] - vel_tm1, 4
            )
        return result

    def predict(self, price: PriceData, ta: dict, news: str,
                context: dict | None = None) -> AISignal:
        pcfg = self._get_primary_config()
        if not pcfg.get("api_key") and pcfg.get("api_type") != "local":
            return self._fallback(price, ta)
        ctx = context or {}
        prompt = self._build_prompt(price, ta, news, ctx)
        system_prompt = self._build_system_prompt()
        
        # Debug: User requested to see the prompt in the console
        log.info(f"====== [DEBUG] SYSTEM PROMPT ======\n{system_prompt}\n"
                 f"====== [DEBUG] USER PROMPT ======\n{prompt}\n===================================")
                 
        try:
            # ── HFT Phase 1: Fast prediction (~0.3-0.5s) ──────────
            from . import swarm as _swarm
            fast_result = _swarm.call_fast_prediction(
                pcfg["api_type"], pcfg.get("api_key", ""),
                pcfg["model"], system_prompt, prompt,
                pcfg.get("base_url"),
            )
            if fast_result and fast_result.get("prediction"):
                r = fast_result
                phase1_ms = round((time.time() - (ctx.get("_t0") or time.time())) * 1000, 1)
                log.info(f"[HFT-Phase1] Fast prediction in ~{phase1_ms}ms: "
                         f"{r.get('prediction')} @ {r.get('confidence')}%")
                # Phase 2: background reasoning (does NOT block the engine)
                self._spawn_phase2_reasoning(pcfg, prompt, r)
                # Build signal with placeholder reasoning (Phase 2 fills DB later)
                r.setdefault("reasoning", "[HFT] Reasoning pending (Phase 2 background)")
                r.setdefault("news_impact", "")
                r.setdefault("risk_score", "MEDIUM")
                r.setdefault("suggested_bet_pct", 0.02)
                r.setdefault("layer_alignment", "")
            else:
                # Phase 1 failed — full traditional call
                log.info("[HFT] Phase 1 failed, falling back to full API call")
                r = self._call_primary_api(system_prompt, prompt)

            confidence = max(0, min(100, int(r.get("confidence", 50))))
            raw_pred = r.get("prediction", "UP").upper().strip()
            # ── Binary tie-breaker: force UP/DOWN, never NEUTRAL ──
            if raw_pred not in ("UP", "DOWN"):
                mom = ta.get("momentum", 0) if ta else 0
                raw_pred = "UP" if mom >= 0 else "DOWN"
                confidence = max(confidence, 51)
                log.info(f"Tie-breaker: '{r.get('prediction')}' forced to "
                         f"{raw_pred} (momentum={mom:+.4f})")
            if confidence == 50:
                confidence = 51
            signal = AISignal(
                prediction=raw_pred,
                confidence=confidence,
                reasoning=r.get("reasoning", "Sin analisis"),
                news_summary=r.get("news_impact", "Sin datos"),
                risk_score=r.get("risk_score", "MEDIUM").upper(),
                suggested_bet_pct=max(0.01, min(0.05, float(r.get("suggested_bet_pct", 0.02)))),
                timestamp=time.time(),
                original_confidence=confidence,
                layer_alignment=r.get("layer_alignment", ""),
            )
            # Apply calibration and bias correction
            signal.confidence = self.calibrate_confidence(signal.confidence)
            # Propagate HMM regime snapshot from the ctx dict if present
            hmm_regime = ctx.get("_hmm_regime") if isinstance(ctx, dict) else None
            signal = self.apply_bias_correction(signal, ta, hmm_regime=hmm_regime)
            signal.usage_tokens = r.get("_usage_tokens")
            self.last_signal = signal
            provider = pcfg.get("api_type", "?")
            log.info(f"AI Signal [{provider}]: {signal.prediction} ({signal.confidence}%) "
                     f"Risk:{signal.risk_score}")
            return signal
        except Exception as e:
            log.error(f"Primary AI error ({pcfg.get('api_type', '?')}): {e}")
            return self._fallback(price, ta)

    def _spawn_phase2_reasoning(self, pcfg: dict, user_prompt: str,
                                fast_result: dict):
        """Spawn a background thread for HFT Phase 2 reasoning backfill."""
        from . import swarm as _swarm
        pred = fast_result.get("prediction", "UP")
        conf = fast_result.get("confidence", 50)
        # Capture window_id and ai_model at spawn time
        window_id = getattr(self, '_current_window_id', None)
        ai_model = pcfg.get("agent_id", "primary")

        def _phase2():
            try:
                result = _swarm.call_background_reasoning(
                    pcfg["api_type"], pcfg.get("api_key", ""),
                    pcfg["model"], pred, conf, user_prompt,
                    pcfg.get("base_url"),
                )
                if result and window_id:
                    reasoning = result.get("reasoning", "")
                    news_impact = result.get("news_impact", "")
                    db.update_prediction_reasoning(
                        window_id, ai_model, reasoning, news_impact,
                    )
                    log.info(f"[HFT-Phase2] Reasoning backfilled for window "
                             f"#{window_id} ({len(reasoning)} chars)")
            except Exception as e:
                log.debug(f"[HFT-Phase2] Background reasoning error: {e}")

        t = threading.Thread(target=_phase2, daemon=True, name="hft-phase2")
        t.start()

    def _build_system_prompt(self) -> str:
        """Build system prompt from primary agent's configured prompt key."""
        pcfg = self._get_primary_config()
        prompt_key = pcfg.get("system_prompt_key", "")
        if prompt_key:
            prompt = rules.get_prompt(prompt_key)
            if prompt:
                log.info(f"System prompt loaded: {prompt_key}")
                return prompt
        # Hardcoded fallback if dynamic_rules.json is missing or empty
        log.info("Using fallback system prompt (no prompt key configured)")
        return (
            "Eres un algoritmo depredador institucional de prediccion de BTC/USD a 5 minutos. "
            "NO eres un analista retail. Responde SOLO con JSON: "
            '{"prediction":"UP","confidence":71,"reasoning":"...","news_impact":"...",'
            '"risk_score":"LOW","suggested_bet_pct":0.03}'
        )

    def get_full_context_prompt(self, rag_block: str = "",
                               regime_snapshot: dict | None = None) -> str:
        """Build the complete shared system prompt for swarm agents.

        Includes: base system prompt + RAG strategy + lessons + bias corrections
        + HMM regime. Swarm agents consume this identical prompt so they reason
        over the same institutional context, producing purely architectural dissent.
        """
        base = self._build_system_prompt()

        # Lessons block
        lessons = ""
        if self.learned_lessons:
            lessons_text = " | ".join(self.learned_lessons[-5:])
            lessons = (
                f"\n\n[LECCIONES APRENDIDAS]\n{lessons_text}"
            )

        # Bias block
        bias = ""
        if self.bias_corrections:
            bias_text = " | ".join(self.bias_corrections.values())
            bias = f"\n\n[CORRECCIONES DE SESGO]\n{bias_text}"

        # RAG strategy block (passed from engine)
        rag = f"\n\n{rag_block}" if rag_block else ""

        # HMM regime block (passed from engine)
        hmm = ""
        if regime_snapshot:
            hmm = (
                f"\n\n[REGIMEN DE MARKOV (HMM)]\n"
                f"Estado oculto actual: {regime_snapshot.get('label', 'N/A')} "
                f"(idx={regime_snapshot.get('state_idx', '?')})\n"
                f"Confianza posterior: {regime_snapshot.get('confidence', 0)*100:.1f}%\n"
                f"Transicion probable: {regime_snapshot.get('transition_to', '?')}"
            )

        # Reflection lesson (one-shot)
        reflection = self.get_and_clear_reflection()
        refl_block = ""
        if reflection:
            refl_block = f"\n\n[REFLEXION POST-PERDIDA]\n{reflection}"

        return base + lessons + bias + rag + hmm + refl_block

    def _build_prompt(self, price: PriceData, ta: dict, news: str,
                      ctx: dict) -> str:
        acc = self.get_accuracy_stats()
        acc_text = "Sin historial" if acc["total"] == 0 else (
            f"Acc:{acc['pct']}% ({acc['correct']}/{acc['total']}) "
            f"UP:{acc['up_acc']}% DN:{acc['down_acc']}% "
            f"Last5:{acc['last5']}"
        )

        # Similar pattern search
        similar_text = ""
        try:
            similar = db.get_similar_predictions(
                ta["rsi"], ta["macd_histogram"], ta["volatility"]
            )
            if similar.get("total", 0) > 0:
                similar_text = similar["text"]
        except Exception:
            pass

        # Bias + lessons (compact)
        corrections = ""
        if self.bias_corrections:
            corrections = " | ".join(self.bias_corrections.values())
        lessons = ""
        if self.learned_lessons:
            lessons = " | ".join(self.learned_lessons[-3:])

        # Epistemic Reflection (from last lost window)
        reflection = self.get_and_clear_reflection()

        # ── HMM Markov regime block (built outside the f-string to avoid
        #    brace-escape gymnastics) ──
        hmm_ctx = ctx.get("_hmm_regime") if isinstance(ctx, dict) else None
        if hmm_ctx:
            _hmm_label = hmm_ctx.get("label", "N/A")
            _hmm_conf = float(hmm_ctx.get("confidence", 0.0)) * 100
            _hmm_probs = hmm_ctx.get("state_probs", [])
            _hmm_trans = hmm_ctx.get("transition_to", {})
            hmm_block = (
                f"[REGIMEN MARKOV (HMM)]\n"
                f"Estado oculto actual: {_hmm_label} "
                f"(conf={_hmm_conf:.0f}%)\n"
                f"Distribucion de estados: {_hmm_probs}\n"
                f"Transicion probable: {_hmm_trans}\n"
                f"REGLA: Si TA contradice el regimen, reduce confianza salvo "
                f"que un sweep/MSS invalide el estado."
            )
        else:
            hmm_block = ""

        # Context features
        trend_slope_15m = ctx.get("trend_slope_15m", "N/A")
        trend_slope_1h = ctx.get("trend_slope_1h", "N/A")
        trend_alignment = ctx.get("trend_alignment")
        trend_align_str = ("ALIGNED" if trend_alignment == 1
                           else "DIVERGENT" if trend_alignment == 0
                           else "NO_DATA")
        signal_quality = ctx.get("signal_quality", "NORMAL")
        quality_reason = ctx.get("quality_reason", "")
        liq_buy = ctx.get("liq_buy_vol_5m", 0)
        liq_sell = ctx.get("liq_sell_vol_5m", 0)
        reversal_alert = ctx.get("reversal_alert", 0)
        funding_rate = ctx.get("funding_rate")
        open_interest = ctx.get("open_interest")
        ob_imbalance = ctx.get("order_book_imbalance")
        vol_zscore = ctx.get("volatility_zscore")
        range_pos = ctx.get("range_position")
        price_accel = ctx.get("price_acceleration")
        return_1m = ctx.get("return_1m")
        return_5m = ctx.get("return_5m")

        # Format optional values
        def fv(val, fmt=".4f"):
            return f"{val:{fmt}}" if val is not None else "N/A"

        # Liquidation-cluster proximity (compact single-line summary).
        _lc = ctx.get("liq_cluster") or {}
        if _lc and _lc.get("nearest_cluster_price") is not None:
            _lc_str = (
                f"nearest=${_lc['nearest_cluster_price']:,.0f} "
                f"({_lc.get('cluster_side','?')}) "
                f"dist={_lc.get('distance_atr','?')}ATR "
                f"mag_pct={_lc.get('magnitude_pct',0)*100:.0f}% "
                f"n={_lc.get('n_events',0)}"
            )
        else:
            _lc_str = "N/A"

        return f"""=== SNAPSHOT BTC/USD @ {datetime.fromtimestamp(price.timestamp).strftime('%H:%M:%S UTC')} ===
Precio: ${price.price:,.2f} ({price.source})

[INDICADORES TECNICOS]
RSI(14): {ta['rsi']:.1f}
MACD_hist: {ta['macd_histogram']:+.2f}
EMA_cross: {ta['ema_cross']}
BB_pct: {ta['bb_pct']:.1f}%
Momentum(5): {ta['momentum']:+.4f}%
Buy_pressure: {ta['tick_buy_pressure']}%
ATR_pct: {ta['atr_pct']:.4f}%
Trend: {ta['trend']}
Soporte: ${ta['support']:,.2f} ({ta['price_vs_support_pct']:.4f}%)
Resistencia: ${ta['resistance']:,.2f} ({ta['price_vs_resistance_pct']:.4f}%)

[MULTI-TIMEFRAME]
Slope_15m: {fv(trend_slope_15m, '.6f')}%/vela
Slope_1h: {fv(trend_slope_1h, '.6f')}%/vela
Trend_alignment: {trend_align_str}

[MICROESTRUCTURA]
Return_1m: {fv(return_1m)}%
Return_5m: {fv(return_5m)}%
Price_acceleration: {fv(price_accel)}
Range_position: {fv(range_pos, '.1f')}% (0=low, 100=high)
Vol_zscore: {fv(vol_zscore, '.3f')}

[DERIVADOS]
Funding_rate: {fv(funding_rate, '.6f')}
Open_interest: {fv(open_interest, '.0f')} BTC
Order_book_imbalance: {fv(ob_imbalance, '.4f')} (>0.55=bid dominant UP, <0.45=ask dominant DOWN)

[LIQUIDACIONES 5min]
Liq_BUY_vol: ${liq_buy:,.0f} (shorts liquidados → presion alcista)
Liq_SELL_vol: ${liq_sell:,.0f} (longs liquidados → presion bajista)
Reversal_alert: {'SI' if reversal_alert else 'NO'}

[LIQ_CLUSTER 2h] {_lc_str}

[CALIDAD DE SENAL]
Signal_quality: {signal_quality}
{f'Motivo: {quality_reason}' if quality_reason else ''}

[POLYMARKET EV]
Strike_price: {f"${ctx.get('poly_strike_price', 0):,.2f}" if ctx.get('poly_strike_price') else "N/A"}
Distancia_al_strike: {f"{ctx.get('poly_strike_dist'):+.2f} USD" if ctx.get('poly_strike_dist') is not None else "N/A"}
UP_odds: {fv(ctx.get('poly_up_odds_cents'), '.1f')}c (payout x{fv(ctx.get('poly_up_payout'), '.2f')})
DOWN_odds: {fv(ctx.get('poly_down_odds_cents'), '.1f')}c (payout x{fv(ctx.get('poly_down_payout'), '.2f')})
REGLA_EV: EV = (tu_confianza/100) * payout. Si EV<1.0, la apuesta destruye valor. Ajusta confianza considerando si el precio ya refleja la direccion.

[NOTICIAS]
{news}

[HISTORIAL]
{acc_text}
Trust: {self.trust_score:.0f}/100 | Sesion: #{self.sessions_count}
{f'Patrones: {self.db_patterns}' if self.db_patterns else ''}
{f'Similares: {similar_text}' if similar_text else ''}
{f'Sesgos: {corrections}' if corrections else ''}
{f'Lecciones: {lessons}' if lessons else ''}
{f'⚠️ REFLEXION_RECIENTE: {reflection}' if reflection else ''}

{hmm_block}

[ORDER FLOW / CVD]
CVD_imbalance: {fv(ctx.get('cvd_imbalance_pct'), '.1f')}% (>0=compras agresivas dominan, <0=ventas)
CVD_net: ${ctx.get('cvd_net', 0):,.0f}
CVD_aggressor_ratio: {fv(ctx.get('cvd_aggressor_ratio'), '.3f')} (>0.55=buyers dominate)

{f"[GAMMA EXPOSURE]" if ctx.get('gex_net_bias') is not None else ""}
{f"GEX_net_bias: {ctx.get('gex_net_bias', 'N/A')}" if ctx.get('gex_net_bias') is not None else ""}
{f"Call_Wall: ${ctx.get('gex_call_wall', 0):,.0f}" if ctx.get('gex_call_wall') else ""}
{f"Put_Wall: ${ctx.get('gex_put_wall', 0):,.0f}" if ctx.get('gex_put_wall') else ""}

CALCULAR: Suma pesos UP vs DOWN segun la tabla del system prompt. Responde SOLO el JSON."""

    def _fallback(self, price: PriceData, ta: dict) -> AISignal:
        score = 50
        reasons = []

        # Momentum and trend are the strongest short-term signals
        if ta["momentum"] > 0.01:
            score += 8; reasons.append(f"Mom+ ({ta['momentum']:+.4f}%)")
        elif ta["momentum"] < -0.01:
            score -= 8; reasons.append(f"Mom- ({ta['momentum']:+.4f}%)")
        if ta["tick_buy_pressure"] > 60:
            score += 7; reasons.append(f"presion compra {ta['tick_buy_pressure']}%")
        elif ta["tick_buy_pressure"] < 40:
            score -= 7; reasons.append(f"presion venta {ta['tick_buy_pressure']}%")
        if ta["macd_histogram"] > 0:
            score += 6; reasons.append(f"MACD+ ({ta['macd_histogram']:.2f})")
        elif ta["macd_histogram"] < 0:
            score -= 6; reasons.append(f"MACD- ({ta['macd_histogram']:.2f})")
        if ta["ema_cross"] == "BULLISH":
            score += 6; reasons.append("EMA12>EMA26")
        elif ta["ema_cross"] == "BEARISH":
            score -= 6; reasons.append("EMA12<EMA26")
        if ta["sma_cross"] == "BULLISH":
            score += 4; reasons.append("SMA bullish")
        elif ta["sma_cross"] == "BEARISH":
            score -= 4; reasons.append("SMA bearish")
        # RSI extremes suggest reversal only at true extremes
        if ta["rsi"] > 80:
            score -= 5; reasons.append(f"RSI sobrecompra extrema ({ta['rsi']})")
        elif ta["rsi"] > 60:
            score += 4; reasons.append(f"RSI alcista ({ta['rsi']})")
        elif ta["rsi"] < 20:
            score += 5; reasons.append(f"RSI sobreventa extrema ({ta['rsi']})")
        elif ta["rsi"] < 40:
            score -= 4; reasons.append(f"RSI bajista ({ta['rsi']})")
        if ta["bb_pct"] < 15:
            score += 4; reasons.append(f"BB bajo ({ta['bb_pct']:.0f}%) rebote")
        elif ta["bb_pct"] > 85:
            score -= 4; reasons.append(f"BB alto ({ta['bb_pct']:.0f}%) reversal")
        if ta["price_vs_support_pct"] < 0.03:
            score += 3; reasons.append("muy cerca soporte")
        if ta["price_vs_resistance_pct"] < 0.03:
            score -= 3; reasons.append("muy cerca resistencia")

        # Tie-breaker: if score == 50, nudge by momentum sign
        if score == 50:
            score = 51 if ta["momentum"] >= 0 else 49
        raw_conf = score if score > 50 else 100 - score
        confidence = max(51, min(85, raw_conf))
        prediction = "UP" if score > 50 else "DOWN"
        risk = "HIGH" if ta["volatility"] > 0.1 else \
               "MEDIUM" if ta["volatility"] > 0.05 else "LOW"
        return AISignal(
            prediction=prediction, confidence=confidence,
            reasoning="; ".join(reasons) or "Analisis tecnico local",
            news_summary="Modo sin API — solo tecnicos",
            risk_score=risk,
            suggested_bet_pct=0.02 if confidence > 60 else 0.01,
            timestamp=time.time(),
            original_confidence=confidence,
        )
