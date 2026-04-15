from ..models import AISignal
from ..config import log
from ..config_manager import rules
from ..utils import trust_bet_multiplier

from .. import config as cfg
from .. import database as db

# ── Optional RL Portfolio Allocator ──────────────────────────
# Loaded lazily once at import. If stable-baselines3 / the trained .zip are
# missing, `get_rl_agent()` returns a detector with `available=False` and
# `calculate_agent_bet` transparently falls back to the classic Kelly path.
try:
    from .rl_wrapper import get_rl_agent, build_observation, action_to_bet
    _RL_WRAPPER_OK = True
except Exception as _rl_err:  # pragma: no cover — safety net
    log.info(f"[RISK] RL wrapper unavailable: {_rl_err} — classic Kelly only")
    get_rl_agent = None       # type: ignore
    build_observation = None  # type: ignore
    action_to_bet = None      # type: ignore
    _RL_WRAPPER_OK = False


class RiskManager:
    """Kelly-inspired bet sizing for swarm agents with $100 sub-balances.

    When an `rl_context` dict is passed to ``calculate_agent_bet`` AND the
    Stable-Baselines3 policy at ``models/rl_risk_agent.zip`` is loaded, the
    sizing decision is delegated to the RL Portfolio Allocator instead of
    the Kelly / confidence-linear formula. The classic path is kept intact
    for retrocompatibility: any caller that does not supply ``rl_context``
    keeps the exact behaviour it had before.
    """

    @staticmethod
    def _risk_val(key: str, fallback):
        """Read a risk parameter from dynamic_rules.json, with fallback."""
        return rules.get("risk", key, fallback)

    # ── Classic Kelly path (unchanged, extracted into a private helper) ──
    @staticmethod
    def _classic_agent_bet(signal: AISignal, agent_balance: float,
                           odds_cents: int, trust_score: float,
                           agent_id: str,
                           ) -> tuple[bool, float, str]:
        return RiskManager._calculate_agent_bet_classic(
            signal, agent_balance, odds_cents, trust_score, agent_id,
        )

    @staticmethod
    def calculate_agent_bet(signal: AISignal, agent_balance: float,
                            odds_cents: int = 50,
                            trust_score: float = 50.0,
                            agent_id: str = "deepseek_smc",
                            rl_context: dict | None = None,
                            ) -> tuple[bool, float, str]:
        """Bet sizing dispatcher.

        If `rl_context` is provided AND the RL policy is loaded → route to
        ``_calculate_agent_bet_rl``. Otherwise → classic Kelly / confidence
        sizing (unchanged). See ``rl_wrapper.build_observation`` for the
        expected keys inside ``rl_context``.
        """
        # ── RL path (only if we have both the context and the loaded policy) ──
        if rl_context and _RL_WRAPPER_OK:
            try:
                rl_agent = get_rl_agent()
            except Exception as e:
                log.warning(f"[RISK] get_rl_agent failed: {e}")
                rl_agent = None
            if rl_agent is not None and rl_agent.available:
                try:
                    return RiskManager._calculate_agent_bet_rl(
                        signal=signal,
                        agent_balance=agent_balance,
                        odds_cents=odds_cents,
                        trust_score=trust_score,
                        agent_id=agent_id,
                        rl_context=rl_context,
                        rl_agent=rl_agent,
                    )
                except Exception as e:
                    log.warning(f"[RISK] RL sizing failed, falling back to "
                                f"Kelly: {e}")

        # ── Classic Kelly / confidence path (default / fallback) ──
        return RiskManager._calculate_agent_bet_classic(
            signal, agent_balance, odds_cents, trust_score, agent_id,
        )

    # ── RL path ──────────────────────────────────────────────
    @staticmethod
    def _calculate_agent_bet_rl(signal: AISignal, agent_balance: float,
                                odds_cents: int, trust_score: float,
                                agent_id: str, rl_context: dict,
                                rl_agent,
                                ) -> tuple[bool, float, str]:
        """RL Portfolio Allocator path.

        Safety rails that stay **hard-coded** regardless of what the policy
        outputs (RL cannot override these — they live in the env and here):
            - daily loss limit
            - drawdown halt
            - min balance floor
            - low-quality signal cap
            - minimum confidence gate
        The RL agent only decides the *sizing* inside the remaining budget.
        """
        MIN_AGENT_BET = 1.0
        # Micro-balance mode: allow the account to bleed to ~$1 before
        # aborting, so the RL/Judge dataset captures the full tail of
        # adverse-regime outcomes under $10 stress tests.
        MIN_AGENT_BALANCE = 1.0

        if agent_balance < MIN_AGENT_BALANCE:
            return False, 0, f"Balance agente insuficiente (${agent_balance:.2f})"

        min_conf = RiskManager._risk_val("min_confidence", 45)
        if signal.confidence < min_conf:
            return False, 0, f"Confianza baja ({signal.confidence}% < {min_conf}%)"

        # Reuse classic circuit breakers
        daily_loss_limit = RiskManager._risk_val("daily_loss_limit_pct", 0.10)
        max_dd_halt = RiskManager._risk_val("max_drawdown_halt_pct", 10)

        try:
            stats = db.get_agent_daily_stats(agent_id)
        except Exception:
            stats = {"daily_pnl": 0, "daily_bets": 0, "losing_streak": 0}

        daily_loss_cap = agent_balance * daily_loss_limit
        if stats["daily_pnl"] < 0 and abs(stats["daily_pnl"]) >= daily_loss_cap:
            return (False, 0,
                    f"Daily loss limit alcanzado: P&L=${stats['daily_pnl']:+.2f} "
                    f">= cap ${daily_loss_cap:.2f} ({daily_loss_limit:.0%})")

        try:
            port = db.get_agent_portfolio(agent_id)
            peak = port["peak_balance"] if port else agent_balance
        except Exception:
            peak = agent_balance
        # Live-mode guard: if peak is stale from paper history, cap it.
        # Prevents phantom drawdown when switching to live trading.
        if not getattr(cfg, "PAPER_TRADING_MODE", True) and peak > agent_balance:
            log.debug(f"[RISK] Live mode: capping stale peak ${peak:.2f} "
                      f"to live balance ${agent_balance:.2f}")
            peak = agent_balance
        drawdown_pct = ((peak - agent_balance) / peak * 100) if peak > 0 else 0
        if drawdown_pct >= max_dd_halt:
            return (False, 0,
                    f"Drawdown halt: {drawdown_pct:.1f}% >= {max_dd_halt}% "
                    f"(peak=${peak:.2f}, bal=${agent_balance:.2f})")

        # ── Build the 13-dim observation vector ──
        obs = build_observation(
            judge_prob=float(rl_context.get("judge_prob",
                                            signal.confidence / 100.0)),
            hmm=rl_context.get("hmm"),                              # dict or None
            balance=agent_balance,
            initial_balance=float(rl_context.get("initial_balance", 100.0)),
            last_pnls=list(rl_context.get("last_pnls", [])),
            atr_pct=float(rl_context.get("atr_pct", 0.03)),
            odds_cents=odds_cents,
        )
        action = rl_agent.predict(obs)
        if action is None:
            # Policy refused or malfunctioned — fall back to classic
            log.info(f"[RL] {agent_id}: policy returned None, "
                     "falling back to Kelly classic")
            return RiskManager._calculate_agent_bet_classic(
                signal, agent_balance, odds_cents, trust_score, agent_id,
            )

        amount, label = action_to_bet(action, agent_balance)

        # RL chose SKIP
        if amount <= 0:
            return (False, 0,
                    f"RL skip (action={action:.3f}, "
                    f"judge={rl_context.get('judge_prob', 'N/A')}, "
                    f"conf={signal.confidence}%) → ghost")

        # LOW_QUALITY signal still gets a hard cap even on the RL path
        quality = getattr(signal, "signal_quality", "NORMAL")
        if quality == "LOW_QUALITY":
            low_cap = RiskManager._risk_val("low_quality_cap_pct", 0.001)
            lq_max = max(MIN_AGENT_BET, agent_balance * low_cap)
            if amount > lq_max:
                amount = round(lq_max, 2)
                label += f" [LQ cap ${lq_max:.2f}]"

        # Final clamp — cannot exceed available balance
        amount = max(MIN_AGENT_BET, min(amount, agent_balance))
        amount = round(amount, 2)

        pct = (amount / agent_balance * 100) if agent_balance > 0 else 0
        reason = (f"{label} → ${amount:.2f} "
                  f"({pct:.1f}% of ${agent_balance:.0f}, "
                  f"conf={signal.confidence}%, algo={rl_agent.algo})")
        return True, amount, reason

    # ── Classic Kelly path (moved into its own method, logic unchanged) ──
    @staticmethod
    def _calculate_agent_bet_classic(signal: AISignal, agent_balance: float,
                                     odds_cents: int = 50,
                                     trust_score: float = 50.0,
                                     agent_id: str = "deepseek_smc",
                                     ) -> tuple[bool, float, str]:
        """Bet sizing for swarm agents with $100 sub-balances.

        Returns (should_bet, amount, reason).
        Applies: trust multiplier, balance-proportional scaling,
        daily loss limit, drawdown halt, and losing streak reduction.
        """
        MIN_AGENT_BET = 1.0
        MAX_AGENT_BET_HARD = 5.0
        # Micro-balance mode: allow the account to bleed to ~$1 before
        # aborting, so the RL/Judge dataset captures the full tail of
        # adverse-regime outcomes under $10 stress tests.
        MIN_AGENT_BALANCE = 1.0

        if agent_balance < MIN_AGENT_BALANCE:
            return False, 0, f"Balance agente insuficiente (${agent_balance:.2f})"

        min_conf = RiskManager._risk_val("min_confidence", 45)
        if signal.confidence < min_conf:
            return False, 0, f"Confianza baja ({signal.confidence}% < {min_conf}%)"

        # ── Circuit breakers from dynamic_rules.json ──────────
        daily_loss_limit = RiskManager._risk_val("daily_loss_limit_pct", 0.10)
        max_dd_halt = RiskManager._risk_val("max_drawdown_halt_pct", 10)
        max_dd_reduce = RiskManager._risk_val("max_drawdown_reduce_pct", 5)
        streak_threshold = RiskManager._risk_val("losing_streak_threshold", 3)
        streak_reduction = RiskManager._risk_val("losing_streak_reduction", 0.5)

        # Query daily P&L and streak from DB
        try:
            stats = db.get_agent_daily_stats(agent_id)
        except Exception:
            stats = {"daily_pnl": 0, "daily_bets": 0, "losing_streak": 0}

        # 1) Daily loss limit: block if today's losses exceed % of balance
        daily_loss_cap = agent_balance * daily_loss_limit
        if stats["daily_pnl"] < 0 and abs(stats["daily_pnl"]) >= daily_loss_cap:
            return (False, 0,
                    f"Daily loss limit alcanzado: P&L=${stats['daily_pnl']:+.2f} "
                    f">= cap ${daily_loss_cap:.2f} ({daily_loss_limit:.0%})")

        # 2) Drawdown halt: block if balance dropped too far from peak
        try:
            port = db.get_agent_portfolio(agent_id)
            peak = port["peak_balance"] if port else agent_balance
        except Exception:
            peak = agent_balance
        # Live-mode guard: if peak is stale from paper history, cap it.
        # Prevents phantom drawdown when switching to live trading.
        if not getattr(cfg, "PAPER_TRADING_MODE", True) and peak > agent_balance:
            log.debug(f"[RISK] Live mode: capping stale peak ${peak:.2f} "
                      f"to live balance ${agent_balance:.2f}")
            peak = agent_balance
        drawdown_pct = ((peak - agent_balance) / peak * 100) if peak > 0 else 0
        if drawdown_pct >= max_dd_halt:
            return (False, 0,
                    f"Drawdown halt: {drawdown_pct:.1f}% >= {max_dd_halt}% "
                    f"(peak=${peak:.2f}, bal=${agent_balance:.2f})")

        # ── Base bet from confidence mapping ──────────────────
        # Balance-proportional max: never risk more than 5% of current balance
        balance_cap_pct = RiskManager._risk_val("max_bet_pct", 0.05)
        max_bet = min(MAX_AGENT_BET_HARD, agent_balance * balance_cap_pct)
        max_bet = max(MIN_AGENT_BET, max_bet)  # floor at $1

        # Linear confidence-to-bet: 45%->$1, 90%->max_bet
        conf_norm = max(0, min(1, (signal.confidence - min_conf) / 45))
        amount = MIN_AGENT_BET + conf_norm * (max_bet - MIN_AGENT_BET)

        # ── Modifiers ─────────────────────────────────────────
        modifiers = []

        # LOW_QUALITY signal cap
        quality = getattr(signal, "signal_quality", "NORMAL")
        if quality == "LOW_QUALITY":
            low_cap = RiskManager._risk_val("low_quality_cap_pct", 0.001)
            lq_max = max(MIN_AGENT_BET, agent_balance * low_cap)
            amount = min(amount, lq_max)
            modifiers.append("LQ")

        # Trust multiplier
        trust_mult = trust_bet_multiplier(trust_score)
        if trust_mult != 1.0:
            amount *= trust_mult
            modifiers.append(f"trust={trust_mult:.1f}x")

        # Drawdown reduction zone (soft): reduce sizing if in drawdown
        if drawdown_pct >= max_dd_reduce:
            dd_factor = 0.5
            amount *= dd_factor
            modifiers.append(f"DD={drawdown_pct:.0f}%->0.5x")

        # Losing streak reduction
        if stats["losing_streak"] >= streak_threshold:
            amount *= streak_reduction
            modifiers.append(f"streak={stats['losing_streak']}->0.5x")

        # ── STRICT CLAMP: $1 min, dynamic max, never exceed balance ──
        amount = max(MIN_AGENT_BET, min(amount, max_bet, agent_balance))
        amount = round(amount, 2)

        pct = (amount / agent_balance * 100) if agent_balance > 0 else 0
        mod_str = f" [{','.join(modifiers)}]" if modifiers else ""
        reason = (f"AgentBet=${amount:.2f} "
                  f"({pct:.1f}% of ${agent_balance:.0f}, "
                  f"conf={signal.confidence}%, trust={trust_score:.0f}){mod_str}")
        return True, amount, reason
