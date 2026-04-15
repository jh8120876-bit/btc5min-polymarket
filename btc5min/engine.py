import atexit
import time
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

from . import config as cfg
from .config import WINDOW_MINUTES, PAPER_TRADING_MODE, log
from .data_feeds.polymarket import PolymarketClient, calc_polymarket_pnl
from .observability.prometheus_exporter import metrics as prom_metrics

# ── ML Judge (XGBoost filter) ───────────────────────────────
ML_CONFIDENCE_THRESHOLD = 0.51
_JUDGE_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "ml_judge_model.pkl"
_JUDGE_FEATURES = [
    "confidence", "rsi", "bb_pct", "volatility",
    "momentum", "atr_pct", "hour_utc", "price_change_pct",
    "trend_alignment", "funding_rate", "order_book_imbalance",
    "liq_buy_vol_5m", "liq_sell_vol_5m", "reversal_alert",
    "smc_bos_last", "smc_choch_last", "smc_in_kill_zone",
    "smc_ob_distance_pct", "bid_ask_spread",
    # Liquidation cluster proximity (Vibe-Trading inspired).
    # Encoded side: 1=ABOVE (bullish sweep target), -1=BELOW (bearish),
    # 0=AT_SPOT/NEUTRAL. See _evaluate_oracle() for encoding.
    "liq_cluster_distance_atr", "liq_cluster_magnitude_pct",
    "liq_cluster_side_enc",
]
_judge_model = None
_judge_model_features: list[str] | None = None  # actual features the model was trained on

def _load_judge():
    """Load the XGBoost Judge model if available. Called once at import."""
    global _judge_model, _judge_model_features
    if _JUDGE_MODEL_PATH.exists():
        try:
            import joblib
            _judge_model = joblib.load(_JUDGE_MODEL_PATH)
            # Auto-detect which features the model was actually trained on
            model_fnames = getattr(_judge_model, "feature_names_in_", None)
            if model_fnames is not None:
                _judge_model_features = list(model_fnames)
            else:
                # XGBoost stores feature names differently
                booster_fnames = getattr(
                    getattr(_judge_model, "get_booster", lambda: None)(),
                    "feature_names", None,
                )
                if booster_fnames:
                    _judge_model_features = list(booster_fnames)
                else:
                    _judge_model_features = _JUDGE_FEATURES
            log.info(f"[JUDGE] XGBoost model loaded: {_JUDGE_MODEL_PATH} "
                     f"({len(_judge_model_features)} features: "
                     f"{_judge_model_features})")
        except Exception as e:
            log.warning(f"[JUDGE] Failed to load model: {e}")
            _judge_model = None
            _judge_model_features = None
    else:
        log.info("[JUDGE] Model not found — Judge disabled "
                 f"(expected at {_JUDGE_MODEL_PATH})")

_load_judge()
from .models import WindowState, Bet, BetResult, AISignal
from .data_feeds.price_feed import ChainlinkPriceFeed
from .analysis.technical import TechnicalAnalysis
from .ai.ai_engine import AIEngine
from .risk.risk import RiskManager
from .ai.local_llm import LocalLLM
from .analysis.feature_engineering import FeatureEngineer
from .data_feeds.binance_data import BinanceMarketData
from .data_feeds.options_data import DeribitGEXProvider
from . import database as db
from .config_manager import rules
from .ai import swarm
from . import sentiment
from .ai.rag_db import StrategyMemory
from .analysis import smc_features
from .analysis.regime_hmm import get_regime_detector, regime_rag_hint

def get_primary_agent_id() -> str:
    """Dynamic lookup — always returns the current primary agent_id."""
    return swarm.get_primary_agent_id()

# Global semaphore: max 2 concurrent remote API calls at any time
# Prevents 429 rate-limit bans when Swarm + main prediction overlap
_primary_semaphore = threading.Semaphore(2)

# ── Fatigue rotation labels (fallback if dynamic_rules.json has none) ──
_FATIGUE_ROTATION_LABELS = [
    "Reversión Extrema",
    "Contrarian Total",
    "Momentum Puro",
]


class PredictionEngine:
    """Main coordinator: windows, predictions, bets, resolution."""

    def __init__(self):
        self.price_feed = ChainlinkPriceFeed()
        self.ta = TechnicalAnalysis()
        self.ai = AIEngine()
        self.risk = RiskManager()
        self.local_llm = LocalLLM()
        self.features = FeatureEngineer()
        self.binance = BinanceMarketData()
        self.polymarket = PolymarketClient()
        # ── Web3 Execution Layer (None if offline_sim) ──
        self.executor = None  # initialized in start() after DB init
        self.survival = None  # SurvivalMonitor, initialized in start()
        self.gex = DeribitGEXProvider()
        # Tool broker for ReAct second-opinion (Vibe-Trading inspired).
        # Keys = tool name, values = callables returning JSON-serializable dicts.
        self.ai.tool_broker = {
            "get_cvd": lambda minutes=1: self.binance.get_cvd_summary(minutes=int(minutes)),
            "get_liquidations": lambda minutes=5: self.binance.get_liquidation_summary(minutes=int(minutes)),
            "get_liq_cluster": lambda: self._current_liq_cluster or {},
            "get_gex": lambda: self.gex.get_gex() if hasattr(self.gex, "get_gex") else self.gex.get_status(),
            "get_smc": lambda: self._current_smc_features or {},
            "get_mss": lambda: getattr(self, "_current_mss_features", {}) or {},
        }
        self.rag = StrategyMemory()
        self.window = WindowState()
        self._recent_results: list[BetResult] = []
        self.current_bet: Optional[Bet] = None
        self.current_signal = None
        self.current_ta: dict = {}
        self.running = False
        # RLock because AI/prediction paths re-enter state accessors; keeps
        # `_tick`-held sections safe from nested state reads.
        self._lock = threading.RLock()
        self._last_mid_update: float = 0
        self._last_price_save: float = 0
        self._last_ta_update: float = 0
        self._last_tick_cleanup_date: str = ""
        self._bet_evaluation_done: bool = False
        # ── Sniper consensus state ──
        self._window_evaluations: list[dict] = []   # [{prediction, confidence, timestamp}]
        self._window_bet_placed: bool = False
        self._last_eval_time: float = 0
        self._is_ghost_trade: bool = False
        self._ptb_sizing_factor: float = 1.0
        self._ghost_reason: str = ""
        # Score-based consensus tracking
        self._consensus_score: int = 0
        self._consensus_direction: str | None = None
        self._consensus_idx: int = -1        # index in _window_evaluations that crossed the fire threshold
        self._last_trigger_type: str | None = None  # CONSENSUS | EARLY-FLASH | FLASH-FIRE | RESCUE | OVERRIDE
        # ── HFT microstructure tracking (per-window, reset on window roll) ──
        self._last_sniper_ask_cents: float | None = None
        self._last_sniper_prob: float | None = None
        self._last_sniper_price: float | None = None
        self._last_sniper_price_ts: float = 0.0
        self._sniper_trigger_ready_ts: float = 0.0  # stamp when trigger condition met
        # ── Second Opinion (LLM tactical consult) ──
        # Cached payload: {direction, confidence, veto, reasoning,
        #                  arrived_at, source, latency_ms, window_id}
        self._second_opinion: dict | None = None
        self._second_opinion_in_flight: bool = False
        self._second_opinion_launched_at: float = 0.0
        # ── ML Oracle state ──
        self._ml_oracle_prob: float | None = None
        self._ml_skipped: bool = False
        # ── Async prediction thread guard ──
        self._prediction_in_flight: bool = False
        # ── Multi-Agent Swarm state ──
        self._swarm_signals: dict[str, dict] = {}  # {agent_id: {prediction, confidence, ...}}
        # ── RAG strategy for current window ──
        self._current_rag_strategy: str = ""
        self._current_smc_features: dict = {}
        self._current_liq_cluster: dict = {}
        # ── Watchful Hold: pending soft-veto re-evaluation state ──
        # {side, reason, odds, odds_source, is_early, poly_quote,
        #  retries, armed_at, consecutive_ok}
        self._pending_retry: dict | None = None
        self._current_mss_features: dict = {}
        # ── HMM Regime Detector (unsupervised Markov state) ──
        self.regime = get_regime_detector()
        self._current_regime: dict | None = None   # last decoded snapshot
        self._last_regime_decode_ts: float = 0.0
        self._REGIME_DECODE_INTERVAL = 30.0        # mid-window refresh cadence (s)
        # ── Primary AI Circuit Breaker ──
        self._primary_halted: bool = False
        self._primary_halt_reason: str = ""
        self._primary_halt_time: float = 0
        # ── Live Observer (15s commentary) ──
        self._live_suggestion: str = ""
        self._live_suggestion_ts: float = 0
        # ── Fatigue lock: protects strategy + agent fatigue dicts across threads ──
        self._fatigue_lock = threading.Lock()
        # ── Strategy Fatigue / Rotation (RAG) ──
        self._strategy_losses: dict[str, int] = {}       # consecutive losses per strategy
        self._strategy_start_time: dict[str, float] = {} # first-use timestamp per strategy
        self._strategy_blacklist: dict[str, float] = {}  # {strategy: expiry_timestamp}
        self._STRATEGY_MAX_LOSSES = 7
        self._STRATEGY_MAX_AGE = 7200    # 2 hours
        self._STRATEGY_COOLDOWN = 7200   # 2 hours blacklist
        # ── Agent Fatigue / Rotation (Swarm + Local LLM) ──
        # {agent_id: {"losses": int, "start_time": float, "fatigued": bool, "rotation_idx": int}}
        self._agent_fatigue: dict[str, dict] = {}
        # ── get_state() cache (1s TTL) — prevents redundant DB/TA work ──
        self._state_cache: dict | None = None
        self._state_cache_ts: float = 0
        self._STATE_CACHE_TTL = 1.0

        # ── Startup recovery: resolve orphaned bets from prior sessions ──
        try:
            resolved = db.resolve_orphaned_bets()
            cancelled = db.cancel_stale_bets(max_age_seconds=600)
            if resolved or cancelled:
                log.info(f"[STARTUP] Recovery: {resolved} orphaned resolved, "
                         f"{cancelled} stale cancelled")
        except Exception as e:
            log.warning(f"[STARTUP] Orphan recovery failed (safe-fail): {e}")

    @property
    def _AGENT_MAX_LOSSES(self) -> int:
        fp = rules.get_section("fatigue_prompts")
        return int(fp.get("max_losses", 7)) if fp else 7

    @property
    def _AGENT_MAX_AGE(self) -> float:
        fp = rules.get_section("fatigue_prompts")
        return float(fp.get("max_age_sec", 7200)) if fp else 7200.0

    @property
    def _AGENT_FATIGUE_COOLDOWN(self) -> float:
        fp = rules.get_section("fatigue_prompts")
        return float(fp.get("cooldown_sec", 3600)) if fp else 3600.0

    @property
    def _EVAL_INTERVAL(self) -> int:
        return rules.get("sniper", "eval_interval_sec", 15)

    @property
    def _EVAL_CUTOFF(self) -> float:
        return rules.get("sniper", "eval_cutoff_sec", 235.0)

    @property
    def _FIRST_EVAL_DELAY(self) -> float:
        return rules.get("sniper", "first_eval_delay_sec", 25.0)

    # ── Score-based consensus tunables ──
    @property
    def _MIN_CONSENSUS_AGE(self) -> float:
        return rules.get("sniper", "min_consensus_age_sec", 20.0)

    @property
    def _CONSENSUS_SCORE_FIRE(self) -> int:
        return int(rules.get("sniper", "consensus_score_fire", 2))

    @property
    def _CONSENSUS_SCORE_FLIP(self) -> int:
        return int(rules.get("sniper", "consensus_score_flip", -2))

    @property
    def _CONFIDENCE_DECAY_PER_MIN(self) -> float:
        return float(rules.get("sniper", "confidence_decay_per_min", 2.0))

    @property
    def _DECAY_FLOOR_CONF(self) -> int:
        return int(rules.get("sniper", "decay_floor_conf", 51))

    @property
    def _EV_EARLY_FIRE(self) -> float:
        """Early Flash Fire threshold — dispara desde el eval #1 si EV > X."""
        return float(rules.get("sniper", "ev_early_fire_threshold", 1.30))

    @property
    def _LAST_MIN_RESCUE_CONF(self) -> int:
        return int(rules.get("sniper", "last_minute_rescue_conf", 60))

    @property
    def _LAST_MIN_RESCUE_EV(self) -> float:
        return float(rules.get("sniper", "last_minute_rescue_ev", 1.05))

    # ── HFT tunables (adaptive pacing & microstructure triggers) ──
    @property
    def _ADAPTIVE_INTERVAL_ENABLED(self) -> bool:
        return bool(rules.get("sniper", "adaptive_interval_enabled", True))

    @property
    def _ADAPTIVE_FAST_SEC(self) -> float:
        return float(rules.get("sniper", "adaptive_interval_fast_sec", 5))

    @property
    def _ADAPTIVE_SLOW_SEC(self) -> float:
        return float(rules.get("sniper", "adaptive_interval_slow_sec", 15))

    @property
    def _VELOCITY_FAST_USD_PER_SEC(self) -> float:
        return float(rules.get("sniper", "velocity_fast_usd_per_sec", 1.5))

    @property
    def _MIN_CONSENSUS_AGE_HOT(self) -> float:
        return float(rules.get("sniper", "min_consensus_age_hot_sec", 15.0))

    @property
    def _MIN_CONSENSUS_AGE_EV_THRESHOLD(self) -> float:
        return float(rules.get("sniper", "min_consensus_age_ev_threshold", 1.25))

    @property
    def _CVD_SHOCK_ENABLED(self) -> bool:
        return bool(rules.get("sniper", "cvd_shock_enabled", True))

    @property
    def _CVD_SHOCK_IMBALANCE_PCT(self) -> float:
        return float(rules.get("sniper", "cvd_shock_imbalance_pct", 70.0))

    @property
    def _MICROPRICE_FLASH_ENABLED(self) -> bool:
        return bool(rules.get("sniper", "microprice_flash_enabled", True))

    @property
    def _MICROPRICE_FLASH_DELTA_CENTS(self) -> float:
        return float(rules.get("sniper", "microprice_flash_delta_cents", 3.0))

    @property
    def _DECAY_MODE(self) -> str:
        return str(rules.get("sniper", "decay_mode", "quadratic")).lower()

    @property
    def _SLIPPAGE_BAIL_ENABLED(self) -> bool:
        return bool(rules.get("sniper", "slippage_bail_enabled", True))

    @property
    def _SLIPPAGE_BAIL_CENTS(self) -> float:
        return float(rules.get("sniper", "slippage_bail_cents", 2.0))

    @property
    def _REGIME_AWARE_FIRE(self) -> bool:
        return bool(rules.get("sniper", "regime_aware_fire", True))

    @property
    def _REGIME_SHORT_GAMMA_FIRE(self) -> int:
        return int(rules.get("sniper", "regime_short_gamma_fire_score", 1))

    @property
    def _REGIME_LONG_GAMMA_FIRE(self) -> int:
        return int(rules.get("sniper", "regime_long_gamma_fire_score", 3))

    # ── Second Opinion tunables ─────────────────────────────
    @property
    def _SECOND_OPINION_ENABLED(self) -> bool:
        return bool(rules.get("second_opinion", "enabled", True))

    @property
    def _SECOND_OPINION_TTL(self) -> float:
        return float(rules.get("second_opinion", "cache_ttl_sec", 60.0))

    @property
    def _VARIANT_D_LAUNCH_SEC(self) -> float:
        return float(rules.get("second_opinion", "variant_d_launch_sec", 120.0))

    @property
    def _VARIANT_D_ENABLED(self) -> bool:
        return (self._SECOND_OPINION_ENABLED
                and bool(rules.get("second_opinion", "variant_d_enabled", True)))

    @property
    def _VARIANT_B_ENABLED(self) -> bool:
        return (self._SECOND_OPINION_ENABLED
                and bool(rules.get("second_opinion", "variant_b_enabled", True)))

    @property
    def _VARIANT_C_ENABLED(self) -> bool:
        return (self._SECOND_OPINION_ENABLED
                and bool(rules.get("second_opinion", "variant_c_enabled", True)))

    @property
    def _VARIANT_A_ENABLED(self) -> bool:
        return (self._SECOND_OPINION_ENABLED
                and bool(rules.get("second_opinion", "variant_a_enabled", True)))

    # ── Super Sniper: EV-aware triggers ──
    @property
    def _EV_FLASH_FIRE(self) -> float:
        """Edge brutal: EV > este umbral dispara Flash Consensus."""
        return rules.get("sniper", "ev_flash_fire_threshold", 1.18)

    @property
    def _EV_GHOST_FLOOR(self) -> float:
        """Si EV cae por debajo de este piso tras tiempo mínimo → ghost."""
        return rules.get("sniper", "ev_ghost_floor", 1.00)

    @property
    def _EV_GHOST_AFTER_SEC(self) -> float:
        """Segundos mínimos antes de declarar EV Ghosting (minuto 2.5)."""
        return rules.get("sniper", "ev_ghost_after_sec", 150.0)

    # ── Live Dynamic Sizing helpers ───────────────────────────

    def _default_initial_balance(self) -> float:
        """Return the active "paper" default balance.

        Precedence:
            1. dynamic_rules.json  →  defaults.initial_balance  (hot-reloadable)
            2. config.DEFAULT_INITIAL_BALANCE                   (static)
            3. 25.0                                              (last resort)
        """
        try:
            v = rules.get("defaults", "initial_balance",
                          getattr(cfg, "DEFAULT_INITIAL_BALANCE", 25.0))
            return float(v)
        except Exception:
            return float(getattr(cfg, "DEFAULT_INITIAL_BALANCE", 25.0))

    def _resolve_agent_balance(self, agent_id: str) -> float:
        """Resolve the spendable balance for ``agent_id``.

        If the ``LIVE_DYNAMIC_SIZING`` toggle is ON **and** the primary
        agent is the caller **and** the CLOB executor is connected,
        Kelly ingests the real on-chain USDC.e balance. Otherwise it
        falls back to the simulated paper balance from agent_portfolios,
        and finally to the configured default (~$25).
        """
        live_toggle = bool(getattr(cfg, "LIVE_DYNAMIC_SIZING", False))
        default_paper = self._default_initial_balance()

        if (live_toggle
                and self.executor is not None
                and getattr(self.executor, "connected", False)
                and agent_id == get_primary_agent_id()):
            
            # Use window-level caching to avoid 1-second polling spam.
            current_wid = getattr(self.window, "window_id", 0) if getattr(self, "window", None) else 0
            cache = getattr(self, "_live_balance_cache", {})
            if agent_id in cache and cache[agent_id].get("window_id") == current_wid:
                return cache[agent_id]["balance"]

            try:
                live_bal = self.executor.get_balance_usdc()
                if live_bal is not None and live_bal > 0:
                    log.info(f"[Live Sizing] Kelly ingesting REAL USDC "
                             f"balance for {agent_id}: ${live_bal:.2f}")
                    # Save to cache bound to this window
                    self._live_balance_cache = {agent_id: {"window_id": current_wid, "balance": float(live_bal)}}
                    return float(live_bal)
                log.info(f"[Live Sizing] Toggle ON but live probe returned "
                         f"{live_bal} — using paper balance fallback")
            except Exception as e:
                log.warning(f"[Live Sizing] Probe raised: {e} — paper fallback")

        try:
            port = db.get_agent_portfolio(agent_id)
            if port and port.get("balance") is not None:
                return float(port["balance"])
        except Exception as e:
            log.debug(f"[Live Sizing] portfolio lookup failed for "
                      f"{agent_id}: {e}")
        return default_paper

    def _calc_window(self) -> tuple[float, float]:
        now = datetime.now(timezone.utc)
        wm = (now.minute // WINDOW_MINUTES) * WINDOW_MINUTES
        start = now.replace(minute=wm, second=0, microsecond=0)
        end = start + timedelta(minutes=WINDOW_MINUTES)
        return start.timestamp(), end.timestamp()

    def _time_left(self) -> int:
        return max(0, int(self.window.end_time - time.time()))

    def start(self):
        # Share the rate-limit semaphore with swarm module
        swarm._api_semaphore = _primary_semaphore
        # Init DB
        db.init_db()
        # Start Prometheus metrics exporter (no-op if disabled)
        prom_metrics.start()
        # ── Initialize Web3 Executor (live_mainnet only — py-clob-client) ──
        try:
            from .execution.clob_executor import ClobExecutor
            self.executor = ClobExecutor.from_config()
        except Exception as e:
            log.warning(f"[EXEC] ClobExecutor init failed: {e} — offline_sim mode")
            self.executor = None
        # ── Initialize SurvivalMonitor (Profit-Lock / Flip-Stop) ──
        try:
            from .execution.survival import SurvivalMonitor
            self.survival = SurvivalMonitor(
                self.executor, self.polymarket.live,
            )
            self.survival.start()
        except Exception as e:
            log.warning(f"[SURVIVAL] Init failed: {e}")
            self.survival = None
        # ── Initialize Gasless Redeem Relayer (nightly cron) ──
        try:
            from .execution.relayer import RedeemRelayer
            self._redeem_relayer = RedeemRelayer(self.executor)
            self._redeem_relayer.start()
        except Exception as e:
            log.warning(f"[REDEEM] Init failed: {e}")
        # Restore window state + recent results from DB (survive restarts)
        self._restore_state()
        # Load historical data for AI context
        self.ai.load_history_from_db()
        # ── RAG auto-ingestion: refill ChromaDB if empty or migrated ──
        try:
            if self.rag.available and (self.rag.needs_reingest
                                       or self.rag.count() == 0):
                log.info("[RAG] Auto-ingestion triggered — collection empty "
                         "or model migration detected")
                from .ai.ingestar_rag import run_ingestion
                run_ingestion(store=self.rag)
                log.info(f"[RAG] Auto-ingestion complete — "
                         f"{self.rag.count()} strategies loaded")
        except Exception as e:
            log.warning(f"[RAG] Auto-ingestion failed (non-fatal): {e}")
        # Initialize agent portfolios in DB for active agents
        _default_bal = self._default_initial_balance()
        for agent in swarm.get_active_agents():
            db.upsert_agent_portfolio(
                agent["agent_id"],
                agent.get("display_name", agent["agent_id"]),
                agent.get("strategy", ""),
                agent.get("initial_balance", _default_bal),
            )
        # Hydrate agent fatigue state from DB (survives restarts)
        fatigue_states = db.load_all_fatigue_states()
        with self._fatigue_lock:
            for agent_id, state in fatigue_states.items():
                self._agent_fatigue[agent_id] = state
        if fatigue_states:
            fatigued = [a for a, s in fatigue_states.items() if s["fatigued"]]
            log.info(f"Restored fatigue state for {len(fatigue_states)} agents"
                     f"{f' ({len(fatigued)} fatigued)' if fatigued else ''}")
        self.price_feed.start()
        self.binance.start()
        self.gex.start()
        self.running = True
        self._main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self._observer_thread = threading.Thread(target=self._live_observer_loop, daemon=True)
        self._main_thread.start()
        self._observer_thread.start()
        atexit.register(self._graceful_shutdown)
        log.info("Engine started — waiting for Chainlink price data...")

    def _graceful_shutdown(self):
        """Flip running=False and briefly join worker threads so in-flight
        predictions/DB writes get a chance to complete before process exit."""
        if not getattr(self, "running", False):
            return
        log.info("Engine shutdown: signaling threads to stop...")
        self.running = False
        for t in (getattr(self, "_main_thread", None),
                  getattr(self, "_observer_thread", None)):
            if t is not None and t.is_alive():
                try:
                    t.join(timeout=3)
                except Exception:
                    pass
        try:
            if getattr(self, "survival", None) is not None:
                self.survival.stop()
        except Exception:
            pass

    def _restore_state(self):
        """Restore window state + recent results from DB so restarts don't reset."""
        # ── Restore window_id counter and open_time (prevent duplicates) ──
        try:
            last_win = db.get_last_window_state()
            last_wid = last_win["window_id"]
            last_open_time = last_win["open_time"]
            if last_wid > 0:
                self.window = WindowState(
                    window_id=last_wid,
                    open_time=last_open_time,
                )
                log.info(
                    f"Window state restored: last window_id={last_wid}, "
                    f"open_time={last_open_time}"
                )
        except Exception as e:
            log.error(f"Window state restore failed: {e}")

        # ── Restore recent results for dashboard display ──
        try:
            conn = db._get_conn()
            results = []
            
            # Use a robust JOIN to get the exact prediction metrics for the executed primary bet
            db_rows = conn.execute(
                """SELECT b.window_id, b.side, b.amount, b.odds_cents, b.open_price, 
                          b.close_price, b.outcome, b.payout, b.profit, b.won, b.created_at,
                          b.is_manual,
                          p.prediction, p.confidence, p.rag_strategy, p.ml_oracle_prob, p.ml_skipped
                   FROM bets b
                   LEFT JOIN predictions p ON b.window_id = p.window_id AND b.ai_model = p.ai_model
                   WHERE b.outcome IS NOT NULL AND b.is_shadow=0
                   ORDER BY b.window_id DESC LIMIT 100"""
            ).fetchall()
            
            for bet_row in db_rows:
                ts = "--:--"
                try:
                    raw_ts = bet_row["created_at"]
                    if raw_ts and len(raw_ts) >= 16:
                        ts = raw_ts[11:16]
                except Exception:
                    pass
                results.append(BetResult(
                    window_id=bet_row["window_id"],
                    side=bet_row["side"],
                    amount=bet_row["amount"],
                    price_cents=bet_row["odds_cents"],
                    open_price=bet_row["open_price"],
                    close_price=bet_row["close_price"] or 0,
                    outcome=bet_row["outcome"],
                    payout=bet_row["payout"] or 0,
                    profit=bet_row["profit"] or 0,
                    won=bool(bet_row["won"] or 0),
                    timestamp=ts,
                    ai_confidence=bet_row["confidence"] or 0,
                    ai_prediction=bet_row["prediction"] or "",
                    is_auto=not bool(bet_row["is_manual"] or 0),
                    rag_strategy=bet_row["rag_strategy"] or "",
                    ml_oracle_prob=bet_row["ml_oracle_prob"],
                    ml_skipped=bool(bet_row["ml_skipped"] or 0),
                ))
            results.reverse()
            self._recent_results = results[:100]
            log.info(f"Restored {len(self._recent_results)} recent results from DB")
        except Exception as e:
            log.error(f"Results restore failed: {e}")

    def _main_loop(self):
        while self.running:
            p = self.price_feed.get_price()
            if p and p.price > 0:
                log.info(f"First price received: ${p.price:,.2f} from {p.source}")
                break
            time.sleep(1)

        TICK_INTERVAL = 1.0  # seconds between ticks
        WINDOW_SECONDS = WINDOW_MINUTES * 60  # 300s

        while self.running:
            tick_start = time.time()
            try:
                self._tick()
            except Exception:
                log.exception("Tick error (full traceback)")

            # ── Absolute clock sync: wake up precisely at window boundaries ──
            now = time.time()
            tick_elapsed = now - tick_start
            secs_into_window = now % WINDOW_SECONDS
            secs_to_boundary = WINDOW_SECONDS - secs_into_window

            if secs_to_boundary <= TICK_INTERVAL + 0.5:
                # Next window boundary is imminent — sleep exactly until it
                sleep_time = max(0.05, secs_to_boundary + 0.01)
                log.debug(f"Clock sync: boundary in {secs_to_boundary:.1f}s, "
                          f"sleeping {sleep_time:.2f}s")
            else:
                # Normal tick — subtract elapsed time to stay on cadence
                sleep_time = max(0.1, TICK_INTERVAL - tick_elapsed)

            time.sleep(sleep_time)

    def _tick(self):
        need_prediction = False
        new_window_price = None
        new_window_id = 0

        # ── Phase 1: Quick state update (inside lock) ──────────
        with self._lock:
            now = time.time()
            ws, we = self._calc_window()

            # Save price tick to DB every ~10s
            if now - self._last_price_save > 10:
                p = self.price_feed.get_price()
                if p:
                    try:
                        db.save_price_tick(p.price, p.timestamp, p.source)
                    except Exception as e:
                        log.error(f"DB write failed (save_price_tick): {e}")
                    self._last_price_save = now

            if ws != self.window.open_time:
                # ── Capture open price FIRST, before resolve consumes time ──
                # Chainlink WS updates are irregular (1-5s gaps). If the last
                # update is stale (>3s), the captured price may lag behind
                # Polymarket's opening price. Use freshest source available.
                p = self.price_feed.get_price()  # Chainlink primary
                open_price_age = (now - p.timestamp) if p else 999
                if p and open_price_age > 3.0:
                    # Chainlink stale — try Binance HF (~1s resolution)
                    bp = self.price_feed.get_binance_price()
                    if bp and bp.price > 0 and (now - bp.timestamp) < 2.0:
                        log.info(f"[OPEN] Chainlink stale ({open_price_age:.1f}s), "
                                 f"using {bp.source}: ${bp.price:,.2f}")
                        p = bp
                if not p:
                    return

                # Resolve previous window AFTER capturing the open price
                if self.window.is_active:
                    self._resolve()
                self.window = WindowState(
                    window_id=self.window.window_id + 1,
                    open_price=p.price, current_price=p.price,
                    open_time=ws, end_time=we, is_active=True,
                    up_odds=50, down_odds=50,
                )
                self.current_bet = None
                self.current_signal = None
                self._last_mid_update = 0
                self._bet_evaluation_done = False
                # Reset sniper consensus state
                self._window_evaluations = []
                self._window_bet_placed = False
                self._last_eval_time = 0
                self._is_ghost_trade = False
                self._ghost_reason = ""
                self._consensus_score = 0
                self._consensus_direction = None
                self._consensus_idx = -1
                self._last_trigger_type = None
                self._last_sniper_ask_cents = None
                self._last_sniper_prob = None
                self._last_sniper_price = None
                self._last_sniper_price_ts = 0.0
                self._sniper_trigger_ready_ts = 0.0
                self._ml_oracle_prob = None
                self._ml_skipped = False
                self._pending_retry = None
                self._swarm_signals = {}
                # Second Opinion — fresh slate per window
                self._second_opinion = None
                self._second_opinion_in_flight = False
                self._second_opinion_launched_at = 0.0
                self._prediction_arrival_time = 0.0

                utc_start = datetime.fromtimestamp(ws, tz=timezone.utc).strftime('%H:%M')
                utc_end = datetime.fromtimestamp(we, tz=timezone.utc).strftime('%H:%M')
                log.info(f"=== Window #{self.window.window_id} | "
                         f"{utc_start}-{utc_end} UTC | "
                         f"Open: ${p.price:,.2f} ({p.source}) ===")

                try:
                    db.save_window(self.window.window_id, ws, p.price)
                except Exception as e:
                    log.error(f"DB write failed (save_window): {e}")

                # Flag for heavy work outside lock
                elapsed_since_open = now - ws
                if elapsed_since_open > 60:
                    log.info(f"[BOOT] Omitiendo ventana #{self.window.window_id} ({int(elapsed_since_open)}s de retraso) "
                             f"— esperando la siguiente ventana para ahorrar tokens de IA.")
                    need_prediction = False
                else:
                    need_prediction = True
                    
                new_window_price = p
                new_window_id = self.window.window_id
            else:
                # Mid-window: Chainlink primary (matches resolution oracle), Binance fallback
                p = self.price_feed.get_price()
                if not p or p.price <= 0:
                    p = self.price_feed.get_binance_price()
                if p:
                    self.window.current_price = p.price

                    # Odds: prefer Polymarket live WS prices > internal formula
                    ws_up, ws_down, ws_ts = self.polymarket.live.get_live_prices()
                    ws_fresh = (now - ws_ts < 5.0) if ws_ts > 0 else False
                    if ws_up > 0 and ws_down > 0 and ws_fresh:
                        # Real Polymarket odds (token prices → cents)
                        self.window.up_odds = max(5, min(95, round(ws_up * 100)))
                        self.window.down_odds = max(5, min(95, round(ws_down * 100)))
                    else:
                        # Fallback: internal formula
                        diff = p.price - self.window.open_price
                        pct = (diff / self.window.open_price * 10000) if self.window.open_price else 0
                        base = 50 + pct * 2
                        if self.current_signal:
                            base += (self.current_signal.confidence - 50) * 0.2
                        self.window.up_odds = max(5, min(95, round(base)))
                        self.window.down_odds = 100 - self.window.up_odds

                    # ── TA + HMM moved OUTSIDE lock (megarefactor B3) ──
                    # Flags for deferred heavy work
                    self._need_ta_update = (now - self._last_ta_update > 10)
                    self._need_regime_decode = (
                        self.regime.available
                        and now - self._last_regime_decode_ts
                            > self._REGIME_DECODE_INTERVAL
                    )

                    # Mid-window confidence update every ~30s (no API)
                    if (self.current_signal and self.window.is_active and
                            now - self._last_mid_update > 30):
                        self.current_signal = self.ai.update_confidence(
                            self.current_signal, p.price,
                            self.window.open_price,
                            self.current_ta, self._time_left()
                        )
                        self._last_mid_update = now

                    # ── Variant D: Background Second Opinion @ T+120s ──
                    # Once per window, launches a tactical LLM consult that
                    # updates signal.confidence via Option Y (decay-clamped).
                    if (self._VARIANT_D_ENABLED
                            and self.current_signal
                            and self.window.is_active
                            and not self.current_bet
                            and not self._window_bet_placed
                            and not self._bet_evaluation_done
                            and (now - self.window.open_time)
                                >= self._VARIANT_D_LAUNCH_SEC
                            and not self._second_opinion_in_flight
                            and self._consume_second_opinion() is None):
                        log.info(f"[2nd-OP:D] launching background consult "
                                 f"@ elapsed={now - self.window.open_time:.0f}s")
                        self._launch_second_opinion_async(p, source="D")

                    # ── Sniper Consensus: intra-candle evaluation ──
                    elapsed = now - self.window.open_time
                    time_left = self._time_left()
                    if (self.current_signal and self.window.is_active
                            and not self.current_bet
                            and not self._window_bet_placed
                            and not self._bet_evaluation_done):

                        # Time-stop: cutoff reached without score consensus.
                        # Order of rescue attempts before SKIP:
                        #   1) Last-Minute Rescue — best eval passing conf+EV floors
                        #   2) Meta-Swarm Override — unanimous secondary agents
                        #   3) SKIP
                        if elapsed >= self._EVAL_CUTOFF:
                            rescued = self._try_last_minute_rescue(elapsed)
                            if not rescued:
                                summary = self._summarize_window_evals()
                                overridden = self._check_meta_swarm_override(
                                    f"SNIPER TIME-STOP (evals={summary['n']}, "
                                    f"max_conf={summary['max_conf']}%, "
                                    f"flips={summary['num_flips']}, "
                                    f"score={summary['last_score']})"
                                )
                                if not overridden:
                                    self._bet_evaluation_done = True
                                    log.info(
                                        f"SNIPER TIME-STOP: elapsed={elapsed:.0f}s "
                                        f"evals={summary['n']} "
                                        f"max_conf={summary['max_conf']}% "
                                        f"last_dir={summary['last_direction']} "
                                        f"score={summary['last_score']} "
                                        f"flips={summary['num_flips']} "
                                        f"reason=no_consensus+rescue_failed — SKIP"
                                    )

                        # Periodic re-evaluation — adaptive interval based on velocity
                        elapsed_pred = (now - self._prediction_arrival_time) if self._prediction_arrival_time > 0 else 0.0
                        if (elapsed_pred >= self._FIRST_EVAL_DELAY
                              and now - self._last_eval_time >= self._next_eval_interval(p, now)):
                            self._run_sniper_evaluation(p, now, elapsed)

            if now >= self.window.end_time and self.window.is_active:
                self._resolve()

        # ── Phase 1.5: Heavy CPU work OUTSIDE lock (megarefactor B3) ──
        if getattr(self, '_need_ta_update', False):
            try:
                prices = self.price_feed.get_recent_prices(500)
                if prices:
                    ohlcv = self.binance.get_klines_5m_ohlcv()
                    self.current_ta = self.ta.analyze(prices, ohlcv or None)
                    self._last_ta_update = time.time()
            except Exception as e:
                log.debug(f"[TA] mid-window refresh failed: {e}")
            self._need_ta_update = False

        if getattr(self, '_need_regime_decode', False):
            try:
                ohlcv_r = self.binance.get_klines_5m_ohlcv()
                snap = self.regime.refresh(ohlcv_r or [])
                if snap:
                    self._current_regime = snap
                self._last_regime_decode_ts = time.time()
            except Exception as e:
                log.debug(f"[HMM] mid-window refresh failed: {e}")
                self._last_regime_decode_ts = time.time()
            self._need_regime_decode = False

        # ── Phase 2: Launch prediction in background thread ─────
        #    Tick loop returns immediately — prices keep flowing
        #    Circuit Breaker: skip AI work if primary is halted (prices keep flowing)
        if need_prediction and self._primary_halted:
            log.warning(f"[CIRCUIT BREAKER] Primary AI HALTED — skipping prediction "
                        f"for window #{new_window_id}. Reason: {self._primary_halt_reason}")
            return
        if need_prediction and new_window_price and not self._prediction_in_flight:
            self._prediction_in_flight = True
            threading.Thread(
                target=self._fetch_and_process_prediction,
                args=(new_window_price, new_window_id),
                daemon=True,
                name=f"ai-predict-w{new_window_id}",
            ).start()
            log.info(f"AI prediction thread launched for window #{new_window_id}")

    def _fetch_and_process_prediction(self, p, new_window_id: int):
        """Heavy AI prediction work — runs in a background thread.

        Gathers market data, calls Primary AI + Local LLM, applies
        post-prediction adjustments, then acquires _lock to store results.
        The tick loop keeps running while this executes.
        """
        try:
            # ── Early Polymarket subscription: get token IDs for live WS stream ──
            try:
                poly_q = self.polymarket.get_current_btc5min_quote()
                if not poly_q.is_stale and poly_q.up_token_id:
                    log.info(f"[POLYMARKET] Window #{new_window_id} → "
                             f"{poly_q.question} | "
                             f"UP={poly_q.up_price:.3f} DOWN={poly_q.down_price:.3f}")
            except Exception as e:
                log.debug(f"[POLYMARKET] Early fetch failed (non-fatal): {e}")

            news = sentiment.get_market_context()

            # ── RAG: retrieve best institutional strategy for current context ──
            # Clean expired blacklist entries first (kept as safety net only)
            now_ts = time.time()
            with self._fatigue_lock:
                expired = [s for s, exp in self._strategy_blacklist.items() if now_ts >= exp]
                for s in expired:
                    log.info(f"[FATIGUE] Strategy '{s}' cooldown expired — available again")
                    del self._strategy_blacklist[s]
                exclude = list(self._strategy_blacklist.keys())

            # ── HMM Regime: decode current Markov state (Viterbi, ~1ms) ──
            regime_snapshot: dict | None = None
            if self.regime.available:
                try:
                    ohlcv_for_regime = self.binance.get_klines_5m_ohlcv() or []
                    regime_snapshot = self.regime.refresh(ohlcv_for_regime)
                    if regime_snapshot:
                        self._current_regime = regime_snapshot
                        self._last_regime_decode_ts = now_ts
                        log.info(
                            f"[HMM] Window #{new_window_id} regime="
                            f"{regime_snapshot['label']} "
                            f"(p={regime_snapshot['confidence']:.2%}, "
                            f"state={regime_snapshot['state_idx']}, "
                            f"probs={regime_snapshot['state_probs']})"
                        )
                except Exception as e:
                    log.warning(f"[HMM] decode failed (non-fatal): {e}")

            rag_strategy = None
            if self.rag.available:
                fg = sentiment.get_fear_greed()
                fg_text = (f"Fear&Greed={fg['value']}({fg['label']})"
                           if fg.get("value") else "")
                base_ctx = f"{fg_text} | {news}" if fg_text else news
                # Regime drives rotation: bias retrieval toward compatible families
                regime_hint = ""
                if regime_snapshot:
                    regime_hint = regime_rag_hint(regime_snapshot["label"])
                rag_context = f"{regime_hint} || {base_ctx}" if regime_hint else base_ctx
                rag_strategy = self.rag.get_best_strategy(
                    rag_context, exclude_strategies=exclude,
                )
                if rag_strategy:
                    strat_name = rag_strategy.get("strategy_name", "")
                    self._current_rag_strategy = strat_name
                    # Track start time for fatigue timer
                    with self._fatigue_lock:
                        if strat_name not in self._strategy_start_time:
                            self._strategy_start_time[strat_name] = time.time()
                    _regime_tag = (f" (regime={regime_snapshot['label']})"
                                   if regime_snapshot else "")
                    _excl_tag = f" (excluidas: {exclude})" if exclude else ""
                    log.info(f"[RAG] Estrategia seleccionada para ventana "
                             f"#{new_window_id}: {self._current_rag_strategy}"
                             f"{_regime_tag}{_excl_tag}")
                else:
                    self._current_rag_strategy = ""

            prices = self.price_feed.get_recent_prices(500)
            ohlcv_5m = self.binance.get_klines_5m_ohlcv()
            ta = self.ta.analyze(prices, ohlcv_5m or None)

            # Gather multi-timeframe and market data
            hf_prices = self.price_feed.get_hf_prices()
            market_ctx = self.binance.get_context()
            klines_15m = self.binance.get_klines("15m")
            klines_1h = self.binance.get_klines("1h")

            # ── CVD / Order Flow data ──
            cvd_summary = self.binance.get_cvd_summary(minutes=5)
            self.binance.reset_cvd_window()  # Reset accumulator for new window

            # ── GEX / Gamma Exposure data ──
            gex_data = self.gex.get_gex()

            # Compute ML features (now includes CVD + GEX)
            ml_features = self.features.compute(
                prices, ta,
                price_history_1h=hf_prices if hf_prices else None,
                market_ctx=market_ctx,
                klines_15m=klines_15m if klines_15m else None,
                klines_1h=klines_1h if klines_1h else None,
                cvd_data=cvd_summary if cvd_summary.get("connected") else None,
                gex_data=gex_data if gex_data else None,
            )
            ml_features["calibrated_confidence"] = None

            # ATR Quality Gate (dynamic rolling baseline)
            try:
                rolling_atr = db.get_rolling_atr_baseline(hours=24)
            except Exception as e:
                log.error(f"DB read failed (get_rolling_atr_baseline): {e}")
                rolling_atr = None
            atr_quality, atr_reason = self.features.atr_quality_check(
                ta, p.price, rolling_atr
            )
            ml_features["signal_quality"] = atr_quality
            ml_features["quality_reason"] = atr_reason if atr_reason else None

            # Liquidation data
            liq_summary = self.binance.get_liquidation_summary(minutes=5)
            ml_features["liq_buy_vol_5m"] = liq_summary.get("buy_vol", 0)
            ml_features["liq_sell_vol_5m"] = liq_summary.get("sell_vol", 0)
            ml_features["reversal_alert"] = 0

            # Liquidation cluster proximity (Vibe-Trading inspired).
            # Stores under a single namespace so UI + prompts can read without
            # polluting the 38-feature Judge input. Add individual scalars to
            # _JUDGE_FEATURES only after a retrain.
            try:
                _liq_cluster = self.binance.get_liq_cluster_proximity(
                    current_price=p.price,
                    atr=float(ta.get("atr", 0.0) or 0.0),
                )
                ml_features["liq_cluster"] = _liq_cluster
                ml_features["liq_cluster_distance_atr"] = _liq_cluster.get("distance_atr")
                ml_features["liq_cluster_magnitude_pct"] = _liq_cluster.get("magnitude_pct")
                ml_features["liq_cluster_side"] = _liq_cluster.get("cluster_side")
                self._current_liq_cluster = _liq_cluster
            except Exception as _e:
                log.debug(f"liq_cluster_proximity failed: {_e}")
                self._current_liq_cluster = {}
            ml_features["order_book_imbalance"] = (
                market_ctx.get("order_book_imbalance") if market_ctx else None
            )
            ml_features["bid_ask_spread"] = (
                market_ctx.get("bid_ask_spread") if market_ctx else None
            )
            ml_features["volatility_regime"] = self.ai.volatility_regime
            from . import sentiment as _sent
            _fg = _sent.get_fear_greed()
            ml_features["fear_greed"] = _fg.get("value") if _fg else None

            # Multi-timeframe trend direction
            trend_direction = self.features.get_trend_direction(klines_15m, klines_1h)

            # ── SMC Features (smartmoneyconcepts library) ──
            ohlcv_5m = self.binance.get_klines_5m_ohlcv()
            smc_feats = smc_features.compute_smc_features(ohlcv_5m)
            if smc_feats:
                ml_features.update(smc_feats)
                self._current_smc_features = smc_feats
            else:
                self._current_smc_features = {}

            # ── MSS / Liquidity Sweep (mid-window real-time) ──
            mss_feats = smc_features.compute_midwindow_mss(
                ohlcv_5m,
                window_open_price=p.price,
                current_price=p.price,
                daily_high=self._get_daily_high(ohlcv_5m),
                daily_low=self._get_daily_low(ohlcv_5m),
            )
            if mss_feats:
                ml_features.update(mss_feats)
                self._current_mss_features = mss_feats

            # Save market context to DB
            if market_ctx:
                try:
                    db.save_market_context(new_window_id, market_ctx)
                except Exception as e:
                    log.error(f"DB write failed (save_market_context): {e}")

            # Cleanup old liquidations + price ticks (once per day)
            try:
                db.cleanup_old_liquidations(hours=24)
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if self._last_tick_cleanup_date != today:
                    db.cleanup_old_ticks(hours=72)
                    self._last_tick_cleanup_date = today
            except Exception as e:
                log.error(f"DB cleanup failed: {e}")

            # ── SMC: inject computed SMC analysis into prompt context ──
            if smc_feats:
                smc_block = smc_features.build_smc_prompt_block(
                    smc_feats, p.price
                )
                if smc_block:
                    news = news + "\n\n" + smc_block

            # ── MSS: inject liquidity sweep & MSS into prompt context ──
            if mss_feats:
                mss_block = smc_features.build_mss_prompt_block(mss_feats)
                if mss_block:
                    news = news + "\n\n" + mss_block

            # ── RAG: inject institutional strategy into prompt context ──
            if rag_strategy:
                strat_name = rag_strategy.get("strategy_name", "N/A")
                entry_rules = rag_strategy.get("entry_rules", "")
                rag_block = (
                    f"\n\n[ESTRATEGIA RAG ACTIVA: {strat_name}]\n"
                    f"Reglas de entrada: {entry_rules}\n"
                    f"⚠️ OBLIGATORIO: Evalua si las condiciones actuales cumplen las reglas de entrada.\n"
                    f"- Si se cumplen → adopta la direccion, +10 confianza.\n"
                    f"- Si NO se cumplen → ignora, declara en reasoning \"RAG descartada: [motivo]\".\n"
                    f"- Si contradice TA → confianza MAXIMA 60%, declara conflicto."
                )
                news = news + rag_block
                ml_features["rag_strategy"] = strat_name

            # ── HMM: inject Markov regime state into prompt + ml_features ──
            if regime_snapshot:
                regime_block = (
                    f"\n\n[REGIMEN DE MARKOV (HMM)]\n"
                    f"Estado oculto actual: {regime_snapshot['label']} "
                    f"(idx={regime_snapshot['state_idx']})\n"
                    f"Confianza posterior: {regime_snapshot['confidence']*100:.1f}%\n"
                    f"Distribucion de estados: {regime_snapshot['state_probs']}\n"
                    f"Transicion probable: {regime_snapshot['transition_to']}\n"
                    f"⚠️ REGLA: Si TA contradice el regimen, reduce confianza "
                    f"salvo que sweep/MSS invalide el estado actual."
                )
                news = news + regime_block
                # Expose to AI ctx + ml_features for LSTM/XGBoost judges
                ml_features["_hmm_regime"] = regime_snapshot
                ml_features["regime_label"] = regime_snapshot["label"]
                ml_features["regime_state_idx"] = regime_snapshot["state_idx"]
                ml_features["regime_confidence"] = regime_snapshot["confidence"]

            # ── Polymarket EV context for AI prompt ──
            try:
                _pq = self.polymarket.get_current_btc5min_quote()
                if _pq and not _pq.is_stale:
                    ml_features["poly_strike_price"] = _pq.strike_price
                    ml_features["poly_strike_dist"] = (
                        round(p.price - _pq.strike_price, 2)
                        if _pq.strike_price > 0 else None
                    )
                    ml_features["poly_up_odds_cents"] = round(_pq.up_price * 100, 1)
                    ml_features["poly_down_odds_cents"] = round(_pq.down_price * 100, 1)
                    # Simple EV = conf * payout; payout = 100/price_cents
                    # Injected as raw odds so the AI can compute EV per side
                    ml_features["poly_up_payout"] = (
                        round(1.0 / _pq.up_price, 2) if _pq.up_price > 0.01 else None
                    )
                    ml_features["poly_down_payout"] = (
                        round(1.0 / _pq.down_price, 2) if _pq.down_price > 0.01 else None
                    )
            except Exception as _pq_err:
                log.debug(f"[EV-CTX] Polymarket quote for prompt failed: {_pq_err}")

            # ── Temporal velocity features (rolling TA buffer) ──
            _primary_aid = get_primary_agent_id()
            self.ai.push_ta_snapshot(_primary_aid, ta)
            _velocities = self.ai.compute_ta_velocities(_primary_aid)
            ml_features.update(_velocities)

            # Primary AI prediction (the main blocking call — semaphore-guarded)
            # Circuit Breaker: catches fatal API errors (402/500/timeout) and halts system
            _primary_cfg = self.ai._get_primary_config()
            _skip_semaphore = _primary_cfg.get("api_type") == "local"
            _t0 = time.time()
            # HFT Phase 2 needs window_id + timestamp for reasoning backfill
            self.ai._current_window_id = new_window_id
            ml_features["_t0"] = _t0
            if not _skip_semaphore:
                _primary_semaphore.acquire()
            try:
                signal = self.ai.predict(p, ta, news, context=ml_features)
            except Exception as ai_err:
                err_str = str(ai_err)
                # Detect fatal errors: 402 (tokens exhausted), 500, timeout
                is_fatal = any(code in err_str for code in ("402", "500", "503"))
                if "timeout" in err_str.lower():
                    is_fatal = True
                if is_fatal:
                    self._primary_halted = True
                    self._primary_halt_reason = err_str[:200]
                    self._primary_halt_time = time.time()
                    log.critical(f"[CIRCUIT BREAKER] Primary AI HALTED — {err_str[:200]}")
                    log.critical("[CIRCUIT BREAKER] El sistema pausará predicciones. "
                                 "Usa /api/resume_primary para reanudar.")
                else:
                    log.error(f"Primary AI prediction failed (non-fatal): {ai_err}")
                return
            finally:
                if not _skip_semaphore:
                    _primary_semaphore.release()
            ml_features["ai_latency_ms"] = round((time.time() - _t0) * 1000, 1)
            ml_features["price_at_bet"] = p.price
            ml_features["api_tokens_used"] = getattr(signal, "usage_tokens", None)

            # Local-LLM ensemble features are now dead columns — local
            # models run through the normal swarm path, not a parallel
            # second door. Keep the keys for ML schema compatibility.
            ml_features["local_ai_prediction"] = None
            ml_features["local_ai_confidence"] = None
            ml_features["local_ai_reasoning"] = None

            # ── Post-prediction contextual adjustments ─────────
            signal = self.ai.mark_signal_quality(signal, atr_quality, atr_reason)
            signal = self.ai.apply_trend_alignment(signal, trend_direction)
            signal, should_skip_liq = self.ai.check_liquidation_reversal(
                signal, liq_summary
            )
            if liq_summary.get("spike"):
                ml_features["reversal_alert"] = 1

            ml_features["calibrated_confidence"] = signal.confidence
            ml_features["layer_alignment"] = signal.layer_alignment

            # consensus_agreement was Primary-vs-LocalLLM ensemble. Local
            # models are now regular swarm agents, so there is no "primary
            # vs secondary" pairing at T+0. Column retained as None.
            ml_features["consensus_agreement"] = None

            # ── Polymarket EV features for DB/ML (direction-aware) ──
            _side_odds = ml_features.get(
                "poly_up_odds_cents" if signal.prediction == "UP"
                else "poly_down_odds_cents"
            )
            ml_features["polymarket_odds"] = _side_odds
            ml_features["distance_to_strike"] = ml_features.get("poly_strike_dist")
            if _side_odds and _side_odds > 0:
                _payout = 100.0 / _side_odds
                ml_features["expected_value"] = round(
                    (signal.confidence / 100.0) * _payout, 4
                )
            else:
                ml_features["expected_value"] = None

            # ── Acquire lock to store results ──────────────────
            with self._lock:
                # Verify we're still on the same window (could have rolled over
                # if primary AI was extremely slow — discard stale prediction)
                if self.window.window_id != new_window_id:
                    log.warning(f"AI prediction for window #{new_window_id} arrived "
                                f"but current window is #{self.window.window_id} — discarding")
                    return

                self.current_ta = ta
                self.current_signal = signal

                # ── ML Oracle evaluation ──
                oracle_prob, oracle_skip = self._evaluate_oracle(ml_features)
                self._ml_oracle_prob = oracle_prob
                self._ml_skipped = oracle_skip

                try:
                    db.save_prediction(
                        self.window.window_id,
                        signal.prediction, signal.confidence,
                        signal.original_confidence, signal.reasoning,
                        signal.risk_score, ta,
                        ml_features=ml_features,
                        ai_model=get_primary_agent_id(),
                        rag_strategy=self._current_rag_strategy or None,
                        ml_oracle_prob=oracle_prob,
                        ml_skipped=oracle_skip,
                    )
                except Exception as e:
                    log.error(f"DB write failed (save_prediction): {e}")

                if should_skip_liq and not getattr(cfg, 'PAPER_TRADING_MODE', True):
                    log.info(f"Signal: {signal.prediction} ({signal.confidence}%) "
                             f"— SKIPPED due to liquidation chaos")
                    self._bet_evaluation_done = True
                else:
                    mode_tag = "PAPER-SNIPER" if getattr(cfg, 'PAPER_TRADING_MODE', True) else "SNIPER"
                    self._window_evaluations = [{
                        "minute": 0,
                        "prediction": signal.prediction,
                        "confidence": signal.confidence,
                        "timestamp": time.time(),
                    }]
                    self._prediction_arrival_time = time.time()
                    log.info(f"{mode_tag} INIT: {signal.prediction} ({signal.confidence}%) "
                             f"— registered as eval #1, waiting {int(self._FIRST_EVAL_DELAY)}s for consensus")

            # ── Phase 3: Multi-Agent Swarm (parallel, non-blocking) ──
            self._run_swarm_agents(new_window_id, p, ta, news, ml_features)

        except Exception:
            log.exception(f"AI prediction thread error (window #{new_window_id})")
        finally:
            self._prediction_in_flight = False

    # ── Second Opinion helpers ──────────────────────────────

    def _poly_quote_snapshot(self) -> dict | None:
        """Return a compact {up_price_cents, down_price_cents} snapshot
        for quick_second_opinion prompts. Returns None if stale/missing."""
        try:
            q = self.polymarket.get_current_btc5min_quote()
        except Exception:
            return None
        if q is None or getattr(q, "is_stale", True):
            return None
        up = getattr(q, "up_price", None)
        dn = getattr(q, "down_price", None)
        if up is None or dn is None:
            return None
        try:
            return {
                "up_price_cents": float(up) * 100.0,
                "down_price_cents": float(dn) * 100.0,
            }
        except (TypeError, ValueError):
            return None

    def _market_context_snapshot(self) -> dict:
        """Captures a thread-safe snapshot of live market data for LLM."""
        try:
            ctx = self.binance.get_context() if getattr(self, "binance", None) else {}
            cvd = self.binance.get_cvd_summary(minutes=3) if getattr(self, "binance", None) else {}
            ctx["cvd_imbalance_pct"] = cvd.get("cvd_imbalance_pct", 0)
            return ctx
        except Exception:
            return {}

    def _consume_second_opinion(self, max_age: float | None = None) -> dict | None:
        """Return current _second_opinion if fresh (age ≤ max_age) AND
        belongs to the current window. Non-destructive — multiple callers
        can share the same opinion. Returns None if stale/missing/wrong window."""
        op = self._second_opinion
        if not op:
            return None
        ttl = max_age if max_age is not None else self._SECOND_OPINION_TTL
        age = time.time() - float(op.get("arrived_at", 0))
        if age > ttl:
            return None
        if op.get("window_id") != getattr(self.window, "window_id", None):
            return None
        return op

    def _apply_second_opinion_to_signal(self, op: dict) -> None:
        """Update self.current_signal.confidence using Option Y (time-decay
        preserved). Only applied when op direction agrees with the signal.
        Caller already holds self._lock."""
        if not self.current_signal or not op:
            return
        if op.get("direction") != self.current_signal.prediction:
            log.info(f"[2nd-OP:{op.get('source','?')}] direction mismatch "
                     f"(llm={op.get('direction')} vs signal="
                     f"{self.current_signal.prediction}) — signal unchanged")
            return
        # Option Y: respect time-decay floor
        llm_conf = int(op.get("confidence", 0))
        orig_conf = int(getattr(self.current_signal, "original_confidence",
                                self.current_signal.confidence))
        elapsed = max(0.0, time.time() - self.window.open_time)
        decay = (elapsed / 60.0) * self._CONFIDENCE_DECAY_PER_MIN
        pre_decay_cap = max(self._DECAY_FLOOR_CONF, int(orig_conf - decay))
        new_conf = max(self._DECAY_FLOOR_CONF, min(llm_conf, pre_decay_cap))
        if new_conf != self.current_signal.confidence:
            log.info(f"[2nd-OP:{op.get('source','?')}] confidence update "
                     f"{self.current_signal.confidence}% → {new_conf}% "
                     f"(llm={llm_conf} pre_decay_cap={pre_decay_cap})")
            self.current_signal.confidence = new_conf

    def _launch_second_opinion_async(self, price_data, source: str) -> None:
        """Launch a non-blocking second-opinion LLM consult in a daemon
        thread. No-op if already in-flight or if a fresh cache exists."""
        if not self._SECOND_OPINION_ENABLED:
            return
        if self._second_opinion_in_flight:
            return
        if self._consume_second_opinion() is not None:
            return

        # Capture immutable snapshot under the caller's lock
        window_id = self.window.window_id
        current_price = float(price_data.price)
        open_price = float(self.window.open_price)
        ta_snapshot = dict(self.current_ta) if self.current_ta else {}
        elapsed = max(0.0, time.time() - self.window.open_time)
        time_left = self._time_left()
        poly_quote = self._poly_quote_snapshot()
        mkt_ctx_snap = self._market_context_snapshot()

        self._second_opinion_in_flight = True
        self._second_opinion_launched_at = time.time()

        def _worker():
            try:
                op = self.ai.quick_second_opinion(
                    current_price=current_price,
                    open_price=open_price,
                    ta=ta_snapshot,
                    elapsed_sec=elapsed,
                    time_left_sec=time_left,
                    poly_quote=poly_quote,
                    source=source,
                    context=mkt_ctx_snap,
                )
            except Exception as e:
                log.warning(f"[2nd-OP:{source}] worker crashed: {e}")
                with self._lock:
                    self._second_opinion_in_flight = False
                return

            with self._lock:
                self._second_opinion_in_flight = False
                # Drop result if window rolled during the call
                if self.window.window_id != window_id:
                    log.info(f"[2nd-OP:{source}] window rolled during call — "
                             f"dropping result")
                    return
                if op.get("error"):
                    return
                self._second_opinion = {
                    **op,
                    "arrived_at": time.time(),
                    "window_id": window_id,
                }
                # For async variants (D, B) apply to signal immediately.
                # Sync variants (C, A) read the return value directly.
                if source in ("D", "B"):
                    self._apply_second_opinion_to_signal(self._second_opinion)

        threading.Thread(
            target=_worker,
            daemon=True,
            name=f"second-opinion-{source}",
        ).start()

    def _apply_market_context_penalties(self, signal) -> list[str]:
        """Apply real-time market context penalties to signal confidence.

        Uses live Binance data (funding rate, CVD, liquidations) and
        Polymarket odds to penalize signals that conflict with market
        microstructure. All thresholds are configurable via
        dynamic_rules.json → "market_filters".

        Returns list of applied penalty tags for logging.
        """
        if not rules.get("market_filters", "enabled", True):
            return []

        penalties = []
        original_conf = signal.confidence

        # ── 1. Funding Rate: crowded trade penalty ──────────────
        # If funding is extreme and signal aligns with the crowded side,
        # reduce confidence. Crowded trades are vulnerable to liquidation
        # cascades in the opposite direction.
        try:
            ctx = self.binance.get_context()
            funding = ctx.get("funding_rate", 0) if ctx else 0
        except Exception:
            funding = 0

        fr_threshold = float(rules.get("market_filters",
                                        "funding_rate_threshold", 0.0003))
        fr_penalty = float(rules.get("market_filters",
                                      "funding_rate_penalty", 0.15))
        if funding and abs(funding) > fr_threshold:
            # Positive funding = longs crowded; negative = shorts crowded
            crowded_side = "UP" if funding > 0 else "DOWN"
            if signal.prediction == crowded_side:
                reduction = int(signal.confidence * fr_penalty)
                signal.confidence = max(45, signal.confidence - reduction)
                penalties.append(
                    f"FR={funding:+.4%}→-{reduction}% "
                    f"(crowded {crowded_side})")

        # ── 2. CVD Divergence: exhaustion detection ─────────────
        # If price moved in one direction but aggressive order flow
        # (CVD) went the opposite way, the move lacks conviction.
        try:
            cvd = self.binance.get_cvd_summary(minutes=3)
            cvd_imb = cvd.get("cvd_imbalance_pct", 0) if cvd else 0
        except Exception:
            cvd_imb = 0

        cvd_threshold = float(rules.get("market_filters",
                                         "cvd_divergence_threshold", 15))
        cvd_penalty_pts = int(rules.get("market_filters",
                                         "cvd_divergence_penalty", 8))
        if cvd_imb != 0 and abs(cvd_imb) > cvd_threshold:
            # CVD positive = buyers dominating, negative = sellers
            cvd_favors = "UP" if cvd_imb > 0 else "DOWN"
            if signal.prediction != cvd_favors:
                signal.confidence = max(45, signal.confidence - cvd_penalty_pts)
                penalties.append(
                    f"CVD_DIV={cvd_imb:+.0f}%→-{cvd_penalty_pts} "
                    f"(flow favors {cvd_favors})")

        # ── 3. Liquidation cascade: active liquidations on our side ──
        # If Binance is actively liquidating positions in OUR direction,
        # the market is punishing that side — don't join the losers.
        try:
            liq = self.binance.get_liquidation_summary(minutes=2)
        except Exception:
            liq = {}

        liq_penalty_pts = int(rules.get("market_filters",
                                         "liquidation_penalty", 10))
        liq_min_vol = float(rules.get("market_filters",
                                       "liquidation_min_vol_usd", 50000))
        if liq.get("spike"):
            # "spike" means unusual liquidation activity
            buy_liq = liq.get("buy_vol", 0)   # longs being liquidated
            sell_liq = liq.get("sell_vol", 0)  # shorts being liquidated
            if signal.prediction == "UP" and buy_liq > liq_min_vol:
                signal.confidence = max(45, signal.confidence - liq_penalty_pts)
                penalties.append(
                    f"LIQ_CASCADE=${buy_liq:,.0f}→-{liq_penalty_pts} "
                    f"(longs liquidated)")
            elif signal.prediction == "DOWN" and sell_liq > liq_min_vol:
                signal.confidence = max(45, signal.confidence - liq_penalty_pts)
                penalties.append(
                    f"LIQ_CASCADE=${sell_liq:,.0f}→-{liq_penalty_pts} "
                    f"(shorts liquidated)")

        # ── 4. Dead hour filter: low-volume UTC hours ─────────
        # BTC volume drops significantly during 2-7 UTC. Lower volume
        # → wider Polymarket spreads, less predictable moves, worse EV.
        from datetime import datetime, timezone
        hour_utc = datetime.now(timezone.utc).hour
        dead_start = int(rules.get("market_filters", "dead_hour_start", 2))
        dead_end = int(rules.get("market_filters", "dead_hour_end", 7))
        dead_penalty = float(rules.get("market_filters",
                                        "dead_hour_penalty", 0.10))
        if dead_start <= hour_utc <= dead_end:
            reduction = int(signal.confidence * dead_penalty)
            signal.confidence = max(45, signal.confidence - reduction)
            penalties.append(f"DEAD_HOUR={hour_utc}hUTC→-{reduction}%")

        if penalties:
            self.current_signal = signal
            log.info(f"[MKT FILTERS] {original_conf}%→{signal.confidence}% "
                     f"| {' | '.join(penalties)}")

        return penalties

    def _compute_polymarket_ev(self, signal) -> tuple[float | None, float | None, float | None]:
        """Lee odds vivas de Polymarket y calcula EV teórico.

        Returns (ev, price_cents, payout_ratio) o (None, None, None) si
        el book está stale / inválido.
            payout_ratio = 100 / price_cents
            ev           = (confidence / 100) * payout_ratio
        EV > 1 indica edge positivo; > _EV_FLASH_FIRE = edge brutal.
        """
        try:
            quote = self.polymarket.get_current_btc5min_quote()
        except Exception as e:
            log.debug(f"[EV] polymarket quote error: {e}")
            return None, None, None
        if quote is None or quote.is_stale:
            return None, None, None
        side_price = quote.up_price if signal.prediction == "UP" else quote.down_price
        if not side_price or side_price <= 0 or side_price >= 1:
            return None, None, None
        price_cents = side_price * 100.0
        payout_ratio = 100.0 / price_cents
        ev = (float(signal.confidence) / 100.0) * payout_ratio
        return ev, price_cents, payout_ratio

    def _next_eval_interval(self, price_data, now: float) -> float:
        """Adaptive interval: speed up when BTC is moving fast, slow when flat.

        Uses |dPrice/dt| over the window since last sampled price. When velocity
        exceeds ``velocity_fast_usd_per_sec``, returns ``adaptive_interval_fast_sec``;
        when near zero, returns ``adaptive_interval_slow_sec``; otherwise falls back
        to the static ``eval_interval_sec``.
        """
        base = float(self._EVAL_INTERVAL)
        if not self._ADAPTIVE_INTERVAL_ENABLED:
            return base
        prev_p = self._last_sniper_price
        prev_ts = self._last_sniper_price_ts
        # refresh snapshot each tick so velocity is always ~2s window
        self._last_sniper_price = float(price_data.price)
        self._last_sniper_price_ts = now
        if prev_p is None or prev_ts <= 0 or (now - prev_ts) < 1.0:
            return base
        velocity = abs(float(price_data.price) - prev_p) / max(now - prev_ts, 1.0)
        if velocity >= self._VELOCITY_FAST_USD_PER_SEC:
            return self._ADAPTIVE_FAST_SEC
        if velocity < self._VELOCITY_FAST_USD_PER_SEC * 0.25:
            return max(base, self._ADAPTIVE_SLOW_SEC)
        return base

    def _run_sniper_evaluation(self, price_data, now: float, elapsed: float):
        """Run an AI re-evaluation and check consensus. Called with lock held."""
        self._last_eval_time = now
        eval_minute = int(elapsed // 60)

        # ── Mid-window MSS refresh: detect liquidity sweeps in real-time ──
        try:
            ohlcv_5m = self.binance.get_klines_5m_ohlcv()
            if ohlcv_5m:
                mss = smc_features.compute_midwindow_mss(
                    ohlcv_5m,
                    window_open_price=self.window.open_price,
                    current_price=price_data.price,
                    daily_high=self._get_daily_high(ohlcv_5m),
                    daily_low=self._get_daily_low(ohlcv_5m),
                )
                self._current_mss_features = mss
                if mss.get("post_sweep_reversal"):
                    log.info(f"[MSS] Post-sweep reversal at eval min {eval_minute}: "
                             f"{mss['sweep_direction']} → high-probability entry")
        except Exception as e:
            log.debug(f"[MSS] Mid-window refresh error: {e}")

        # Capture pre-eval state to detect FLIPs
        pre_prediction = self.current_signal.prediction if self.current_signal else None
        pre_confidence = self.current_signal.confidence if self.current_signal else None

        try:
            should_bet, signal, reason = self.ai.reevaluate_for_bet(
                self.current_signal, price_data.price,
                self.window.open_price, self.current_ta,
                force_decision=True
            )
            self.current_signal = signal
        except Exception as e:
            log.error(f"SNIPER eval error at min {eval_minute}: {e}")
            return

        # ── Watchful Hold: retry pending soft-veto before normal logic ──
        if self._pending_retry and not self._window_bet_placed:
            self._watchful_retry_tick(signal, price_data, elapsed)
            if self._window_bet_placed:
                return

        # ── Sync DB if prediction changed (FLIP or confidence shift) ──
        # Critical for ML Judge data integrity: DB must reflect the
        # actual decision used for betting, not the stale initial prediction.
        flipped = False
        if (pre_prediction and
                (signal.prediction != pre_prediction
                 or signal.confidence != pre_confidence)):
            flip_note = ""
            if signal.prediction != pre_prediction:
                flipped = True
                flip_note = (f" | SNIPER_FLIP: {pre_prediction}"
                             f"->{signal.prediction} @ eval min {eval_minute}")
                log.info(f"[DB-SYNC] Prediction FLIP detected for window "
                         f"#{self.window.window_id}: {pre_prediction}"
                         f"->{signal.prediction} ({pre_confidence}%"
                         f"->{signal.confidence}%)")
            try:
                db.update_prediction_direction(
                    self.window.window_id,
                    signal.prediction,
                    signal.confidence,
                    flip_reasoning=flip_note,
                    ai_model=get_primary_agent_id(),
                )
            except Exception as e:
                log.error(f"DB write failed (update_prediction_direction): {e}")

        # ── Variant B: tactical LLM consult on FLIP ──
        # Launch async so the next eval (or Rescue/Meta-Swarm) can
        # consume the fresh opinion. No-op if a fresh cache exists.
        if (flipped and self._VARIANT_B_ENABLED
                and self._consume_second_opinion() is None
                and not self._second_opinion_in_flight):
            log.info(f"[2nd-OP:B] flip detected "
                     f"({pre_prediction}→{signal.prediction}) — "
                     f"launching async consult")
            self._launch_second_opinion_async(price_data, source="B")

        # ── Confidence decay: time-based penalty for stale signals ──
        # `decay_mode`:
        #   "linear"   → rate × minutes  (classic)
        #   "quadratic"→ rate × minutes × (elapsed/300)^1.5 — lenient early,
        #                aggressive late. Better fit for 5m HFT windows where
        #                the first 60s are golden and the last 60s are noise.
        base_decay = (elapsed / 60.0) * self._CONFIDENCE_DECAY_PER_MIN
        if self._DECAY_MODE == "quadratic":
            decay = base_decay * (max(elapsed, 1.0) / 300.0) ** 1.5 * 3.0
        else:
            decay = base_decay
        if decay > 0:
            decayed_conf = max(self._DECAY_FLOOR_CONF,
                               int(signal.confidence - decay))
            if decayed_conf < signal.confidence:
                log.info(f"SNIPER decay: {signal.confidence}% -> {decayed_conf}% "
                         f"(elapsed={elapsed:.0f}s, rate="
                         f"{self._CONFIDENCE_DECAY_PER_MIN}%/min)")
                signal.confidence = decayed_conf
                self.current_signal = signal

        # ── Market microstructure penalties ──────────────────────
        # Apply funding rate, CVD divergence, liquidation cascade,
        # and dead-hour penalties BEFORE EV calculation so they
        # propagate through all downstream gates.
        mkt_penalties = self._apply_market_context_penalties(signal)

        eval_entry = {
            "minute": eval_minute,
            "prediction": signal.prediction,
            "confidence": signal.confidence,
            "timestamp": now,
        }
        self._window_evaluations.append(eval_entry)
        log.info(f"SNIPER eval #{len(self._window_evaluations)} @ min {eval_minute}: "
                 f"{signal.prediction} {signal.confidence}%")

        # ── EV-aware gating (Super Sniper) ─────────────────────────
        # Lee cuota viva de Polymarket y calcula esperanza matemática
        # teórica para balancear Tiempo vs Probabilidad vs Valor.
        ev, price_cents, payout_ratio = self._compute_polymarket_ev(signal)
        if ev is not None:
            log.info(f"[EV] {signal.prediction} @ {price_cents:.1f}¢ "
                     f"(payout ×{payout_ratio:.2f}) · conf={signal.confidence}% "
                     f"→ EV={ev:.3f}")

        # ── Regla 1 — EV Ghosting: decaimiento temporal destruyó R/R ──
        # Sólo se activa tras el minuto 2.5 (150s) y con EV < 1.
        # Única vía post-consenso hacia ghost (junto con news circuit
        # breaker / liquidation chaos / Judge gate).
        if (ev is not None
                and elapsed >= self._EV_GHOST_AFTER_SEC
                and ev < self._EV_GHOST_FLOOR):
            reason = (f"EV Ghosting (EV={ev:.2f} < {self._EV_GHOST_FLOOR:.2f} "
                      f"@ {elapsed:.0f}s, price={price_cents:.1f}¢)")
            log.info(f"SNIPER EV GHOSTING → {reason}")
            # Meta-Swarm Override: let swarm consensus rescue before ghosting
            overridden = self._check_meta_swarm_override(reason)
            if not overridden:
                self._bet_evaluation_done = True
                self._window_bet_placed = True
                self._is_ghost_trade = True
                self._ghost_reason = f"[GHOST] Decaimiento temporal destruyó el R/R ({reason})"
            return

        # ── Regla 2 — Flash Fire: EV brutal bypassa la regla de consenso ──
        flash_fire = (ev is not None and ev > self._EV_FLASH_FIRE)
        if flash_fire:
            log.info(f"SNIPER FLASH FIRE — EV={ev:.2f} > {self._EV_FLASH_FIRE} "
                     f"(price={price_cents:.1f}¢, payout ×{payout_ratio:.2f}) "
                     f"— bypassing score consensus")

        # ── Regla 2b — Early Flash Fire: EV extremo ya en el primer eval ──
        # Si EV > ev_early_fire_threshold y aún estamos en el primer minuto,
        # disparamos sin esperar al score. Captura ventanas "obvias" (gap
        # grande, odds desequilibradas) sin quemar 60 s de espera.
        early_flash = (ev is not None
                       and ev > self._EV_EARLY_FIRE
                       and eval_minute == 0
                       and len(self._window_evaluations) <= 2)
        if early_flash and not flash_fire:
            log.info(f"SNIPER EARLY FLASH FIRE — EV={ev:.2f} > "
                     f"{self._EV_EARLY_FIRE} @ eval #"
                     f"{len(self._window_evaluations)} (min {eval_minute})")

        # ── Variant C: Flash Fire sanity check (sync + veto) ──
        # Before committing to a Flash Fire (or Early Flash), ask the LLM
        # for a quick second opinion. If it vetoes, we fall through to the
        # normal score-consensus path instead of firing this eval.
        if (flash_fire or early_flash) and self._VARIANT_C_ENABLED:
            cached = self._consume_second_opinion()
            if cached is None:
                poly_snap = self._poly_quote_snapshot()
                mkt_snap = self._market_context_snapshot()
                ta_snap = dict(self.current_ta) if self.current_ta else {}
                elapsed_snap = elapsed
                time_left_snap = self._time_left()
                current_price_snap = float(price_data.price)
                open_price_snap = float(self.window.open_price)
                # Release lock for the sync LLM call (3s timeout hard cap)
                self._lock.release()
                try:
                    op = self.ai.quick_second_opinion(
                        current_price=current_price_snap,
                        open_price=open_price_snap,
                        ta=ta_snap,
                        elapsed_sec=elapsed_snap,
                        time_left_sec=time_left_snap,
                        poly_quote=poly_snap,
                        source="C",
                        context=mkt_snap,
                    )
                finally:
                    self._lock.acquire()
                # Re-check window didn't roll
                if self.window.window_id is not None and not op.get("error"):
                    self._second_opinion = {
                        **op,
                        "arrived_at": time.time(),
                        "window_id": self.window.window_id,
                    }
                    cached = self._second_opinion
                    # Apply confidence update (direction-aligned, decay-clamped)
                    self._apply_second_opinion_to_signal(cached)
                    # Refresh local reference after clamp
                    signal = self.current_signal
            if cached and cached.get("veto"):
                log.info(f"[2nd-OP:C] VETO Flash Fire "
                         f"(llm_dir={cached.get('direction')} "
                         f"signal_dir={signal.prediction}) — falling back "
                         f"to score consensus: {cached.get('reasoning','')[:80]}")
                flash_fire = False
                early_flash = False

        # ── HFT #6: Slippage bail — penalise score if ask rises w/o prob gain ──
        # If between evals the CLOB ask climbed >N cents but our probability
        # didn't follow, the edge is silently evaporating → subtract score.
        if self._SLIPPAGE_BAIL_ENABLED and price_cents is not None:
            prev_ask = self._last_sniper_ask_cents
            prev_prob = self._last_sniper_prob
            if (prev_ask is not None and prev_prob is not None):
                ask_delta = price_cents - prev_ask
                prob_delta = signal.confidence - prev_prob
                if (ask_delta >= self._SLIPPAGE_BAIL_CENTS and prob_delta < 1.0
                        and self._consensus_score > self._CONSENSUS_SCORE_FLIP):
                    self._consensus_score -= 1
                    log.info(f"[SLIPPAGE BAIL] ask {prev_ask:.1f}→{price_cents:.1f}¢ "
                             f"(+{ask_delta:.1f}) prob Δ={prob_delta:+.0f} → "
                             f"score -1 (now {self._consensus_score})")
            self._last_sniper_ask_cents = price_cents
            self._last_sniper_prob = signal.confidence

        # ── HFT #4: Microprice EARLY-FLASH (bypass first_eval_delay/age gate) ──
        # Computes Polymarket microprice shift on the signal's side. If the
        # book's weighted mid moved >Δ cents in our direction in <15s, we
        # already have tape confirmation and can fire without waiting.
        microprice_flash = False
        if self._MICROPRICE_FLASH_ENABLED and elapsed < 60:
            try:
                poly_q = self.polymarket.get_current_btc5min_quote()
                if not poly_q.is_stale:
                    if signal.prediction == "UP":
                        bid, ask = poly_q.up_best_bid, poly_q.up_best_ask
                    else:
                        bid, ask = poly_q.down_best_bid, poly_q.down_best_ask
                    if bid > 0 and ask > 0 and ask > bid:
                        microprice = (bid + ask) / 2.0 * 100.0  # cents
                        baseline = getattr(self, "_microprice_baseline", None)
                        if baseline is None:
                            self._microprice_baseline = microprice
                        else:
                            delta = microprice - baseline
                            if delta >= self._MICROPRICE_FLASH_DELTA_CENTS:
                                log.info(f"[MICROPRICE FLASH] side={signal.prediction} "
                                         f"mid {baseline:.1f}→{microprice:.1f}¢ "
                                         f"(+{delta:.1f}) @ t={elapsed:.0f}s — firing")
                                microprice_flash = True
            except Exception as e:
                log.debug(f"[MICROPRICE] check failed: {e}")

        # ── HFT #3: CVD imbalance shock — tape conviction bypass ──
        cvd_shock = False
        if self._CVD_SHOCK_ENABLED and self._consensus_direction == signal.prediction:
            try:
                cvd = self.binance.get_cvd_summary(minutes=1)
                imb = float(cvd.get("imbalance_pct", 0) or 0)
                # UP wants buy-side imbalance (+), DOWN wants sell-side (−)
                aligned = ((signal.prediction == "UP" and imb >= self._CVD_SHOCK_IMBALANCE_PCT)
                           or (signal.prediction == "DOWN" and imb <= -self._CVD_SHOCK_IMBALANCE_PCT))
                if aligned and self._consensus_score >= 1:
                    log.info(f"[CVD SHOCK] imbalance={imb:+.0f}% aligns with "
                             f"{signal.prediction} @ t={elapsed:.0f}s — "
                             f"bypassing age gate")
                    cvd_shock = True
            except Exception as e:
                log.debug(f"[CVD SHOCK] check failed: {e}")

        # ── HFT #7: Regime-aware fire score + #2: dynamic age gate (EV hot) ──
        fire_override = None
        age_override = None
        if self._REGIME_AWARE_FIRE:
            try:
                gex = self.gex.get_gex() or {}
                regime = gex.get("gex_regime")
                if regime == "SHORT_GAMMA":
                    fire_override = self._REGIME_SHORT_GAMMA_FIRE
                elif regime == "LONG_GAMMA":
                    fire_override = self._REGIME_LONG_GAMMA_FIRE
            except Exception:
                pass
        if ev is not None and ev >= self._MIN_CONSENSUS_AGE_EV_THRESHOLD:
            age_override = self._MIN_CONSENSUS_AGE_HOT

        # Check consensus trigger (Flash Fire variants force trigger=True)
        score_fired = self._check_sniper_consensus(
            elapsed, fire_score_override=fire_override, age_override=age_override,
        )
        triggered = flash_fire or early_flash or score_fired or cvd_shock or microprice_flash
        if triggered:
            self._bet_evaluation_done = True
            n_evals = len(self._window_evaluations)
            time_to_bet = elapsed
            had_flip = any(
                self._window_evaluations[i]["prediction"] != self._window_evaluations[i - 1]["prediction"]
                for i in range(1, n_evals)
            )
            if microprice_flash:
                trigger_tag = "MICROPRICE-FLASH"
            elif cvd_shock:
                trigger_tag = "CVD-SHOCK"
            elif early_flash:
                trigger_tag = "EARLY-FLASH"
            elif flash_fire:
                trigger_tag = "FLASH-FIRE"
            else:
                trigger_tag = "CONSENSUS"
            self._last_trigger_type = trigger_tag
            self._sniper_trigger_ready_ts = time.time()
            if self._consensus_idx < 0:
                self._consensus_idx = n_evals - 1
            log.info(f"SNIPER TRIGGER [{trigger_tag}] — {n_evals} evals, "
                     f"t={time_to_bet:.0f}s")
            # Save sniper metadata to DB for ML training
            try:
                conn = db._get_conn()
                conn.execute(
                    """UPDATE predictions SET sniper_eval_count=?,
                       sniper_time_to_bet_sec=?, sniper_flipped=?
                       WHERE window_id=? AND ai_model=?""",
                    (n_evals, round(time_to_bet, 1), int(had_flip),
                     self.window.window_id, get_primary_agent_id()),
                )
                conn.commit()
            except Exception as e:
                log.error(f"DB write failed (sniper metadata): {e}")

            # ── ML Judge gate: recycle rejections into explore mode ──
            # Per architectural directive: no Judge veto goes to pure ghost.
            # Every rejection becomes a mandatory $1 exploration ticket so
            # the ML dataset keeps learning "who was right" on contested
            # windows. Swarm rides along automatically via the Grito de
            # Fuego broadcast at the tail of _place_auto_bet.
            if self._ml_skipped:
                prob_str = (f"{self._ml_oracle_prob:.1%}"
                            if self._ml_oracle_prob is not None else "N/A")
                log.info(f"[JUDGE] P(win)={prob_str} < "
                         f"{ML_CONFIDENCE_THRESHOLD:.0%} — reciclando a "
                         f"force_explore=$1.00 (data harvest)")
                self._place_auto_bet(
                    signal,
                    is_early=(eval_minute <= 2),
                    force_explore=True,
                )
            else:
                self._place_auto_bet(signal, is_early=(eval_minute <= 2))

    def _check_sniper_consensus(self, elapsed: float,
                                fire_score_override: int | None = None,
                                age_override: float | None = None) -> bool:
        """Score-based consensus rule.

        Each eval that confirms the current _consensus_direction adds +1
        to the score. Each contradicting eval subtracts 1. If the score
        falls below ``consensus_score_flip`` (e.g. -2), the direction
        flips and the score resets to 1 in the new direction.

        Fire is triggered when score >= ``consensus_score_fire`` AND
        at least ``min_consensus_age_sec`` seconds have elapsed since
        window open. The age gate prevents super-early bets on very
        volatile opens.
        """
        evals = self._window_evaluations
        if not evals:
            return False

        curr = evals[-1]

        # Seed direction on very first tracked eval
        if self._consensus_direction is None:
            self._consensus_direction = curr["prediction"]
            self._consensus_score = 1
            log.info(f"[CONSENSUS seed] dir={self._consensus_direction} "
                     f"score=1/{self._CONSENSUS_SCORE_FIRE} age={elapsed:.0f}s")
            return False

        if curr["prediction"] == self._consensus_direction:
            self._consensus_score += 1
        else:
            self._consensus_score -= 1
            if self._consensus_score <= self._CONSENSUS_SCORE_FLIP:
                old_dir = self._consensus_direction
                self._consensus_direction = curr["prediction"]
                self._consensus_score = 1
                log.info(f"[CONSENSUS FLIP] {old_dir}→"
                         f"{self._consensus_direction} "
                         f"(score reset to 1) age={elapsed:.0f}s")

        eff_age = age_override if age_override is not None else self._MIN_CONSENSUS_AGE
        eff_fire = (fire_score_override
                    if fire_score_override is not None
                    else self._CONSENSUS_SCORE_FIRE)
        age_ok = elapsed >= eff_age
        fire_ok = self._consensus_score >= eff_fire

        log.info(f"[CONSENSUS score={self._consensus_score}/"
                 f"{eff_fire} dir={self._consensus_direction} "
                 f"age={elapsed:.0f}s last={curr['prediction']}"
                 f"({curr['confidence']}%)]")

        if fire_ok and age_ok:
            self._consensus_idx = len(evals) - 1
            log.info(f"SNIPER CONSENSUS FIRE — dir={self._consensus_direction} "
                     f"score={self._consensus_score} age={elapsed:.0f}s")
            return True
        return False

    def _summarize_window_evals(self) -> dict:
        """Return a compact diagnostic snapshot of the current window's evals.

        Used by TIME-STOP logging and (optionally) get_state telemetry.
        """
        evals = self._window_evaluations
        if not evals:
            return {
                "n": 0, "max_conf": 0, "min_conf": 0,
                "num_flips": 0, "last_direction": None,
                "last_score": self._consensus_score,
            }
        confs = [e["confidence"] for e in evals]
        num_flips = sum(
            1 for i in range(1, len(evals))
            if evals[i]["prediction"] != evals[i - 1]["prediction"]
        )
        return {
            "n": len(evals),
            "max_conf": max(confs),
            "min_conf": min(confs),
            "num_flips": num_flips,
            "last_direction": evals[-1]["prediction"],
            "last_score": self._consensus_score,
        }

    def _try_last_minute_rescue(self, elapsed: float) -> bool:
        """Try to fire a bet at cutoff using the best eval of the window.

        Selects the highest-confidence eval that (a) matches the final
        consensus direction (if any) and (b) meets confidence + EV floors.
        Called ONCE per window, right before Meta-Swarm Override.

        Returns True if the rescue fired a bet, False otherwise.
        """
        evals = self._window_evaluations
        if not evals or self.current_signal is None:
            return False

        # Bias toward the consensus direction if we have one; otherwise
        # fall back to the overall max-confidence eval (any direction).
        target_dir = self._consensus_direction
        pool = ([e for e in evals if e["prediction"] == target_dir]
                if target_dir else list(evals))
        if not pool:
            pool = list(evals)

        best = max(pool, key=lambda e: e["confidence"])

        # ── Variant A: LLM veto gate before rescue ──────────
        # Ask the primary LLM to sanity-check the rescue. If it vetoes,
        # return False so the caller falls through to Meta-Swarm Override.
        if self._VARIANT_A_ENABLED:
            cached = self._consume_second_opinion()
            if cached is None:
                poly_snap = self._poly_quote_snapshot()
                mkt_snap = self._market_context_snapshot()
                ta_snap = dict(self.current_ta) if self.current_ta else {}
                current_price_snap = (float(self.window.current_price)
                                      if self.window.current_price
                                      else float(self.window.open_price))
                open_price_snap = float(self.window.open_price)
                time_left_snap = self._time_left()
                # Release lock for the sync LLM call
                self._lock.release()
                try:
                    op = self.ai.quick_second_opinion(
                        current_price=current_price_snap,
                        open_price=open_price_snap,
                        ta=ta_snap,
                        elapsed_sec=elapsed,
                        time_left_sec=time_left_snap,
                        poly_quote=poly_snap,
                        source="A",
                        context=mkt_snap,
                    )
                finally:
                    self._lock.acquire()
                if self.window.window_id is not None and not op.get("error"):
                    self._second_opinion = {
                        **op,
                        "arrived_at": time.time(),
                        "window_id": self.window.window_id,
                    }
                    cached = self._second_opinion
            if cached and cached.get("veto"):
                log.info(f"[2nd-OP:A] VETO rescue "
                         f"(llm_dir={cached.get('direction')} "
                         f"best_dir={best['prediction']}) — fallthrough "
                         f"to Meta-Swarm: {cached.get('reasoning','')[:80]}")
                return False
            # Direction-aligned: clamp best confidence to min(llm, best)
            if cached and cached.get("direction") == best["prediction"]:
                clamped = min(int(cached.get("confidence", 0)),
                              int(best["confidence"]))
                if clamped != best["confidence"]:
                    log.info(f"[2nd-OP:A] clamp rescue conf "
                             f"{best['confidence']}% → {clamped}% "
                             f"(llm={cached.get('confidence')})")
                    best = dict(best)
                    best["confidence"] = clamped

        if best["confidence"] < self._LAST_MIN_RESCUE_CONF:
            log.info(f"[RESCUE] skip — best_conf={best['confidence']}% < "
                     f"{self._LAST_MIN_RESCUE_CONF}% floor")
            return False

        # Build a signal snapshot aligned with the rescued eval's direction
        # so EV is computed against the right Polymarket side.
        snap = self.current_signal
        snap.prediction = best["prediction"]
        snap.confidence = best["confidence"]
        ev, price_cents, payout_ratio = self._compute_polymarket_ev(snap)
        if ev is None or ev < self._LAST_MIN_RESCUE_EV:
            log.info(f"[RESCUE] skip — ev={ev} < {self._LAST_MIN_RESCUE_EV}")
            return False

        log.info(f"[RESCUE] FIRE — dir={best['prediction']} "
                 f"conf={best['confidence']}% ev={ev:.2f} "
                 f"price={price_cents:.1f}¢ elapsed={elapsed:.0f}s")
        self._last_trigger_type = "RESCUE"
        self._consensus_idx = evals.index(best)
        self._bet_evaluation_done = True

        # Stamp sniper metadata so ML pipeline sees the rescue decision.
        try:
            n_evals = len(evals)
            had_flip = any(
                evals[i]["prediction"] != evals[i - 1]["prediction"]
                for i in range(1, n_evals)
            )
            conn = db._get_conn()
            conn.execute(
                """UPDATE predictions SET sniper_eval_count=?,
                   sniper_time_to_bet_sec=?, sniper_flipped=?
                   WHERE window_id=? AND ai_model=?""",
                (n_evals, round(elapsed, 1), int(had_flip),
                 self.window.window_id, get_primary_agent_id()),
            )
            conn.commit()
        except Exception as e:
            log.error(f"DB write failed (rescue metadata): {e}")

        self._place_auto_bet(snap, is_early=False)
        return True

    def _check_meta_swarm_override(self, veto_reason: str) -> bool:
        """Meta-Swarm Consensus Override — Francotirador Override.

        When the primary agent vetoes a bet (time-stop, EV ghosting, etc.),
        checks if ALL enabled secondary swarm agents are unanimous with high
        confidence.  If so, constructs a synthetic AISignal and fires
        ``_place_auto_bet`` on behalf of the swarm consensus.

        Only counts agents explicitly enabled (``enabled: true``) in
        ``dynamic_rules.json`` — inactive/off agents are ignored entirely.

        Returns True if override fired, False otherwise.
        """
        # ── Gate: feature toggle ──
        ms_enabled = rules.get("meta_swarm", "enabled", True)
        if not ms_enabled:
            return False

        min_conf = int(rules.get("meta_swarm", "min_confidence", 70))
        min_agents = int(rules.get("meta_swarm", "min_agents", 2))

        # ── Collect armed signals from ENABLED secondaries only ──
        secondary_cfgs = swarm.get_secondary_agents()  # already filters enabled=true
        if len(secondary_cfgs) < min_agents:
            return False

        secondary_ids = {a["agent_id"] for a in secondary_cfgs}
        armed: list[dict] = []
        for aid, blob in self._swarm_signals.items():
            if aid in secondary_ids and not blob.get("fired", False):
                armed.append(blob)

        if len(armed) < min_agents or len(armed) < len(secondary_ids):
            # Not all enabled secondaries have armed signals — no unanimity
            log.debug(f"[Meta-Swarm] {len(armed)}/{len(secondary_ids)} armed — "
                      f"need all enabled agents, skipping override")
            return False

        # ── Unanimity check: all same direction, all >= min_confidence ──
        directions = {b["prediction"] for b in armed}
        if len(directions) != 1:
            preds = ", ".join(f'{b["display_name"]}={b["prediction"]}' for b in armed)

            # ── Bull-vs-Bear debate fallback (Vibe-Trading inspired) ──
            debate_on = bool(rules.get("meta_swarm", "debate_enabled", False))
            if debate_on:
                bull = [{"display_name": b["display_name"],
                         "confidence": b["confidence"],
                         "reasoning": b.get("reasoning", "")}
                        for b in armed if b["prediction"] == "UP"]
                bear = [{"display_name": b["display_name"],
                         "confidence": b["confidence"],
                         "reasoning": b.get("reasoning", "")}
                        for b in armed if b["prediction"] == "DOWN"]
                # Short market context for the judge (best-effort).
                try:
                    cs = self.current_signal
                    ctx_line = (
                        f"primary_pred={getattr(cs,'prediction','?')} "
                        f"primary_conf={getattr(cs,'confidence','?')}% "
                        f"veto_reason={veto_reason}"
                    )
                except Exception:
                    ctx_line = f"veto_reason={veto_reason}"

                verdict = self.ai.bull_bear_debate(bull, bear, ctx_line)
                j_dir = verdict.get("direction")
                j_conf = int(verdict.get("confidence", 0))
                log.info(f"[Meta-Swarm] Debate verdict: {j_dir} {j_conf}% "
                         f"veto={verdict.get('veto')} — "
                         f"reason='{verdict.get('reason','')[:120]}'")

                if verdict.get("veto") or j_dir not in ("UP", "DOWN"):
                    log.info(f"[Meta-Swarm] No unanimity ({preds}) + debate veto")
                    return False
                if j_conf < min_conf:
                    log.info(f"[Meta-Swarm] Debate verdict {j_dir} {j_conf}% < "
                             f"min_conf {min_conf} — blocked")
                    return False

                agent_names = ", ".join(b["display_name"] for b in armed)
                log.info(f"[Meta-Swarm] DEBATE OVERRIDE — judge={j_dir} {j_conf}% "
                         f"on split swarm [{preds}] bypassing primary veto "
                         f"({veto_reason}) | agents: [{agent_names}]")

                synth = AISignal(
                    prediction=j_dir,
                    confidence=j_conf,
                    reasoning=(f"[Meta-Swarm Debate] Judge={j_dir} {j_conf}%. "
                               f"Split swarm: {preds}. "
                               f"Reason: {verdict.get('reason','')[:280]}"),
                    news_summary="",
                    risk_score="MEDIUM",
                    suggested_bet_pct=2.0,
                    timestamp=time.time(),
                    signal_quality="NORMAL",
                )
                self.current_signal = synth
                self._bet_evaluation_done = True
                self._last_trigger_type = "DEBATE"
                self._place_auto_bet(synth, is_early=False)
                return True

            log.info(f"[Meta-Swarm] No unanimity ({preds}) — override blocked")
            return False

        unanimous_dir = directions.pop()  # "UP" or "DOWN"
        low_conf = [b for b in armed if b["confidence"] < min_conf]
        if low_conf:
            weak = ", ".join(f'{b["display_name"]}={b["confidence"]}%' for b in low_conf)
            log.info(f"[Meta-Swarm] Unanimity {unanimous_dir} but low confidence "
                     f"({weak} < {min_conf}%) — override blocked")
            return False

        # ── All checks passed — FIRE Francotirador Override ──
        avg_conf = int(sum(b["confidence"] for b in armed) / len(armed))
        agent_names = ", ".join(b["display_name"] for b in armed)
        log.info(f"[Meta-Swarm] OVERRIDE — {len(armed)} enabled swarm agents "
                 f"unanimous {unanimous_dir} (avg conf {avg_conf}%), "
                 f"bypassing primary veto ({veto_reason}) | "
                 f"agents: [{agent_names}]")

        # Build synthetic AISignal from swarm consensus
        synth_signal = AISignal(
            prediction=unanimous_dir,
            confidence=avg_conf,
            reasoning=(f"[Meta-Swarm Override] {len(armed)} secondary agents "
                       f"unanimous {unanimous_dir} (avg {avg_conf}%). "
                       f"Primary veto: {veto_reason}"),
            news_summary="",
            risk_score="MEDIUM",
            suggested_bet_pct=2.0,
            timestamp=time.time(),
            signal_quality="NORMAL",
        )

        # Override the current signal so _place_auto_bet uses the consensus
        self.current_signal = synth_signal
        self._bet_evaluation_done = True
        self._last_trigger_type = "OVERRIDE"
        self._place_auto_bet(synth_signal, is_early=False)
        return True

    def _evaluate_oracle(self, ml_features: dict) -> tuple[float | None, bool]:
        """Run the XGBoost Judge on the current prediction features.

        Returns (probability_of_win, should_skip).
        If model is not loaded or Judge is disabled, returns (None, False).
        """
        if _judge_model is None or not cfg.ENABLE_ML_JUDGE:
            return None, False

        try:
            import pandas as pd
            # Use model's actual features (auto-detected at load time)
            features = _judge_model_features or _JUDGE_FEATURES
            row = {f: ml_features.get(f, 0) or 0 for f in features}
            # Encode volatility_regime: LOW=0, NORMAL=1, HIGH=2
            _vol_map = {"LOW": 0, "NORMAL": 1, "HIGH": 2}
            if "volatility_regime" in row and isinstance(row["volatility_regime"], str):
                row["volatility_regime"] = _vol_map.get(row["volatility_regime"], 1)
            # Encode liq_cluster_side: ABOVE=+1, BELOW=-1, AT_SPOT/NEUTRAL=0.
            if "liq_cluster_side_enc" in row:
                side_raw = ml_features.get("liq_cluster_side") or ""
                _side_map = {"ABOVE": 1, "BELOW": -1, "AT_SPOT": 0, "NEUTRAL": 0}
                row["liq_cluster_side_enc"] = _side_map.get(str(side_raw).upper(), 0)
            X = pd.DataFrame([row], columns=features)
            X = X.fillna(0)
            prob = float(_judge_model.predict_proba(X)[0, 1])  # P(win)
            skip = prob < ML_CONFIDENCE_THRESHOLD

            log.info(f"[JUDGE] P(win)={prob:.3f} | "
                     f"threshold={ML_CONFIDENCE_THRESHOLD} | "
                     f"{'RECHAZADO' if skip else 'APROBADO'}")
            return prob, skip
        except Exception as e:
            log.warning(f"[JUDGE] Error en inferencia: {e}")
            return None, False

    def _build_rl_context(self, agent_id: str,
                          ml_features: dict | None,
                          ta: dict | None,
                          signal=None,
                          initial_balance: float | None = None) -> dict:
        if initial_balance is None:
            initial_balance = self._default_initial_balance()
        """Assemble the observation context consumed by the RL portfolio
        allocator inside ``RiskManager.calculate_agent_bet``.

        Safe-fail: never raises — any missing piece is replaced with a sane
        default so the dispatcher can still route to the RL path without
        crashing the tick loop. If every piece is missing the RL wrapper
        will just see a neutral observation and (in the worst case)
        fall back to classic Kelly on its own.
        """
        mf = ml_features or {}
        ta_d = ta or {}

        # ── judge_prob: ML Oracle > calibrated conf > raw confidence ──
        judge_prob = mf.get("ml_oracle_prob")
        if judge_prob is None and signal is not None:
            try:
                judge_prob = float(getattr(signal, "confidence", 50)) / 100.0
            except Exception:
                judge_prob = 0.5
        if judge_prob is None:
            judge_prob = 0.5

        # ── HMM snapshot: ml_features first, then engine cache ──
        hmm = mf.get("_hmm_regime") or self._current_regime

        # ── atr_pct: ml_features > ta dict ──
        atr_pct = mf.get("atr_pct")
        if atr_pct is None:
            atr_pct = ta_d.get("atr_pct", 0.03)

        # ── last 5 realized pnls for this agent (safe-fail SQL) ──
        last_pnls: list[float] = []
        try:
            conn = db._get_conn()
            rows = conn.execute(
                """SELECT profit FROM bets
                   WHERE ai_model=? AND won IS NOT NULL AND is_ghost=0
                   ORDER BY window_id DESC
                   LIMIT 5""",
                (agent_id,),
            ).fetchall()
            # Oldest → newest (build_observation expects chronological order)
            last_pnls = [float(r["profit"] or 0.0) for r in reversed(rows)]
        except Exception as e:
            log.debug(f"[RL] last_pnls fetch failed for {agent_id}: {e}")
            last_pnls = []

        return {
            "judge_prob":      float(judge_prob),
            "hmm":             hmm,
            "initial_balance": float(initial_balance),
            "last_pnls":       last_pnls,
            "atr_pct":         float(atr_pct or 0.03),
        }

    def _place_auto_bet(self, signal, is_early: bool = False,
                        force_explore: bool = False):
        """Place auto-bet or ghost trade + orchestrate the swarm.

        HFT latency metric: elapsed ms from `_sniper_trigger_ready_ts` (the
        moment consensus/flash/CVD/microprice trigger fired) to the start of
        this method is recorded into Prometheus `sniper_fire_latency_ms`.
        """
        _fire_ready = getattr(self, "_sniper_trigger_ready_ts", 0.0)
        if _fire_ready > 0:
            try:
                lat_ms = (time.time() - _fire_ready) * 1000.0
                prom_metrics.record_sniper_fire_latency(
                    self._last_trigger_type or "UNKNOWN", lat_ms,
                )
            except Exception:
                pass
            self._sniper_trigger_ready_ts = 0.0
        if self.current_bet or self._window_bet_placed:
            log.info("SKIP BET: Ya hay apuesta activa en esta ventana")
            return

        trust_info = self.ai.get_trust_info()
        side = signal.prediction

        # ── Fetch real Polymarket token price (CLOB executable) ──
        # Snapshot ONCE — same quote is passed to the swarm broadcast so
        # every agent rides the primary's exact EV context.
        poly_quote = self.polymarket.get_current_btc5min_quote()
        odds_source = "internal_fallback"
        if not poly_quote.is_stale and poly_quote.up_price > 0:
            # Use CLOB executable price (best_ask via /price?side=BUY)
            # Keep decimal precision: 0.535 → 53.5 cents (not rounded to int)
            raw_token = (poly_quote.up_price if side == "UP"
                         else poly_quote.down_price)
            odds = round(max(5.0, min(95.0, raw_token * 100)), 1)
            odds_source = "clob_live"
            log.info(f"[POLYMARKET] CLOB quote: UP=${poly_quote.up_price:.3f} "
                     f"DOWN=${poly_quote.down_price:.3f} | "
                     f"Using {side}@{odds}c for bet")
        else:
            # Fallback: internal odds + simulated spread penalty (2c)
            # This prevents unrealistically cheap fills in paper mode
            raw_odds = (self.window.up_odds if side == "UP"
                        else self.window.down_odds)
            spread_penalty = 2  # simulate CLOB spread: buyer pays 2c above mid
            odds = max(5.0, min(95.0, float(raw_odds + spread_penalty)))
            log.info(f"[POLYMARKET] No CLOB quote — internal odds {raw_odds}c "
                     f"+ {spread_penalty}c spread = {odds}c")

        # ── PTB Filter: mathematical strike-vs-spot gatekeeper ──
        # If the strike price is known and the gap is unreachable,
        # recycle to force_explore (never ghost — data harvest).
        if poly_quote.strike_price > 0 and not force_explore:
            from .execution.ptb_filter import evaluate_ptb
            spot_now = self.price_feed.get_price()
            atr_pct = self.current_ta.get("atr_pct", 0.003)
            ttl = self._time_left()
            ptb_v = evaluate_ptb(side, poly_quote.strike_price,
                                 spot_now, ttl, atr_pct)
            if ptb_v.action == "block":
                log.info(f"[PTB] Blocked → force_explore=True | {ptb_v.reason}")
                force_explore = True
            elif ptb_v.action == "penalize":
                log.info(f"[PTB] Penalized → sizing x{ptb_v.sizing_factor} | "
                         f"{ptb_v.reason}")
                # Store penalty for risk sizing below
                self._ptb_sizing_factor = ptb_v.sizing_factor
            else:
                self._ptb_sizing_factor = 1.0

        # ── EV Gate: reject mathematically unprofitable bets ──────
        # If the bet has negative expected value (EV < min_ev) or the
        # payout ratio is too small (token too expensive), ghost-trade
        # it: zero cost, full ML data capture.
        if not force_explore and odds > 0:
            _ev_enabled = rules.get("ev_gate", "enabled", True)
            if _ev_enabled:
                _min_ev = rules.get("ev_gate", "min_ev", 1.02)
                _min_payout = rules.get("ev_gate", "min_payout", 1.15)
                _coinflip_margin = int(rules.get("ev_gate",
                                                   "coinflip_margin_cents", 5))
                _payout_ratio = 100.0 / odds
                _ev = (signal.confidence / 100.0) * _payout_ratio

                # Coinflip Zone: odds too close to 50¢ = no edge
                # At 50¢ Polymarket fee is MAXIMUM (p*(1-p) peaks)
                # and payout is only 2x, needing >52% WR to breakeven.
                if abs(odds - 50) <= _coinflip_margin:
                    ghost_reason = (
                        f"[EV GATE] Coinflip zone: odds={odds}c "
                        f"(±{_coinflip_margin}c of 50¢, no edge)")
                    log.info(ghost_reason)
                    if self._watchful_hold_defer(side, ghost_reason, odds,
                                                  odds_source, is_early,
                                                  poly_quote, "coinflip"):
                        return
                    self._save_ghost_trade(side, odds, ghost_reason, is_early,
                                           odds_source=odds_source)
                    self._broadcast_swarm_fire(
                        poly_quote=poly_quote, is_early=is_early,
                        primary_ghosted=True, force_explore=False)
                    return
                if _payout_ratio < _min_payout:
                    ghost_reason = (
                        f"[EV GATE] Payout demasiado bajo: "
                        f"×{_payout_ratio:.2f} < ×{_min_payout:.2f} "
                        f"(odds={odds}c, conf={signal.confidence}%)")
                    log.info(ghost_reason)
                    if self._watchful_hold_defer(side, ghost_reason, odds,
                                                  odds_source, is_early,
                                                  poly_quote, "low_payout"):
                        return
                    self._save_ghost_trade(side, odds, ghost_reason, is_early,
                                           odds_source=odds_source)
                    self._broadcast_swarm_fire(
                        poly_quote=poly_quote, is_early=is_early,
                        primary_ghosted=True, force_explore=False)
                    return
                if _ev < _min_ev:
                    ghost_reason = (
                        f"[EV GATE] EV negativo: {_ev:.3f} < {_min_ev} "
                        f"(conf={signal.confidence}%, odds={odds}c, "
                        f"payout=×{_payout_ratio:.2f})")
                    log.info(ghost_reason)
                    if self._watchful_hold_defer(side, ghost_reason, odds,
                                                  odds_source, is_early,
                                                  poly_quote, "ev_low"):
                        return
                    self._save_ghost_trade(side, odds, ghost_reason, is_early,
                                           odds_source=odds_source)
                    self._broadcast_swarm_fire(
                        poly_quote=poly_quote, is_early=is_early,
                        primary_ghosted=True, force_explore=False)
                    return

        # ── Circuit Breaker gate (Black Swan news) ──
        # Hard stop — even explore mode cannot bypass a Black Swan.
        # Swarm still receives a ghost broadcast for data harvesting.
        if sentiment.is_circuit_breaker_active():
            cb_status = sentiment.get_circuit_breaker_status()
            if getattr(cfg, 'PAPER_TRADING_MODE', True):
                log.warning(f"[CIRCUIT BREAKER] Paper mode — apuesta PERMITIDA "
                            f"con advertencia (reason: {cb_status['reason'][:80]})")
                # Paper mode: let the bet through, just log the warning
            else:
                ghost_reason = (f"Circuit Breaker activo: {cb_status['reason']} "
                                f"({cb_status['remaining_seconds']}s restantes)")
                log.warning(f"[CIRCUIT BREAKER] Apuesta bloqueada — {ghost_reason}")
                self._save_ghost_trade(side, odds, ghost_reason, is_early,
                                       odds_source=odds_source)
                self._broadcast_swarm_fire(
                    poly_quote=poly_quote,
                    is_early=is_early,
                    primary_ghosted=True,
                    force_explore=False,
                )
                return

        # ── Trust gate: recycle low-trust rejections into explore mode ──
        # Per architectural directive: "NO anules el flujo principal. Deriva
        # a _place_auto_bet con force_explore=True". Every rejection becomes
        # a mandatory $1 exploration ticket for the ML dataset.
        # NOTE: PAPER_TRADING_MODE guard removed — fake money must ALWAYS
        # recycle trust vetoes into explore, never skip the data harvest.
        if not trust_info["bet_allowed"] and not force_explore:
            log.info(f"[EXPLORE RECYCLE] Trust score bajo "
                     f"({trust_info['trust_score']}) — reciclando a "
                     f"force_explore=$1.00 (data harvest)")
            force_explore = True

        tag = "EARLY" if is_early else "DELAYED"
        if force_explore:
            tag += "/EXPLORE"

        # Get primary agent balance (live on-chain if toggle ON, paper otherwise)
        agent_balance = self._resolve_agent_balance(get_primary_agent_id())
        if getattr(cfg, "LIVE_DYNAMIC_SIZING", False):
            log.info(f"[Live Sizing] Primary Kelly input balance = "
                     f"${agent_balance:.2f} "
                     f"(toggle=ON, executor_connected="
                     f"{bool(self.executor and self.executor.connected)})")

        # ── Sizing: explore ($1 flat) or Kelly via risk manager ──
        if force_explore:
            should = True
            amount = 1.00
            reason = "EXPLORE_OVERRIDE: $1.00 flat (Kelly bypassed)"
        else:
            # Build RL observation context (HMM comes from engine cache)
            primary_rl_ctx = self._build_rl_context(
                agent_id=get_primary_agent_id(),
                ml_features=None,             # use self._current_regime cache
                ta=self.current_ta,
                signal=signal,
                initial_balance=self._default_initial_balance(),
            )
            should, amount, reason = self.risk.calculate_agent_bet(
                signal, agent_balance, odds,
                trust_score=trust_info["trust_score"],
                agent_id=get_primary_agent_id(),
                rl_context=primary_rl_ctx,
            )

            # ── PTB sizing penalty (if penalized but not blocked) ──
            ptb_factor = getattr(self, "_ptb_sizing_factor", 1.0)
            if should and ptb_factor < 1.0:
                original = amount
                amount = max(1.0, round(amount * ptb_factor, 2))
                reason += f" [PTB penalty {ptb_factor:.0%}: ${original:.2f}→${amount:.2f}]"

            # ── Supreme Explore Override (Data Harvest) ──
            # In Paper Trading Mode, NADA debe ser un "ghost puro" por falta
            # de capital o riesgo. Todo veto algorítmico (Kelly, drawdown,
            # min-balance, low-confidence) se intercepta y recicla hacia
            # exploración de recolección de ML. Black Swan sigue siendo
            # min-balance, low-confidence) se intercepta y recicla hacia
            # exploración de recolección de ML. Black Swan sigue siendo
            # hard stop (ese gate corre antes de este bloque).
            if (not should and getattr(cfg, 'PAPER_TRADING_MODE', True)
                    and getattr(cfg, "ENABLE_EXPLORE_OVERRIDE", True)):
                log.info(f"[EXPLORE RECYCLE] Veto del Risk Manager "
                         f"({reason}) superado en PAPER MODE. "
                         f"Hackeando hacia exploración de recolección: $1.00.")
                should = True
                amount = 1.00
                force_explore = True
                reason = f"EXPLORE_OVERRIDE: Risk Veto Ignorado ({reason})"
                tag = tag if "EXPLORE" in tag else f"{tag}/EXPLORE"

        log.info(f"Risk({tag}): {reason}")

        if should:
            primary_strategy = ("smc_liquidity|explore"
                                if force_explore else "smc_liquidity")
            self.current_bet = Bet(
                side=side, amount=amount, price_cents=odds,
                open_price=self.window.open_price,
                window_id=self.window.window_id,
                timestamp=time.time(),
            )
            self._window_bet_placed = True
            log.info(f"{tag} BET: {side} ${amount:.2f} @ {odds}c "
                     f"(trust:{trust_info['trust_score']:.0f}, "
                     f"conf:{signal.confidence}%)")
            try:
                db.save_bet(self.window.window_id, side, amount,
                            odds, self.window.open_price,
                            is_manual=False, is_early=is_early,
                            ai_model=get_primary_agent_id(),
                            strategy_name=primary_strategy,
                            odds_source=odds_source)
            except Exception as e:
                log.error(f"DB write failed (save_bet): {e}")

            # ── Web3 CLOB Execution (if executor is connected) ──
            # Submits the real order to Polymarket CLOB via py-clob-client.
            # In offline_sim mode, self.executor is None — this block is skipped
            # and the legacy calc_polymarket_pnl model handles P&L at resolve.
            if getattr(cfg, 'PAPER_TRADING_MODE', True):
                log.info(f"[EXEC] PAPER MODE ACTIVO: Ejecución Web3 evadida. Apuesta simulada localmente.")
                # We skip real execution
            elif self.executor and self.executor.connected:
                try:
                    # Polymarket L2 (the CLOB) trades OUTCOME TOKENS, not
                    # condition_ids. Every order is keyed on the UP or DOWN
                    # CTF token UUID. condition_id is only useful for market
                    # resolution and redemption, never for order placement.
                    token_id = (poly_quote.up_token_id if side == "UP"
                                else poly_quote.down_token_id)
                    if token_id:
                        # Reason string logged alongside the native order hash
                        # so operators can grep execution history.
                        exec_reason = (
                            f"{primary_strategy} | {side} | conf={signal.confidence}% | "
                            f"trust={trust_info['trust_score']:.0f} | "
                            f"agent={get_primary_agent_id()}"
                        )
                        # ── Maker Ladder (FORZADO) vs Market order ──
                        # Default path is ALWAYS the Maker Ladder: we
                        # distribute the Kelly budget as small GTC limit
                        # orders just below the mid price, so a thin book
                        # can never throw the "no match" FOK exception we
                        # hit in production. Market orders are only used
                        # when `force_explore=True` (the $1 data-harvest
                        # ticket, where latency beats price).
                        from .execution.ladder import is_ladder_enabled, build_ladder
                        # ── Spread toxicity check (A4 megarefactor) ──
                        # If spread > threshold, force ladder even on
                        # explore and penalize sizing.
                        _ask_c = (poly_quote.up_best_ask * 100
                                  if side == "UP"
                                  else poly_quote.down_best_ask * 100)
                        _bid_c = (poly_quote.up_best_bid * 100
                                  if side == "UP"
                                  else poly_quote.down_best_bid * 100)
                        _spread_c = (_ask_c - _bid_c
                                     if _ask_c > 0 and _bid_c > 0
                                     else 0)
                        _toxic_spread_threshold = float(
                            rules.get("ladder", "toxic_spread_cents", 5.0))
                        _toxic = _spread_c >= _toxic_spread_threshold
                        if _toxic:
                            penalty = float(rules.get(
                                "ladder", "toxic_spread_sizing_penalty", 0.5))
                            amount = round(amount * penalty, 2)
                            log.warning(
                                f"[SPREAD] Toxic spread detected: "
                                f"{_spread_c:.1f}c >= {_toxic_spread_threshold}c | "
                                f"sizing penalized x{penalty} → ${amount:.2f} | "
                                f"forcing Maker Ladder")
                        ladder_forced = (
                            (is_ladder_enabled() and not force_explore)
                            or _toxic  # toxic spread overrides explore
                        )
                        if ladder_forced:
                            # Reuse spread-check values for ladder anchor
                            best_ask_c = _ask_c
                            best_bid_c = _bid_c
                            if best_ask_c <= 0:
                                best_ask_c = odds  # CLOB stale fallback
                            if best_bid_c <= 0:
                                # Synthesize a bid from the penalty-adjusted
                                # odds so the mid-price path still works.
                                best_bid_c = max(1.0, best_ask_c - 2.0)
                            try:
                                ladder_spec = build_ladder(
                                    amount, best_ask_c,
                                    best_bid_cents=best_bid_c,
                                )
                                ladder_result = self.executor.submit_maker_ladder(
                                    token_id, side, amount, ladder_spec.rungs,
                                    reasoning=exec_reason,
                                )
                            except Exception as ladder_err:
                                # NEVER let a ladder blow up the tick loop.
                                log.warning(
                                    f"[LADDER] Ladder submission crashed "
                                    f"(continuing, no fallback): "
                                    f"{ladder_err}"
                                )
                                ladder_result = None

                            if ladder_result and ladder_result.rungs_submitted > 0:
                                fill_rate = (
                                    ladder_result.rungs_submitted
                                    / max(1, len(ladder_spec.rungs))
                                )
                                try:
                                    db.update_bet_exec(
                                        self.window.window_id,
                                        get_primary_agent_id(),
                                        order_id=",".join(ladder_result.order_ids),
                                        fill_price=0.0,  # updated on fill
                                        fill_size=0.0,
                                        exec_status="pending",
                                    )
                                except Exception as e:
                                    log.error(f"DB update_bet_exec ladder: {e}")
                                prom_metrics.record_ladder_fill_rate(fill_rate)
                                log.info(
                                    f"[LADDER] Posted {ladder_result.rungs_submitted}"
                                    f" maker rungs — no Taker order "
                                    f"issued (crash-safe path)"
                                )
                            else:
                                # All rungs failed. DO NOT fall back to a
                                # blind Market/FOK — that's exactly the
                                # path that crashed the loop last time.
                                # Log, keep the paper bet, move on.
                                errs = (ladder_result.errors[:3]
                                        if ladder_result else
                                        ["ladder submission raised"])
                                log.warning(
                                    f"[LADDER] All rungs failed "
                                    f"(no market fallback): {errs}"
                                )
                        else:
                            # force_explore path or ladder disabled —
                            # thin Market order. Still wrapped so a FOK
                            # exception ("no match") can never abort
                            # the tick loop.
                            try:
                                exec_result = self.executor.submit_market(
                                    token_id, side, amount,
                                    reasoning=exec_reason,
                                )
                                if exec_result.success:
                                    try:
                                        db.update_bet_exec(
                                            self.window.window_id,
                                            get_primary_agent_id(),
                                            order_id=exec_result.order_id,
                                            fill_price=exec_result.fill_price,
                                            fill_size=exec_result.fill_size,
                                            exec_status=exec_result.status,
                                        )
                                    except Exception as e:
                                        log.error(f"DB update_bet_exec: {e}")
                                else:
                                    log.warning(f"[EXEC] Market order failed: "
                                                f"{exec_result.error}")
                            except Exception as mkt_err:
                                log.error(f"[EXEC] Market order crashed "
                                          f"(swallowed): {mkt_err}")
                except Exception as e:
                    log.error(f"[EXEC] Submission exception: {e} — offline fallback")

            # ── Register position with SurvivalMonitor ──
            if self.survival:
                try:
                    from .execution.survival import TrackedPosition
                    token_id_surv = (poly_quote.up_token_id if side == "UP"
                                     else poly_quote.down_token_id)
                    entry_p = odds / 100.0  # decimal price
                    tokens = amount / entry_p if entry_p > 0 else 0
                    window_close = (self.window.window_id + 1) * WINDOW_MINUTES * 60
                    self.survival.track(TrackedPosition(
                        token_id=token_id_surv,
                        side=side,
                        entry_price=entry_p,
                        amount_usd=amount,
                        size_tokens=tokens,
                        window_id=self.window.window_id,
                        window_close_ts=window_close,
                        agent_id=get_primary_agent_id(),
                    ))
                except Exception as e:
                    log.warning(f"[SURVIVAL] Track failed: {e}")

            # ── GRITO DE FUEGO: broadcast to the swarm ──
            # Every armed agent fires NOW with the same poly_quote snapshot,
            # same is_early flag, and same explore mode as the primary.
            self._broadcast_swarm_fire(
                poly_quote=poly_quote,
                is_early=is_early,
                primary_ghosted=False,
                force_explore=force_explore,
            )
        else:
            # ── Ghost Trade: risk manager blocked despite explore attempt ──
            # This branch is only reachable if Kelly said no AND we weren't
            # in force_explore mode. Save ghost + broadcast ghost to swarm
            # (interpretation B: data harvesting).
            log.info(f"[GHOST MODE] Riesgo bloqueó apuesta — "
                     f"registrando intención para ML ({tag}: {reason})")
            self._save_ghost_trade(side, odds, reason, is_early,
                                   odds_source=odds_source)
            self._broadcast_swarm_fire(
                poly_quote=poly_quote,
                is_early=is_early,
                primary_ghosted=True,
                force_explore=False,
            )

    def _broadcast_swarm_fire(self, poly_quote, is_early: bool,
                              primary_ghosted: bool, force_explore: bool):
        """Grito de Fuego: fires every armed swarm agent synchronized with
        the primary's shot. Called from _place_auto_bet with self._lock held.

        Three broadcast modes:

        - normal            (primary_ghosted=False, force_explore=False):
            each agent runs its own risk sizing with the shared CLOB odds;
            should_bet=False → ghost shadow (data harvest).
        - ghost data harvest (primary_ghosted=True, force_explore=False):
            primary ghosted (CB / Kelly blocked); every swarm agent saves a
            ghost shadow bet so the ML pipeline still learns "who was right".
        - explore override  (primary_ghosted=False, force_explore=True):
            every agent fires a flat $1.00 real shadow bet with
            ``|explore`` suffixed to strategy_name. Kelly is bypassed.

        The caller already holds self._lock — this method MUST NOT re-acquire
        it (threading.Lock is not reentrant).
        """
        # Snapshot + mark-consumed in a single pass. The caller's lock keeps
        # _run_swarm_agents from racing with us on _swarm_signals mutation.
        squad: list[tuple[str, dict]] = []
        for aid, blob in self._swarm_signals.items():
            sig = blob.get("signal_obj")
            if sig is None or blob.get("fired"):
                continue
            squad.append((aid, dict(blob)))  # shallow copy for the loop
            blob["fired"] = True              # mark chambered bullet spent

        if not squad:
            log.debug("[SWARM-FIRE] No armed swarm signals — nothing to "
                      "broadcast")
            return

        mode = ("EXPLORE" if force_explore
                else ("GHOST_HARVEST" if primary_ghosted else "NORMAL"))
        log.info(f"[SWARM-FIRE] Broadcasting {len(squad)} swarm agent(s) "
                 f"| mode={mode} | is_early={is_early} | "
                 f"window={self.window.window_id}")

        for aid, blob in squad:
            sw_signal = blob["signal_obj"]
            sw_side = sw_signal.prediction
            strategy_label = blob.get("strategy_label", "unknown")
            display = blob.get("agent_cfg_display", aid)
            initial_balance = float(blob.get("initial_balance",
                                             self._default_initial_balance()))

            # ── Per-side CLOB odds from the shared quote snapshot ──
            if (poly_quote and not poly_quote.is_stale
                    and poly_quote.up_price > 0):
                raw_token = (poly_quote.up_price if sw_side == "UP"
                             else poly_quote.down_price)
                sw_odds = round(max(5.0, min(95.0, raw_token * 100)), 1)
                sw_odds_source = "clob_live"
            else:
                raw = (self.window.up_odds if sw_side == "UP"
                       else self.window.down_odds)
                sw_odds = max(5.0, min(95.0, float(raw + 2)))
                sw_odds_source = "internal_fallback"

            # Ensure portfolio row exists (lazy-init for new agents)
            portfolio_row = db.get_agent_portfolio(aid)
            if not portfolio_row:
                try:
                    db.upsert_agent_portfolio(
                        aid, display, strategy_label, initial_balance,
                    )
                    portfolio_row = db.get_agent_portfolio(aid)
                except Exception as e:
                    log.warning(f"[SWARM-FIRE] portfolio init failed "
                                f"for {aid}: {e}")
            agent_balance = (portfolio_row["balance"] if portfolio_row
                             else initial_balance)

            # ── Decide amount + ghost flag per broadcast mode ──
            if force_explore:
                # Every agent fires $1 flat with |explore suffix
                should_bet = True
                amount = 1.00
                is_ghost = False
                sw_strategy = f"{strategy_label}|explore"
                reason = "explore_override"
            elif primary_ghosted:
                # Data harvest: calculated amount recorded as ghost shadow
                should_bet = False
                amount = round(max(1.0, min(5.0, agent_balance * 0.02)), 2)
                is_ghost = True
                sw_strategy = strategy_label
                reason = "primary_ghosted_harvest"
            else:
                # Normal path: per-agent Kelly with the shared CLOB odds
                swarm_rl_ctx = self._build_rl_context(
                    agent_id=aid,
                    ml_features=None,
                    ta=self.current_ta,
                    signal=sw_signal,
                    initial_balance=initial_balance,
                )
                should_bet, amount, reason = self.risk.calculate_agent_bet(
                    sw_signal, agent_balance, sw_odds,
                    trust_score=50.0,     # swarm agents: neutral trust
                    agent_id=aid,
                    rl_context=swarm_rl_ctx,
                )
                if should_bet:
                    is_ghost = False
                    sw_strategy = strategy_label
                else:
                    # Swarm agent's own risk blocked → ghost shadow
                    amount = round(max(1.0, min(5.0, agent_balance * 0.02)), 2)
                    is_ghost = True
                    sw_strategy = strategy_label

            log.info(f"[SWARM-FIRE] {aid}: {sw_side} ${amount:.2f} @ "
                     f"{sw_odds}c | ghost={is_ghost} | "
                     f"strategy={sw_strategy} | {reason}")

            try:
                db.save_bet(
                    self.window.window_id,
                    sw_side,
                    amount,
                    sw_odds,
                    self.window.open_price,
                    is_manual=False,
                    is_early=is_early,
                    ai_model=aid,
                    is_shadow=True,
                    is_ghost=is_ghost,
                    strategy_name=sw_strategy,
                    odds_source=sw_odds_source,
                )
            except Exception as e:
                log.warning(f"[SWARM-FIRE] db.save_bet failed for {aid}: {e}")

    def _watchful_hold_defer(self, side: str, reason: str, odds: float,
                              odds_source: str, is_early: bool,
                              poly_quote, veto_kind: str) -> bool:
        """Try to stash a soft veto as PENDING_RETRY instead of committing ghost.

        Returns True if deferred (caller should return without ghosting).
        Returns False if caller must proceed with ghost fallback.
        """
        if not rules.get("watchful_hold", "enabled", False):
            return False
        allowed = rules.get("watchful_hold", "retry_soft_vetos",
                            ["ev_low", "kelly_low", "trust_low",
                             "coinflip", "judge_borderline", "low_payout"])
        if veto_kind not in (allowed or []):
            return False
        # Don't defer if window already near close
        commit_before = int(rules.get("watchful_hold",
                                       "commit_ghost_before_close_sec", 15))
        if self._time_left() <= commit_before:
            return False
        self._pending_retry = {
            "side": side, "reason": reason, "odds": odds,
            "odds_source": odds_source, "is_early": is_early,
            "veto_kind": veto_kind,
            "retries": 0, "consecutive_ok": 0,
            "armed_at": time.time(),
        }
        # Re-arm sniper loop so retries can fire (triggered path set it True)
        self._bet_evaluation_done = False
        try:
            prom_metrics.inc_counter("pending_retry_triggered_total",
                                      {"veto": veto_kind})
        except Exception:
            pass
        log.info(f"[WATCHFUL HOLD] {veto_kind} deferido → {side}@{odds}c "
                 f"| {reason[:80]}")
        return True

    def _watchful_retry_tick(self, signal, price_data, elapsed: float):
        """Re-evaluate a PENDING_RETRY. Called inside sniper eval while held.

        Either upgrades to real bet (soft veto cleared) or commits ghost
        when the close deadline is reached.
        """
        pr = self._pending_retry
        if not pr or self._window_bet_placed:
            return
        commit_before = int(rules.get("watchful_hold",
                                       "commit_ghost_before_close_sec", 15))
        max_retries = int(rules.get("watchful_hold", "max_retries", 10))
        consec_needed = int(rules.get("watchful_hold",
                                       "upgrade_requires_consecutive_ok", 2))

        pr["retries"] += 1
        time_left = self._time_left()

        # Direction still aligned?
        if signal and signal.prediction != pr["side"]:
            pr["consecutive_ok"] = 0

        # Refresh odds + EV with current CLOB
        try:
            poly_now = self.polymarket.get_current_btc5min_quote()
            if poly_now and not poly_now.is_stale:
                raw = (poly_now.up_price if pr["side"] == "UP"
                       else poly_now.down_price)
                pr["odds"] = round(max(5.0, min(95.0, raw * 100)), 1)
                pr["odds_source"] = "clob_live"
        except Exception:
            poly_now = None

        # Soft-veto cleared?
        cleared = False
        odds = pr["odds"]
        conf = signal.confidence if signal else 0
        payout = (100.0 / odds) if odds > 0 else 0
        ev = (conf / 100.0) * payout
        min_ev = rules.get("ev_gate", "min_ev", 1.02)
        min_payout = rules.get("ev_gate", "min_payout", 1.15)
        coinflip_margin = int(rules.get("ev_gate",
                                         "coinflip_margin_cents", 5))
        if (abs(odds - 50) > coinflip_margin
                and payout >= min_payout and ev >= min_ev
                and signal and signal.prediction == pr["side"]):
            cleared = True

        if cleared:
            pr["consecutive_ok"] += 1
            log.info(f"[WATCHFUL HOLD] retry #{pr['retries']} OK "
                     f"({pr['consecutive_ok']}/{consec_needed}) "
                     f"odds={odds}c ev={ev:.3f}")
            if pr["consecutive_ok"] >= consec_needed:
                log.info(f"[WATCHFUL HOLD] UPGRADE → apostando {pr['side']} "
                         f"tras {pr['retries']} retries")
                try:
                    prom_metrics.inc_counter(
                        "pending_retry_upgraded_total",
                        {"veto": pr["veto_kind"]})
                except Exception:
                    pass
                self._pending_retry = None
                self._place_auto_bet(signal, is_early=pr["is_early"],
                                     force_explore=False)
                return
        else:
            pr["consecutive_ok"] = 0

        # Timeout → commit ghost
        if time_left <= commit_before or pr["retries"] >= max_retries:
            log.info(f"[WATCHFUL HOLD] TIMEOUT → commit ghost "
                     f"{pr['side']}@{pr['odds']}c ({pr['reason'][:60]})")
            try:
                prom_metrics.inc_counter("pending_retry_timeout_total",
                                          {"veto": pr["veto_kind"]})
            except Exception:
                pass
            self._save_ghost_trade(pr["side"], pr["odds"],
                                    f"[PENDING TIMEOUT] {pr['reason']}",
                                    pr["is_early"],
                                    odds_source=pr["odds_source"])
            self._pending_retry = None

    def _save_ghost_trade(self, side: str, odds: int,
                          reason: str, is_early: bool,
                          odds_source: str = "internal_fallback"):
        """Record a ghost trade with calculated amount for ML data quality."""
        self._window_bet_placed = True
        self._is_ghost_trade = True
        self._ghost_reason = reason
        agent_balance = self._resolve_agent_balance(get_primary_agent_id())
        ghost_amount = max(1.0, min(5.0, agent_balance * 0.02))
        try:
            db.save_bet(self.window.window_id, side, round(ghost_amount, 2),
                        odds, self.window.open_price,
                        is_manual=False, is_early=is_early,
                        ai_model=get_primary_agent_id(),
                        is_ghost=True, strategy_name="smc_liquidity",
                        odds_source=odds_source)
        except Exception as e:
            log.error(f"DB write failed (save_ghost_bet): {e}")

    # ── Helpers for MSS daily levels ─────────────────────────

    @staticmethod
    def _get_daily_high(ohlcv_5m: list[dict]) -> float:
        """Get session high from OHLCV klines (last ~288 candles = 24h)."""
        if not ohlcv_5m:
            return 0
        return max(c.get("high", 0) for c in ohlcv_5m[-288:])

    @staticmethod
    def _get_daily_low(ohlcv_5m: list[dict]) -> float:
        """Get session low from OHLCV klines (last ~288 candles = 24h)."""
        if not ohlcv_5m:
            return 0
        lows = [c.get("low", float("inf")) for c in ohlcv_5m[-288:]]
        return min(lows) if lows else 0

    def _resolve(self):
        """Resolve window: snapshot state under lock, then do heavy DB work outside.

        Phase 1 (under lock): read price, compute outcome, build BetResult,
        mark window inactive. This is fast (~microseconds).
        Phase 2 (background thread): all DB writes, trust updates, swarm
        resolution. The tick loop is free to continue immediately.

        Resolution Oracle: Chainlink BTC/USD (matches Polymarket resolution source).
        Polymarket description: "resolution source is Chainlink BTC/USD data stream"
        Fallback to Binance spot if Chainlink is unavailable.
        """
        # ── Oracle priority: Chainlink BTC/USD (Polymarket resolution source) ──
        p = self.price_feed.get_price()
        if p and p.price > 0 and "chainlink" in p.source.lower():
            close = p.price
            resolution_source = p.source
        elif p and p.price > 0:
            # get_price() returned a non-Chainlink source (Binance relay)
            # Still usable as fallback since it's very close
            close = p.price
            resolution_source = f"{p.source} (chainlink unavailable)"
        else:
            # Last resort: Binance HF stream
            binance_p = self.price_feed.get_binance_price()
            if not binance_p or binance_p.price <= 0:
                self.window.is_active = False
                self.current_bet = None
                return
            close = binance_p.price
            resolution_source = f"{binance_p.source} (fallback)"

        # ── Watchful Hold safety net: unresolved pending → commit ghost ──
        if self._pending_retry and not self._window_bet_placed:
            pr = self._pending_retry
            log.info(f"[WATCHFUL HOLD] Window close sin upgrade → ghost "
                     f"{pr['side']}@{pr['odds']}c")
            try:
                prom_metrics.inc_counter("pending_retry_timeout_total",
                                          {"veto": pr.get("veto_kind", "unknown")})
            except Exception:
                pass
            try:
                self._save_ghost_trade(pr["side"], pr["odds"],
                                        f"[PENDING WINDOW_CLOSE] {pr['reason']}",
                                        pr["is_early"],
                                        odds_source=pr["odds_source"])
            except Exception as e:
                log.error(f"[WATCHFUL HOLD] ghost commit failed: {e}")
            self._pending_retry = None

        outcome = "UP" if close >= self.window.open_price else "DOWN"
        log.info(f"=== RESOLVED: {outcome} | "
                 f"${self.window.open_price:,.2f} -> ${close:,.2f} "
                 f"[oracle: {resolution_source}] ===")

        # ── Phase 1: Snapshot under lock (already held by caller) ──
        snap = {
            "window_id": self.window.window_id,
            "open_price": self.window.open_price,
            "close_price": close,
            "outcome": outcome,
            "close_time": time.time(),
            "signal": None,
            "bet": None,
            "rag_strategy": self._current_rag_strategy or "",
            "ml_oracle_prob": self._ml_oracle_prob,
            "ml_skipped": self._ml_skipped,
            "ta_snap": dict(self.current_ta),
        }

        if self.current_signal:
            s = self.current_signal
            snap["signal"] = {
                "prediction": s.prediction,
                "confidence": s.confidence,
            }

        if self.current_bet:
            b = self.current_bet
            # ── Check if position was exited early by SurvivalMonitor ──
            early_exit = (self.survival.get_exit_info(b.window_id)
                          if self.survival else None)
            if early_exit and early_exit.exited:
                # P&L comes from the actual exit, not the natural resolve
                exit_p = early_exit.exit_price  # decimal
                recovery = exit_p * early_exit.size_tokens
                profit = recovery - b.amount
                payout = recovery
                won = profit > 0
                log.info(f"[SURVIVAL] Using {early_exit.exit_reason} P&L: "
                         f"exit@{exit_p:.4f} profit=${profit:+.2f}")
            else:
                won = b.side == outcome
                pnl = calc_polymarket_pnl(b.amount, b.side, b.price_cents, won)
                payout = pnl["net_payout"]
                profit = pnl["net_profit"]
            snap["bet"] = {
                "window_id": b.window_id,
                "side": b.side,
                "amount": b.amount,
                "price_cents": b.price_cents,
                "open_price": b.open_price,
                "is_auto": b.is_auto,
                "won": won,
                "payout": payout,
                "profit": profit,
                "poly_fee": pnl["fee"],
                "tokens_bought": pnl["tokens_bought"],
            }

            # Build BetResult in-memory (fast, no DB)
            r = BetResult(
                window_id=b.window_id,
                side=b.side, amount=b.amount,
                price_cents=b.price_cents,
                open_price=b.open_price,
                close_price=close,
                outcome=outcome, payout=payout, profit=profit, won=won,
                timestamp=datetime.now().strftime("%H:%M:%S"),
                ai_confidence=(self.current_signal.confidence
                               if self.current_signal else 0),
                ai_prediction=(self.current_signal.prediction
                               if self.current_signal else ""),
                is_auto=b.is_auto,
                rag_strategy=snap["rag_strategy"],
                ml_oracle_prob=snap["ml_oracle_prob"],
                ml_skipped=snap["ml_skipped"],
            )
            self._recent_results.insert(0, r)
            if len(self._recent_results) > 100:
                self._recent_results = self._recent_results[:100]

            # ── Prometheus metrics hook ──
            prom_metrics.record_window(
                get_primary_agent_id(),
                "win" if won else "loss",
            )
            prom_metrics.record_confidence(b.side, self.current_signal.confidence
                                           if self.current_signal else 50)

        elif self._is_ghost_trade and self.current_signal:
            # Register ghost trade in results so the history table shows it
            s = self.current_signal
            ghost_won = s.prediction == outcome
            r = BetResult(
                window_id=self.window.window_id,
                side=s.prediction, amount=0, price_cents=50,
                open_price=self.window.open_price,
                close_price=close,
                outcome=outcome, payout=0, profit=0, won=ghost_won,
                timestamp=datetime.now().strftime("%H:%M:%S"),
                ai_confidence=s.confidence,
                ai_prediction=s.prediction,
                is_auto=True,
                rag_strategy=snap["rag_strategy"],
                ml_oracle_prob=snap["ml_oracle_prob"],
                ml_skipped=snap["ml_skipped"],
                is_ghost=True,
            )
            self._recent_results.insert(0, r)
            if len(self._recent_results) > 100:
                self._recent_results = self._recent_results[:100]

        # Mark window done immediately so tick loop moves on
        self.window.is_active = False
        self.current_bet = None
        # Cancel any outstanding CLOB orders (ladder rungs, etc.)
        if self.executor and self.executor.connected:
            try:
                self.executor.cancel_all()
            except Exception as e:
                log.warning(f"[EXEC] cancel_all on resolve failed: {e}")
        # Untrack positions from SurvivalMonitor
        if self.survival:
            self.survival.untrack_all(snap["window_id"])

        # ── Phase 2: Heavy DB work in background thread ──
        threading.Thread(
            target=self._resolve_db_work,
            args=(snap,),
            daemon=True,
            name=f"resolve-w{snap['window_id']}",
        ).start()

    def _resolve_db_work(self, snap: dict):
        """All DB writes for window resolution. Runs outside _lock."""
        window_id = snap["window_id"]
        close = snap["close_price"]
        outcome = snap["outcome"]

        # Close window in DB
        try:
            db.close_window(window_id, snap["close_time"], close, outcome)
        except Exception as e:
            log.error(f"DB write failed (close_window): {e}")

        # Record AI outcome with trust update
        sig = snap["signal"]
        if sig:
            final_conf = sig["confidence"]
            self.ai.record_outcome(sig["prediction"], outcome, final_conf)
            try:
                db.update_prediction_outcome(window_id, outcome, final_conf)
            except Exception as e:
                log.error(f"DB write failed (update_prediction_outcome): {e}")

            # ── Epistemic Reflection: async post-loss analysis ──
            if sig["prediction"] != outcome:
                cvd_snap = self.binance.get_cvd_summary(minutes=5)
                self.ai.trigger_reflection(
                    sig["prediction"], outcome, final_conf,
                    snap["ta_snap"],
                    cvd_summary=cvd_snap if cvd_snap.get("connected") else None,
                )

            # ── Strategy Fatigue: track consecutive losses per RAG strategy ──
            rag_strat = snap.get("rag_strategy", "")
            if rag_strat:
                won = sig["prediction"] == outcome
                with self._fatigue_lock:
                    if won:
                        self._strategy_losses[rag_strat] = 0
                    else:
                        self._strategy_losses[rag_strat] = (
                            self._strategy_losses.get(rag_strat, 0) + 1
                        )
                        consec = self._strategy_losses[rag_strat]
                        start_t = self._strategy_start_time.get(rag_strat, time.time())
                        age = time.time() - start_t
                        if (consec >= self._STRATEGY_MAX_LOSSES
                                or age > self._STRATEGY_MAX_AGE):
                            cooldown_until = time.time() + self._STRATEGY_COOLDOWN
                            self._strategy_blacklist[rag_strat] = cooldown_until
                            self._strategy_losses.pop(rag_strat, None)
                            self._strategy_start_time.pop(rag_strat, None)
                            reason = (f"losses={consec}" if consec >= self._STRATEGY_MAX_LOSSES
                                      else f"age={age:.0f}s")
                            log.warning(f"[FATIGUE] Strategy '{rag_strat}' BLACKLISTED — "
                                        f"{reason}. Cooldown until "
                                        f"{datetime.fromtimestamp(cooldown_until, tz=timezone.utc):%H:%M}")

            # ── Resolve all swarm agent bets ──
            self._resolve_swarm_bets(window_id, close, outcome)

            # ── Resolve ghost bets (record outcome for ML, no P&L) ──
            try:
                conn = db._get_conn()
                ghost_rows = conn.execute(
                    "SELECT ai_model, side, odds_cents FROM bets "
                    "WHERE window_id=? AND is_ghost=1 "
                    "AND outcome IS NULL",
                    (window_id,),
                ).fetchall()
                for ghost_row in ghost_rows:
                    g_won = ghost_row["side"] == outcome
                    ai_model = ghost_row["ai_model"] or get_primary_agent_id()
                    db.close_bet(window_id, close, outcome,
                                 0, 0, g_won, ai_model=ai_model)
                    log.info(f"[GHOST] {ai_model} resolved: "
                             f"{'WIN' if g_won else 'LOSS'} "
                             f"{ghost_row['side']}->{outcome}")
            except Exception as e:
                log.warning(f"[GHOST] Failed to resolve ghost bet (safe-fail): {e}")

        # ── Primary agent bet DB writes ──
        bet = snap["bet"]
        if bet:
            won = bet["won"]
            profit = bet["profit"]
            payout = bet["payout"]

            if bet["is_auto"]:
                try:
                    db.update_agent_balance(get_primary_agent_id(), profit, won)
                except Exception as e:
                    log.error(f"DB write failed (update_agent_balance): {e}")
                try:
                    db.close_bet(bet["window_id"], close, outcome,
                                 payout, profit, won, ai_model=get_primary_agent_id())
                except Exception as e:
                    log.error(f"DB write failed (close_bet): {e}")
            else:
                log.info(f"Manual bet resolved in-memory only (quarantined from DB)")

            agent_port = db.get_agent_portfolio(get_primary_agent_id())
            bal = agent_port["balance"] if agent_port else self._default_initial_balance()
            log.info(f"{'WIN' if won else 'LOSS'} "
                     f"{bet['side']}->{outcome} "
                     f"P&L:${profit:+,.2f} Bal:${bal:,.2f}")
            # Track fatigue + scoring for primary agent
            self._update_agent_fatigue(get_primary_agent_id(), won)
            # Prometheus: balance + P&L
            prom_metrics.record_balance(get_primary_agent_id(), bal)
            prom_metrics.record_pnl(
                get_primary_agent_id(),
                agent_port["total_profit"] if agent_port else 0.0,
            )

        # ── Auto-Bailout: rescue bankrupt agents ──────────────
        self._check_agent_bailouts()

    _BAILOUT_THRESHOLD = 2.0   # dollars

    def _check_agent_bailouts(self):
        """Check all agent portfolios and rescue any below $2.00.

        Bailout amount tracks the current Global Default (live-reloadable
        via ``dynamic_rules.json → defaults.initial_balance``), so any
        UI change to the default balance also relines the bailout floor.
        """
        bailout_amount = self._default_initial_balance()
        try:
            portfolios = db.get_all_agent_portfolios()
            for port in portfolios:
                agent_id = port["agent_id"]
                balance = port["balance"]
                if balance < self._BAILOUT_THRESHOLD:
                    db.reset_agent_portfolio(agent_id, bailout_amount)
                    log.warning(
                        f"BAILOUT: Agente '{agent_id}' rescatado "
                        f"(${balance:.2f} -> ${bailout_amount:.2f})"
                    )
                    # Reset fatigue state too — fresh start
                    with self._fatigue_lock:
                        if agent_id in self._agent_fatigue:
                            self._agent_fatigue[agent_id] = {
                                "losses": 0,
                                "start_time": time.time(),
                                "fatigued": False,
                                "fatigue_since": 0,
                                "rotation_idx": 0,
                            }
        except Exception as e:
            log.warning(f"[BAILOUT] Check failed (safe-fail): {e}")

    # ── Live Observer (15-second commentary) ──────────────────

    def _live_observer_loop(self):
        """Background loop: every 15s, generate a quick market commentary.

        Runs in its own daemon thread — NEVER blocks the 2s tick loop.
        Uses a lightweight AI call (max 80 tokens) or falls back to TA-based
        commentary if no API key is available.
        """
        # Wait for first price data before starting
        while self.running:
            p = self.price_feed.get_price()
            if p and p.price > 0:
                break
            time.sleep(2)

        while self.running:
            try:
                self._live_observer_tick()
            except Exception:
                log.exception("Live Observer error")
            time.sleep(15)

    def _live_observer_tick(self):
        """Single observer iteration — fast, non-blocking."""
        p = self.price_feed.get_price()
        if not p or p.price <= 0:
            return

        # Grab a snapshot of TA and window state
        with self._lock:
            ta = dict(self.current_ta)
            w_open = self.window.open_price
            w_active = self.window.is_active
            time_left = self._time_left()

        if not w_active or w_open <= 0:
            self._live_suggestion = "Esperando nueva ventana..."
            self._live_suggestion_ts = time.time()
            return

        price = p.price
        diff = price - w_open
        pct = (diff / w_open) * 100 if w_open else 0
        rsi = ta.get("rsi", 50)
        macd_h = ta.get("macd_histogram", 0)
        bp = ta.get("tick_buy_pressure", 50)
        vol = ta.get("volatility", 0)

        # Observer policy: ONLY use a local LLM (free) or TA fallback.
        # NEVER call paid remote APIs from the observer (runs every 15s).
        # The local model is resolved from dynamic_rules (primary or swarm
        # agent with api_type == "local") — no separate stateful toggle.
        suggestion = None
        local_cfg = self._resolve_local_agent_cfg()
        if local_cfg:
            suggestion = self._observer_local_call(
                local_cfg, price, diff, pct, rsi, macd_h, bp, vol, time_left
            )

        if not suggestion:
            suggestion = self._observer_ta_fallback(
                price, diff, pct, rsi, macd_h, bp, vol, time_left
            )

        with self._lock:
            self._live_suggestion = suggestion
            self._live_suggestion_ts = time.time()

    @staticmethod
    def _resolve_local_agent_cfg() -> Optional[dict]:
        """Return a dynamic_rules agent config with api_type == 'local'.

        Prefers the primary agent if it is local; otherwise falls back to
        the first enabled swarm agent that is local. Returns None if no
        local agent is configured — observer will use TA fallback.
        """
        primary = swarm.get_primary_agent()
        if primary and primary.get("api_type") == "local":
            return primary
        for agent in swarm.get_active_agents():
            if agent.get("api_type") == "local":
                return agent
        return None

    def _observer_local_call(self, local_cfg: dict, price, diff, pct, rsi,
                             macd_h, bp, vol, time_left) -> Optional[str]:
        """Quick local LLM call — max 80 tokens, 8s timeout.
        Uses ONLY the local GPU/CPU (free). Non-blocking: if the local LLM
        is busy thinking, skip gracefully instead of queuing."""
        import requests as _req

        url = local_cfg.get("api_base_url") or self.local_llm.url
        model = local_cfg.get("model")
        if not model:
            return None
        if not self.local_llm.is_available(url):
            return None

        # Grab Fear & Greed (cached, no network call)
        fg = sentiment.get_fear_greed()
        fg_text = f"Fear&Greed={fg['value']}({fg['label']})" if fg.get("value") else ""

        prompt = (
            f"Eres un comentarista de mercado BTC en vivo. "
            f"Datos AHORA: Precio ${price:,.2f}, cambio {pct:+.3f}% vs apertura, "
            f"RSI={rsi:.0f}, MACD hist={macd_h:+.2f}, "
            f"presion compra={bp:.0f}%, volatilidad={vol:.4f}, "
            f"{f'{fg_text}, ' if fg_text else ''}"
            f"quedan {time_left}s en ventana. "
            f"Responde SOLO 1-2 oraciones cortas en espanol describiendo "
            f"el pulso del mercado ahora mismo. "
            f"IMPORTANTE: Termina SIEMPRE con exactamente una de estas frases: "
            f"\"CONCLUSION: Podria subir\", \"CONCLUSION: Podria bajar\", "
            f"o \"CONCLUSION: Se mantiene lateral\"."
        )
        try:
            resp = _req.post(
                f"{url}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 80,
                    "temperature": 0.4,
                },
                timeout=8,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            return text[:200]
        except Exception as e:
            log.debug(f"Live Observer local LLM call failed: {e}")
            return None

    @staticmethod
    def _observer_ta_fallback(price, diff, pct, rsi, macd_h, bp, vol,
                              time_left) -> str:
        """TA-based commentary when no AI available."""
        parts = []

        # Price direction
        if abs(pct) < 0.01:
            parts.append("Mercado lateral, sin movimiento significativo.")
        elif pct > 0.05:
            parts.append(f"Fuerte presion alcista (+{pct:.3f}%).")
        elif pct > 0:
            parts.append(f"Leve sesgo alcista (+{pct:.3f}%).")
        elif pct < -0.05:
            parts.append(f"Fuerte presion bajista ({pct:.3f}%).")
        else:
            parts.append(f"Leve sesgo bajista ({pct:.3f}%).")

        # RSI
        if rsi > 70:
            parts.append(f"RSI en sobrecompra ({rsi:.0f}).")
        elif rsi < 30:
            parts.append(f"RSI en sobreventa ({rsi:.0f}).")

        # Buy pressure
        if bp > 65:
            parts.append("Compradores dominan el flujo.")
        elif bp < 35:
            parts.append("Vendedores dominan el flujo.")

        # Volatility
        if vol > 0.003:
            parts.append("Alta volatilidad detectada.")

        # Time urgency
        if time_left < 30:
            parts.append("Cierre inminente.")

        base = " ".join(parts[:3]) if parts else "Mercado en calma."

        # Directional conclusion based on TA signals
        bullish = (pct > 0.01) + (rsi < 40) + (macd_h > 0) + (bp > 55)
        bearish = (pct < -0.01) + (rsi > 60) + (macd_h < 0) + (bp < 45)
        if bullish > bearish:
            base += " CONCLUSION: Podria subir"
        elif bearish > bullish:
            base += " CONCLUSION: Podria bajar"
        else:
            base += " CONCLUSION: Se mantiene lateral"
        return base

    @staticmethod
    def _extract_conclusion(text: str) -> str:
        """Extract directional conclusion from live observer text."""
        if not text:
            return ""
        t = text.upper()
        if "CONCLUSION: PODRIA SUBIR" in t or "CONCLUSIÓN: PODRÍA SUBIR" in t:
            return "UP"
        if "CONCLUSION: PODRIA BAJAR" in t or "CONCLUSIÓN: PODRÍA BAJAR" in t:
            return "DOWN"
        if "CONCLUSION: SE MANTIENE LATERAL" in t or "CONCLUSIÓN: SE MANTIENE LATERAL" in t:
            return "LATERAL"
        return ""

    def get_state(self) -> dict:
        # ── 0.3s TTL cache: avoid redundant work from rapid polling ──
        now = time.time()
        if self._state_cache and (now - self._state_cache_ts) < 0.3:
            return self._state_cache

        # ── Snapshot: copy raw values inside lock (fast) ──────
        with self._lock:
            tl = self._time_left()
            w_id = self.window.window_id
            w_open = self.window.open_price
            w_cur = self.window.current_price
            w_active = self.window.is_active
            w_up_odds = self.window.up_odds
            w_down_odds = self.window.down_odds
            w_end = self.window.end_time

            sig_snap = None
            if self.current_signal:
                s = self.current_signal
                sig_snap = (s.prediction, s.confidence, s.original_confidence,
                            s.reasoning, s.news_summary, s.risk_score,
                            s.update_count, s.updated_at, s.layer_alignment)

            bet_snap = None
            if self.current_bet:
                b = self.current_bet
                bet_snap = (b.side, b.amount, b.price_cents, b.is_auto)

            results_snap = list(self._recent_results[:30])

            ta_snap = dict(self.current_ta)
            bet_eval_done = self._bet_evaluation_done
            has_signal = self.current_signal is not None
            sniper_evals = len(self._window_evaluations)
            sniper_bet_placed = self._window_bet_placed
            sniper_evals_list = list(self._window_evaluations)
            sniper_eval_cutoff = self._EVAL_CUTOFF
            sniper_consensus_score = self._consensus_score
            sniper_consensus_direction = self._consensus_direction
            sniper_consensus_idx = self._consensus_idx
            sniper_last_trigger = self._last_trigger_type
            sniper_score_fire = self._CONSENSUS_SCORE_FIRE
            sniper_min_age = self._MIN_CONSENSUS_AGE
            second_opinion_snap = dict(self._second_opinion) if self._second_opinion else None
            second_opinion_in_flight = self._second_opinion_in_flight
            is_ghost_trade = self._is_ghost_trade
            ghost_reason = self._ghost_reason
            ml_oracle_prob = self._ml_oracle_prob
            ml_skipped = self._ml_skipped
            has_bet = self.current_bet is not None
            live_suggestion = self._live_suggestion
            live_suggestion_ts = self._live_suggestion_ts

        # ── Build response OUTSIDE lock (no contention) ───────
        pd_ = (w_cur - w_open) if w_open else 0
        pp = (pd_ / w_open * 100) if w_open else 0
        winning_side = "UP" if pd_ >= 0 else "DOWN"

        # Primary agent stats from agent_portfolios (source of truth)
        _ap = db.get_agent_portfolio(get_primary_agent_id())
        if _ap:
            p_balance = _ap["balance"]
            p_total_bets = _ap["total_bets"]
            p_wins = _ap["wins"]
            p_losses = _ap["losses"]
            p_total_profit = _ap["total_profit"]
            p_peak = _ap["peak_balance"]
        else:
            p_balance = self._default_initial_balance()
            p_total_bets = p_wins = p_losses = 0
            p_total_profit = 0.0
            p_peak = p_balance

        # ── Live Dynamic Sizing: override displayed balance with real
        #    on-chain USDC.e when the toggle is ON ──
        if getattr(cfg, "LIVE_DYNAMIC_SIZING", False):
            live_bal = self._resolve_agent_balance(get_primary_agent_id())
            if live_bal != p_balance:
                p_balance = live_bal

        p_dd = round(((p_peak - p_balance) / p_peak * 100), 1) if p_peak > 0 else 0
        wr = (p_wins / p_total_bets * 100) if p_total_bets > 0 else 0

        sig = None
        if sig_snap:
            sig = {
                "prediction": sig_snap[0], "confidence": sig_snap[1],
                "original_confidence": sig_snap[2], "reasoning": sig_snap[3],
                "news_summary": sig_snap[4], "risk_score": sig_snap[5],
                "update_count": sig_snap[6], "updated_at": sig_snap[7],
                "layer_alignment": sig_snap[8],
            }

        bet = None
        if bet_snap:
            b_side, b_amount, b_price_cents, b_is_auto = bet_snap
            would_win = b_side == winning_side
            live_pnl = calc_polymarket_pnl(b_amount, b_side, b_price_cents, would_win)
            bet = {
                "side": b_side, "amount": b_amount,
                "price_cents": b_price_cents,
                "potential_payout": live_pnl["net_payout"] if would_win else 0,
                "live_profit": live_pnl["net_profit"],
                "would_win": would_win, "is_auto": b_is_auto,
                "effective_odds": live_pnl["effective_odds"],
            }

        results = []
        for r in results_snap:
            results.append({
                "window_id": r.window_id, "side": r.side,
                "amount": r.amount, "price_cents": r.price_cents,
                "outcome": r.outcome, "profit": r.profit,
                "won": bool(r.won),
                "timestamp": r.timestamp, "ai_confidence": r.ai_confidence,
                "ai_prediction": r.ai_prediction,
                "open_price": r.open_price, "close_price": r.close_price,
                "is_auto": r.is_auto,
                "rag_strategy": r.rag_strategy or "",
                "ml_oracle_prob": r.ml_oracle_prob,
                "ml_skipped": r.ml_skipped,
                "is_ghost": getattr(r, "is_ghost", False),
            })

        # These calls have their own internal locks — safe outside engine lock
        feed_status = self.price_feed.get_status()
        acc = self.ai.get_accuracy_stats()
        trust = self.ai.get_trust_info()
        prices = self.price_feed.get_recent_prices(120)

        # Live gain preview for manual bet UI (fixed $1/$3/$5 tiers)
        live_up_scenarios = {}
        live_down_scenarios = {}
        if w_active:
            for side, odds, out in [
                ("UP", w_up_odds, live_up_scenarios),
                ("DOWN", w_down_odds, live_down_scenarios),
            ]:
                odds_c = max(5, min(95, odds))
                for label, amt in [("small", 1.0), ("medium", 3.0),
                                    ("large", 5.0)]:
                    sc = calc_polymarket_pnl(amt, side, odds_c, True)
                    out[label] = {
                        "amount": amt,
                        "potential_gain": sc["net_profit"],
                        "odds_cents": odds_c,
                        "payout": sc["net_payout"],
                        "effective_odds": sc["effective_odds"],
                    }

        # Bet status (computed from snapshots, no lock needed)
        if has_bet:
            bet_status = "placed"
        elif has_signal and bet_eval_done and not has_bet:
            bet_status = "skipped"
        elif has_signal and w_active and sniper_evals > 0 and not sniper_bet_placed:
            bet_status = f"sniping ({sniper_evals} evals)"
        elif has_signal and w_active:
            bet_status = "observing"
        else:
            bet_status = "no_signal"

        state = {
            "window": {
                "id": w_id, "open_price": w_open,
                "current_price": w_cur,
                "price_diff": round(pd_, 2), "price_pct": round(pp, 4),
                "time_left": tl,         # legacy — conservado para retrocompat
                "end_time": w_end,       # CRITICO: timestamp epoch absoluto del servidor (Fluid Timer UI)
                "is_active": w_active,
                "up_odds": w_up_odds, "down_odds": w_down_odds,
                "winning_side": winning_side,
            },
            "signal": sig,
            "bet": bet,
            "portfolio": {
                "balance": round(p_balance, 2),
                "total_bets": p_total_bets, "wins": p_wins,
                "losses": p_losses, "win_rate": round(wr, 1),
                "total_profit": round(p_total_profit, 2),
                "peak_balance": round(p_peak, 2),
                "max_drawdown": round(p_dd, 1),
                "daily_loss": 0,
                "daily_loss_limit": 10,
            },
            "ta": ta_snap,
            "live_up_scenarios": live_up_scenarios,
            "live_down_scenarios": live_down_scenarios,
            "accuracy": acc,
            "results": results,
            "price_history": prices,
            "feed": feed_status,
            "trust": trust,
            "ai_training": {
                "lessons": self.ai.learned_lessons[-5:],
                "bias": self.ai.bias_corrections,
                "sessions": self.ai.sessions_count,
            },
            "bet_status": bet_status,
            "sniper": {
                "evaluations": sniper_evals_list,
                "bet_placed": sniper_bet_placed,
                "eval_count": sniper_evals,
                "cutoff_seconds": sniper_eval_cutoff,
                "is_ghost_trade": is_ghost_trade,
                "ghost_reason": ghost_reason,
                "consensus_score": sniper_consensus_score,
                "consensus_direction": sniper_consensus_direction,
                "consensus_idx": sniper_consensus_idx,
                "last_trigger_type": sniper_last_trigger,
                "score_fire": sniper_score_fire,
                "min_consensus_age_sec": sniper_min_age,
                "second_opinion": ({
                    "source": second_opinion_snap.get("source"),
                    "direction": second_opinion_snap.get("direction"),
                    "confidence": second_opinion_snap.get("confidence"),
                    "veto": bool(second_opinion_snap.get("veto")),
                    "reasoning": second_opinion_snap.get("reasoning", "")[:200],
                    "age_sec": round(
                        time.time() - float(second_opinion_snap.get("arrived_at", 0)), 1
                    ),
                    "latency_ms": second_opinion_snap.get("latency_ms"),
                    "in_flight": second_opinion_in_flight,
                } if second_opinion_snap else {
                    "source": None,
                    "direction": None,
                    "confidence": None,
                    "veto": False,
                    "reasoning": "",
                    "age_sec": None,
                    "latency_ms": None,
                    "in_flight": second_opinion_in_flight,
                }),
            },
            "ml_oracle": {
                "enabled": _judge_model is not None,
                "prob": round(ml_oracle_prob, 4) if ml_oracle_prob is not None else None,
                "ml_skipped": ml_skipped,
                "threshold": ML_CONFIDENCE_THRESHOLD,
            },
            "ml_judge_enabled": cfg.ENABLE_ML_JUDGE,
            "live_sizing_enabled": getattr(cfg, 'LIVE_DYNAMIC_SIZING', False),
            "has_api_key": bool(self.ai._get_primary_config().get("api_key")),
            "paper_mode": getattr(cfg, 'PAPER_TRADING_MODE', True),
            "trading_mode": cfg.TRADING_MODE,
            "executor_connected": (self.executor.connected
                                   if self.executor else False),
            "primary_halted": self._primary_halted,
            "primary_halt_reason": self._primary_halt_reason,
            "server_time": datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),
            "swarm": self.get_swarm_state(),
            "live_suggestion": live_suggestion,
            "live_suggestion_age": round(time.time() - live_suggestion_ts)
                                  if live_suggestion_ts > 0 else -1,
            "live_pulse_conclusion": self._extract_conclusion(live_suggestion),
            "sentiment": {
                "fear_greed": sentiment.get_fear_greed(),
                "headlines": [h["title"] for h in sentiment.get_headlines()[:5]],
            },
            "rag": self._get_rag_state_snapshot(),
            "agent_fatigue": self._get_fatigue_state_snapshot(),
            "smc": {
                "available": bool(self._current_smc_features),
                "fvg": self._current_smc_features.get("smc_fvg_active", 0),
                "bos": self._current_smc_features.get("smc_bos_last", 0),
                "choch": self._current_smc_features.get("smc_choch_last", 0),
                "ob_nearest": self._current_smc_features.get("smc_ob_nearest", 0),
                "ob_distance_pct": self._current_smc_features.get("smc_ob_distance_pct", 0),
                "liq_nearest": self._current_smc_features.get("smc_liq_nearest", 0),
                "retracement_pct": self._current_smc_features.get("smc_retracement_pct", 0),
                "in_ote_zone": self._current_smc_features.get("smc_in_ote_zone", 0),
                "kill_zone": self._current_smc_features.get("smc_kill_zone", ""),
                "smc_support": self._current_smc_features.get("smc_support", 0),
                "smc_resistance": self._current_smc_features.get("smc_resistance", 0),
            },
            "mss": {
                "sweep": bool(self._current_mss_features.get("is_liquidity_swept_1m")),
                "sweep_dir": self._current_mss_features.get("sweep_direction", ""),
                "mss_detected": bool(self._current_mss_features.get("mss_detected")),
                "mss_dir": self._current_mss_features.get("mss_direction", ""),
                "post_sweep_reversal": bool(self._current_mss_features.get("post_sweep_reversal")),
            },
            "order_flow": self.binance.get_cvd_summary(minutes=5),
            "gex": self.gex.get_status(),
            "liq_cluster": self._current_liq_cluster or {},
            "circuit_breaker": sentiment.get_circuit_breaker_status(),
        }

        # Cache for 1s TTL
        self._state_cache = state
        self._state_cache_ts = time.time()
        return state

    # ── Primary AI Circuit Breaker controls ─────────────────

    def resume_primary(self) -> dict:
        """Manually resume primary AI after a circuit breaker halt."""
        was_halted = self._primary_halted
        self._primary_halted = False
        prev_reason = self._primary_halt_reason
        self._primary_halt_reason = ""
        self._primary_halt_time = 0
        if was_halted:
            log.info("[CIRCUIT BREAKER] Primary AI RESUMED manually")
        return {"was_halted": was_halted, "previous_reason": prev_reason}

    # ── Agent Fatigue Tracking ─────────────────────────────────

    def _get_agent_fatigue(self, agent_id: str) -> dict:
        """Get or initialize fatigue state for an agent. Caller must hold _fatigue_lock."""
        if agent_id not in self._agent_fatigue:
            self._agent_fatigue[agent_id] = {
                "losses": 0,
                "start_time": time.time(),
                "fatigued": False,
                "fatigue_since": 0,
                "rotation_idx": 0,
            }
        return self._agent_fatigue[agent_id]

    def _update_agent_fatigue(self, agent_id: str, won: bool):
        """Update fatigue counters after a bet resolves. Trigger fatigue if thresholds hit."""
        was_fatigued = False
        with self._fatigue_lock:
            state = self._get_agent_fatigue(agent_id)
            was_fatigued = state["fatigued"]

            if won:
                state["losses"] = 0
                # If fatigued and wins with the rotated prompt, reset fatigue
                if state["fatigued"]:
                    log.info(f"[AGENT_FATIGUE] {agent_id} WON with rotated prompt — "
                             f"resetting fatigue")
                    state["fatigued"] = False
                    state["fatigue_since"] = 0
                    state["start_time"] = time.time()
            else:
                state["losses"] += 1

            age = time.time() - state["start_time"]
            consec = state["losses"]

            # Check if agent should enter fatigue
            if not state["fatigued"]:
                if consec >= self._AGENT_MAX_LOSSES or age > self._AGENT_MAX_AGE:
                    state["fatigued"] = True
                    state["fatigue_since"] = time.time()
                    state["rotation_idx"] += 1
                    reason = (f"losses={consec}" if consec >= self._AGENT_MAX_LOSSES
                              else f"age={age:.0f}s")
                    log.warning(f"[AGENT_FATIGUE] {agent_id} FATIGUED — {reason}. "
                                f"Rotating to prompt variant #{state['rotation_idx']}")
                    # Force RAG strategy rotation on fatigue entry
                    if self._current_rag_strategy:
                        self._strategy_blacklist[self._current_rag_strategy] = (
                            time.time() + self._STRATEGY_COOLDOWN
                        )
                        log.info(f"[AGENT_FATIGUE] Blacklisted RAG strategy "
                                 f"'{self._current_rag_strategy}' due to agent fatigue")
            else:
                # Auto-reset after cooldown period in fatigued mode
                fatigue_age = time.time() - state["fatigue_since"]
                if fatigue_age > self._AGENT_FATIGUE_COOLDOWN:
                    log.info(f"[AGENT_FATIGUE] {agent_id} fatigue cooldown expired — "
                             f"resetting to base prompt")
                    state["losses"] = 0
                    state["fatigued"] = False
                    state["fatigue_since"] = 0
                    state["start_time"] = time.time()

            # Persist fatigue state to DB
            db.update_agent_fatigue_db(
                agent_id, state["fatigued"],
                state["rotation_idx"], state["fatigue_since"],
            )

        # Update score & streaks in DB (outside fatigue lock)
        db.update_agent_score_and_streak(agent_id, won, was_fatigued)

    def _get_agent_fatigue_prompt(self, agent_id: str) -> str:
        """Return a fatigue override prompt if agent is fatigued, else empty string.

        DISABLED: Prompt rotation was found to be counterproductive for BTC
        5-min binary bets (Gambler's Fallacy — 3 losses is normal variance).
        The multi-layer analysis (Order Flow + Derivatives + RAG) should never
        be replaced by a simplistic single-indicator prompt.
        Risk controls (Kelly, Trust Score, sizing reduction) handle losing
        streaks properly without destroying the prompt architecture.

        Fatigue tracking remains active in DB for ML dataset enrichment.
        """
        return ""

    def _get_agent_active_strategy(self, agent_id: str, base_strategy: str = "") -> str:
        """Return the human-readable strategy label for an agent.

        If fatigued, returns the fatigue rotation label (e.g. "Contrarian Total").
        Otherwise returns the base strategy name.
        """
        with self._fatigue_lock:
            state = self._get_agent_fatigue(agent_id)
            is_fatigued = state["fatigued"]
            rotation_idx = state["rotation_idx"]
        if not is_fatigued:
            return base_strategy

        # Use rotation labels matching the fatigue prompt index
        fatigue_section = rules.get_section("fatigue_prompts")
        labels = (fatigue_section or {}).get("rotation_labels", _FATIGUE_ROTATION_LABELS)
        idx = rotation_idx % len(labels) if labels else 0
        return labels[idx] if labels else base_strategy

    def reset_agent_fatigue(self, agent_id: str):
        """Reset fatigue state for a single agent (called from API)."""
        with self._fatigue_lock:
            self._agent_fatigue[agent_id] = {
                "losses": 0,
                "start_time": time.time(),
                "fatigued": False,
                "fatigue_since": 0,
                "rotation_idx": 0,
            }
        db.reset_agent_fatigue_db(agent_id)
        log.info(f"[AGENT_FATIGUE] {agent_id} fatigue manually reset via API")

    def _get_rag_state_snapshot(self) -> dict:
        """Thread-safe snapshot of RAG strategy state for get_state()."""
        with self._fatigue_lock:
            blacklisted = list(self._strategy_blacklist.keys())
            strategy_losses = dict(self._strategy_losses)
        return {
            "active_strategy": self._current_rag_strategy or None,
            "strategies_count": self.rag.count(),
            "blacklisted": blacklisted,
            "strategy_losses": strategy_losses,
        }

    def _get_fatigue_state_snapshot(self) -> dict:
        """Thread-safe snapshot of agent fatigue state for get_state()."""
        with self._fatigue_lock:
            snapshot = {
                aid: {
                    "losses": st["losses"],
                    "fatigued": st["fatigued"],
                    "rotation_idx": st["rotation_idx"],
                    "age_sec": int(time.time() - st["start_time"]),
                }
                for aid, st in self._agent_fatigue.items()
            }
        return snapshot

    # ── Multi-Agent Swarm ─────────────────────────────────────

    def get_swarm_state(self) -> dict:
        """Return last signal per agent for frontend display."""
        with self._lock:
            return dict(self._swarm_signals)

    def _is_window_stale(self, window_id: int) -> bool:
        """Check if the window has already moved on (resolve happened)."""
        with self._lock:
            return self.window.window_id != window_id

    def _run_swarm_agents(self, window_id: int, price_data,
                          ta: dict, news: str, ml_features: dict):
        """Run secondary agents in parallel. Each gets its own shadow bet.

        Called from _fetch_and_process_prediction (already in background thread).
        Includes stale-window guards to prevent zombie bets (SRE fix C1).
        """
        secondary = swarm.get_secondary_agents()
        if not secondary:
            return

        # ── Pre-flight check: window still active? ──
        if self._is_window_stale(window_id):
            log.warning(f"[SWARM] Window #{window_id} already closed before "
                        f"swarm launch — skipping")
            return

        # Build a generic user prompt (same data the primary sees)
        user_prompt = self.ai._build_prompt(
            price_data, ta, news,
            ml_features or {},
        )

        # Build the shared system prompt: RAG + lessons + HMM regime
        # All swarm agents inherit the same institutional context as the primary
        rag_block = ml_features.get("_rag_block", "") if ml_features else ""
        regime_snap = ml_features.get("_hmm_regime") if ml_features else None
        shared_system_prompt = self.ai.get_full_context_prompt(
            rag_block=rag_block, regime_snapshot=regime_snap
        )

        # Build fatigue overrides for fatigued agents
        agent_fatigue_prompts: dict[str, str] = {}
        for agent_cfg in secondary:
            aid = agent_cfg["agent_id"]
            fp = self._get_agent_fatigue_prompt(aid)
            if fp:
                agent_fatigue_prompts[aid] = fp

        log.info(f"[SWARM] Launching {len(secondary)} secondary agents "
                 f"for window #{window_id}"
                 f"{f' ({len(agent_fatigue_prompts)} fatigued)' if agent_fatigue_prompts else ''}")

        # Run all agents in parallel (swarm.run_swarm_predictions handles threads)
        results = swarm.run_swarm_predictions(
            user_prompt, ta, fatigue_prompts=agent_fatigue_prompts,
            shared_system_prompt=shared_system_prompt,
        )

        if not results:
            log.info("[SWARM] No secondary agents responded")
            return

        # ── Post-flight check: window still active after API calls? ──
        if self._is_window_stale(window_id):
            log.warning(f"[SWARM] Window #{window_id} closed while agents "
                        f"were thinking — discarding {len(results)} results "
                        f"to prevent zombie bets")
            return

        # Compute weighted consensus: weight each agent's vote by recent accuracy
        # Agents with higher win rate get more influence on majority
        agent_weights = {}
        for aid in results:
            port = db.get_agent_portfolio(aid)
            if port:
                total = port.get("wins", 0) + port.get("losses", 0)
                wr = port["wins"] / total if total >= 5 else 0.5  # default 50% if <5 bets
            else:
                wr = 0.5
            agent_weights[aid] = max(0.1, wr)  # floor at 10% to avoid zero weight

        up_weight = sum(agent_weights[aid] for aid, s in results.items() if s.prediction == "UP")
        down_weight = sum(agent_weights[aid] for aid, s in results.items() if s.prediction == "DOWN")
        total_weight = up_weight + down_weight
        majority = "UP" if up_weight >= down_weight else "DOWN"
        majority_pct = round(max(up_weight, down_weight) / total_weight, 3) if total_weight > 0 else 0.5
        log.info(f"[SWARM] Weighted consensus: UP={up_weight:.2f} DOWN={down_weight:.2f} "
                 f"-> {majority} ({majority_pct:.1%})")

        for agent_id, signal in results.items():
            agent_cfg = next(
                (a for a in secondary if a["agent_id"] == agent_id), {}
            )
            strategy = agent_cfg.get("strategy", "unknown")

            # Determine active strategy label (rotation-aware) for this agent
            # Computed BEFORE the lock block so it can be stored as
            # "strategy_label" in _swarm_signals for the Grito de Fuego.
            _agent_strategy_label = self._get_agent_active_strategy(
                agent_id, strategy
            )

            # Store signal for frontend display AND as "bullet in chamber"
            # consumed later by _broadcast_swarm_fire when the primary sniper
            # fires. No bet is saved at T+0 — the swarm rides the primary scope.
            with self._lock:
                # Final per-agent guard (window could close between iterations)
                if self.window.window_id != window_id:
                    log.warning(f"[SWARM] Window #{window_id} closed mid-save "
                                f"— aborting remaining agents")
                    return
                fatigue_state = self._get_agent_fatigue(agent_id)
                self._swarm_signals[agent_id] = {
                    # ── UI / telemetry ──
                    "prediction": signal.prediction,
                    "confidence": signal.confidence,
                    "reasoning": signal.reasoning[:200],
                    "risk_score": signal.risk_score,
                    "strategy": strategy,
                    "display_name": agent_cfg.get("display_name", agent_id),
                    "timestamp": signal.timestamp,
                    "layer_alignment": signal.layer_alignment,
                    "fatigued": fatigue_state["fatigued"],
                    "fatigue_losses": fatigue_state["losses"],
                    # ── Bala en recámara (consumed by _broadcast_swarm_fire) ──
                    "signal_obj": signal,
                    "strategy_label": _agent_strategy_label,
                    "initial_balance": float(
                        agent_cfg.get("initial_balance",
                                      self._default_initial_balance())
                    ),
                    "agent_cfg_display": agent_cfg.get(
                        "display_name", agent_id
                    ),
                    "armed_at_ts": time.time(),
                    "fired": False,
                }

            # Save prediction to DB with ML enrichment features.
            # NOTE: The bet itself is deferred — it will be placed by
            # _broadcast_swarm_fire() at the moment the primary sniper
            # pulls the trigger, using the SAME Polymarket CLOB quote.
            # Push TA snapshot for this swarm agent + compute velocities
            self.ai.push_ta_snapshot(agent_id, ta)
            _agent_vel = self.ai.compute_ta_velocities(agent_id)
            agent_ml = {
                "ai_latency_ms": getattr(signal, "latency_ms", None),
                "price_at_bet": price_data.price,
                "consensus_agreement": majority_pct,
                "layer_alignment": signal.layer_alignment,
                "api_tokens_used": getattr(signal, "usage_tokens", None),
                **_agent_vel,
            }
            try:
                db.save_prediction(
                    window_id,
                    signal.prediction,
                    signal.confidence,
                    signal.original_confidence,
                    signal.reasoning,
                    signal.risk_score,
                    ta,
                    ml_features=agent_ml,
                    ai_model=agent_id,
                    strategy_name=strategy,
                    rag_strategy=_agent_strategy_label,
                )
            except Exception as e:
                log.warning(f"[SWARM] DB save prediction failed for {agent_id}: {e}")

            log.info(f"[SWARM] {agent_id}: {signal.prediction} "
                     f"({signal.confidence}%) — ARMED "
                     f"(strategy={_agent_strategy_label})")

    def _resolve_swarm_bets(self, window_id: int, close_price: float,
                            outcome: str):
        """Resolve all secondary agent bets for a window.

        Excludes the primary agent — that bet is resolved separately
        in _resolve() to avoid double-counting balance updates.

        Includes ghost shadows: they get their outcome stamped (so the ML
        pipeline can learn "who was right") but the balance loop below
        skips ghosts via `if not is_ghost`. Data harvest stays intact.
        """
        conn = db._get_conn()
        try:
            rows = conn.execute(
                """SELECT ai_model, side, amount, odds_cents, is_ghost
                   FROM bets
                   WHERE window_id=?
                   AND ai_model != ?
                   AND outcome IS NULL""",
                (window_id, get_primary_agent_id()),
            ).fetchall()

            for row in rows:
                agent_id = row["ai_model"]
                won = row["side"] == outcome
                is_ghost = bool(row["is_ghost"])

                if is_ghost:
                    payout = 0
                    profit = 0
                else:
                    sw_pnl = calc_polymarket_pnl(
                        row["amount"], row["side"],
                        row["odds_cents"], won)
                    payout = sw_pnl["net_payout"]
                    profit = sw_pnl["net_profit"]

                db.close_bet(window_id, close_price, outcome,
                             payout, profit, won, ai_model=agent_id)
                db.update_prediction_outcome(
                    window_id, outcome, 0, ai_model=agent_id
                )

                # Update agent portfolio balance
                if not is_ghost:
                    db.update_agent_balance(agent_id, profit, won)

                log.info(f"[SWARM] {agent_id} resolved: "
                         f"{'WIN' if won else 'LOSS'} "
                         f"{row['side']}->{outcome} "
                         f"P&L:${profit:+,.2f}"
                         f"{' (ghost)' if is_ghost else ''}")
                # Track agent fatigue (wins/losses affect prompt rotation)
                self._update_agent_fatigue(agent_id, won)
        except Exception as e:
            log.warning(f"[SWARM] Resolve error: {e}")
