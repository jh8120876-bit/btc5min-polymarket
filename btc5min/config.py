import os
import sys
import logging

from dotenv import load_dotenv

load_dotenv()

# ── API URLs ────────────────────────────────────────────────
# Primary AI provider is now configured dynamically in dynamic_rules.json
# (agents section, is_primary=true). No hardcoded API keys here.
POLYMARKET_WS_URL = "wss://ws-live-data.polymarket.com"

# ── Local LLM (LM Studio / Ollama) ─────────────────────────
# Default base URL used by UI dropdowns to discover local models
# when configuring a local-type agent (primary or swarm). Local agents
# themselves are wired through dynamic_rules.json.
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:1234/v1")

# ── Binance Data ───────────────────────────────────────────
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"
# ── Trading Parameters ──────────────────────────────────────
WINDOW_MINUTES = 5
MAX_BET_PCT = 0.05
MIN_CONFIDENCE = 45

# ── Paper Trading / Data Collection Mode ───────────────────
# When True: bot bets on 100% of windows (no skips), quality gates
# still label signals but don't block bets, manual bets don't persist to DB.
PAPER_TRADING_MODE = True

# ── Default starting balance for agents / Kelly sizing ──────
# Replaces the legacy hardcoded $100. Used as the paper-money
# base when LIVE_DYNAMIC_SIZING is OFF, as the default for newly
# registered agents, and as the fallback for the Global Reset
# button in the Config UI. Hot-overridable via the "defaults"
# section of dynamic_rules.json (see rules.get("defaults",...)).
DEFAULT_INITIAL_BALANCE = 25.0

# ── Live Dynamic Sizing toggle ──────────────────────────────
# When True: Kelly sizing reads the REAL USDC.e balance from the
# connected ClobExecutor (py-clob-client) instead of the simulated
# paper balance. Kept OFF by default — flip ON only when running
# live_mainnet and you want Kelly to scale with actual wallet $$.
# Toggleable from the Config UI / /api/toggle_live_dynamic_sizing.
LIVE_DYNAMIC_SIZING = False

# ── Explore Override (Data Harvest) ─────────────────────────
# When True AND PAPER_TRADING_MODE=True: every Risk Manager veto
# (Kelly rejection, drawdown halt, min balance, low confidence) is
# intercepted and recycled into a $1.00 flat explore bet. Purpose:
# keep the SQLite Judge dataset populated with adverse-regime data
# during micro-balance stress tests. Disable for live trading.
ENABLE_EXPLORE_OVERRIDE = True

# ── ML Judge (XGBoost Filter) ─────────────────────────────
# When True: predictions pass through the trained XGBoost model
# before placing a bet. Requires ml_judge_model.pkl.
ENABLE_ML_JUDGE = True

# ── Execution Mode ─────────────────────────────────────────
# TRADING_MODE governs how bets are placed:
#   offline_sim   — legacy SQLite simulation (calc_polymarket_pnl). Default.
#   paper_simmer  — legacy label, now aliased to offline_sim (Simmer deprecated).
#   live_mainnet  — real money on Polymarket CLOB (clob.polymarket.com)
#                   via py-clob-client (native L2 Polygon CTF integration).
TRADING_MODE = os.getenv("TRADING_MODE", "offline_sim")

# Ethereum private key (or EOA signing key derived from a delegated proxy
# wallet). py-clob-client uses this as the L2 transaction signer.
# WALLET_PRIVATE_KEY is the canonical env var; ETH_PRIVATE_KEY is a legacy
# alias kept so old .env files keep working.
ETH_PRIVATE_KEY = os.getenv("WALLET_PRIVATE_KEY", "") or os.getenv("ETH_PRIVATE_KEY", "")

# Polymarket Proxy/Funder wallet address (the public 0x... tied to the
# Google/Email login on polymarket.com). Required when the account is a
# delegated web wallet — py-clob-client will be initialized with
# signature_type=1 and this address as the `funder` parameter so the L2
# contracts know which Safe/Proxy holds the USDC.e balance.
POLYMARKET_PROXY_FUNDER = (
    os.getenv("POLYMARKET_PROXY_FUNDER", "")
    or os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
)
# CLOB endpoint URL (Polymarket native L2 CLOB).
POLYMARKET_CLOB_URL = os.getenv("POLYMARKET_CLOB_URL", "https://clob.polymarket.com")

# Polygon chain ID
CHAIN_ID_POLYGON = 137

# ── Observability (Prometheus) ─────────────────────────────
PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "false").lower() in ("true", "1", "yes")
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "9108"))

# ── Logging ─────────────────────────────────────────────────
# Dual sink: stdout (terminal) + rotating file ``logs/btc5min.log``.
# File rotates at ``LOG_MAX_BYTES`` (default 10 MB), keeps ``LOG_BACKUP_COUNT``
# archives (default 5) → ~60 MB max on disk. Errors also go to
# ``logs/errors.log`` (WARNING+) so you can grep fast without the noise.
# All knobs are env-overridable.
from logging.handlers import RotatingFileHandler

_LOG_DIR = os.getenv("LOG_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs"))
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))
_LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))
_LOG_TO_FILE = os.getenv("LOG_TO_FILE", "true").lower() in ("true", "1", "yes")

_log_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_root = logging.getLogger()
_root.setLevel(_LOG_LEVEL)
# Clear default handlers (e.g., basicConfig duplicates) to avoid double lines
for _h in list(_root.handlers):
    _root.removeHandler(_h)

_stream = logging.StreamHandler()
_stream.setFormatter(_log_fmt)
_root.addHandler(_stream)

if _LOG_TO_FILE:
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        _main_fh = RotatingFileHandler(
            os.path.join(_LOG_DIR, "btc5min.log"),
            maxBytes=_LOG_MAX_BYTES, backupCount=_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        _main_fh.setFormatter(_log_fmt)
        _main_fh.setLevel(_LOG_LEVEL)
        _root.addHandler(_main_fh)

        _err_fh = RotatingFileHandler(
            os.path.join(_LOG_DIR, "errors.log"),
            maxBytes=_LOG_MAX_BYTES, backupCount=_LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        _err_fh.setFormatter(_log_fmt)
        _err_fh.setLevel(logging.WARNING)
        _root.addHandler(_err_fh)
    except Exception as _e:
        sys.stderr.write(f"[LOG] Failed to attach file handlers: {_e}\n")

log = logging.getLogger("btc-predictor")
