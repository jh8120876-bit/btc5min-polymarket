from dataclasses import dataclass, field


@dataclass
class PriceData:
    price: float
    timestamp: float
    source: str = "chainlink"


@dataclass
class AISignal:
    prediction: str
    confidence: int
    reasoning: str
    news_summary: str
    risk_score: str
    suggested_bet_pct: float
    timestamp: float = 0.0
    updated_at: float = 0.0
    update_count: int = 0
    original_confidence: int = 0
    signal_quality: str = "NORMAL"  # NORMAL / LOW_QUALITY / HIGH_QUALITY
    layer_alignment: str = ""  # "C1=UP, C2=DOWN, C3=N/A -> 1/3"


@dataclass
class WindowState:
    window_id: int = 0
    open_price: float = 0.0
    current_price: float = 0.0
    open_time: float = 0.0
    end_time: float = 0.0
    is_active: bool = False
    up_odds: int = 50
    down_odds: int = 50


@dataclass
class Bet:
    side: str
    amount: float
    price_cents: float      # Token price in cents (e.g., 53.5 for $0.535)
    open_price: float
    window_id: int
    timestamp: float
    is_auto: bool = True


@dataclass
class BetResult:
    window_id: int
    side: str
    amount: float
    price_cents: float
    open_price: float
    close_price: float
    outcome: str
    payout: float
    profit: float
    won: bool
    timestamp: str
    ai_confidence: int = 0
    ai_prediction: str = ""
    is_auto: bool = True
    rag_strategy: str = ""
    ml_oracle_prob: float | None = None
    ml_skipped: bool = False
    is_ghost: bool = False
