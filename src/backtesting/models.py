"""
Backtesting Models - Data structures for backtesting

Defines all data models used throughout the backtesting engine:
- Historical price data (OHLCV, tick data)
- Trade records
- Backtest configuration and results
- Optimization results
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TradeType(str, Enum):
    """Type of trade."""
    ENTRY = "entry"
    EXIT = "exit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class TradeSide(str, Enum):
    """Trade side."""
    BUY = "buy"
    SELL = "sell"


class SignalReason(str, Enum):
    """Reason for trade signal."""
    SPIKE_UP = "spike_up"
    SPIKE_DOWN = "spike_down"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TIMEOUT = "timeout"
    END_OF_DATA = "end_of_data"


@dataclass
class PriceData:
    """
    Single price point with timestamp.

    Used for tick-level data from WebSocket or high-frequency data.
    """
    timestamp: datetime
    price: float
    volume: float = 0.0
    bid: float | None = None
    ask: float | None = None

    @property
    def mid_price(self) -> float:
        """Calculate mid price if bid/ask available."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.price

    @property
    def spread(self) -> float | None:
        """Calculate spread if bid/ask available."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None


@dataclass
class OHLCV:
    """
    OHLCV (Open, High, Low, Close, Volume) candlestick data.

    Standard format for historical price data at various timeframes.
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    trades: int = 0

    @property
    def typical_price(self) -> float:
        """Calculate typical price (HLC/3)."""
        return (self.high + self.low + self.close) / 3

    @property
    def range(self) -> float:
        """Calculate price range."""
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        """Check if candle is bullish."""
        return self.close > self.open

    @property
    def body_size(self) -> float:
        """Calculate candle body size."""
        return abs(self.close - self.open)


@dataclass
class Trade:
    """
    Record of a single trade in the backtest.

    Contains all information about entry, exit, and P&L.
    """
    id: str
    timestamp: datetime
    token_id: str
    side: TradeSide
    trade_type: TradeType
    signal_reason: SignalReason
    entry_price: float
    exit_price: float | None = None
    size: float = 0.0
    position_value: float = 0.0  # In USDC
    pnl: float = 0.0  # Realized P&L
    pnl_percent: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    hold_duration_seconds: float = 0.0
    exit_timestamp: datetime | None = None
    is_closed: bool = False
    metadata: dict = field(default_factory=dict)

    def close(self, exit_price: float, exit_timestamp: datetime, fees: float = 0.0) -> None:
        """Close the trade and calculate P&L."""
        self.exit_price = exit_price
        self.exit_timestamp = exit_timestamp
        self.is_closed = True
        self.fees += fees
        self.hold_duration_seconds = (exit_timestamp - self.timestamp).total_seconds()

        # Calculate P&L
        if self.side == TradeSide.BUY:
            raw_pnl = (exit_price - self.entry_price) * self.size
        else:
            raw_pnl = (self.entry_price - exit_price) * self.size

        self.pnl = raw_pnl - self.fees
        if self.position_value > 0:
            self.pnl_percent = (self.pnl / self.position_value) * 100


@dataclass
class Position:
    """Current open position during backtest."""
    token_id: str
    side: TradeSide
    entry_price: float
    size: float
    entry_timestamp: datetime
    trade_id: str
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0

    def update_price(self, current_price: float) -> None:
        """Update current price and unrealized P&L."""
        self.current_price = current_price

        if self.side == TradeSide.BUY:
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.size

        if self.entry_price > 0:
            if self.side == TradeSide.BUY:
                self.unrealized_pnl_percent = ((current_price - self.entry_price) / self.entry_price) * 100
            else:
                self.unrealized_pnl_percent = ((self.entry_price - current_price) / self.entry_price) * 100


@dataclass
class BacktestConfig:
    """
    Configuration for a backtest run.

    Contains all parameters that define how the backtest is executed.
    """
    # Data range
    start_date: datetime
    end_date: datetime
    token_ids: list[str] = field(default_factory=list)

    # Strategy parameters (spike detection)
    spike_threshold_percent: float = 3.0
    spike_window_seconds: float = 2.0
    limit_order_offset_percent: float = 0.5

    # Exit parameters
    take_profit_percent: float = 2.0
    stop_loss_percent: float = 5.0
    position_timeout_seconds: float = 300.0  # 5 minutes

    # Risk parameters
    initial_capital: float = 10000.0
    position_size_usdc: float = 100.0
    max_positions: int = 3
    max_trades_per_day: int = 20

    # Simulation settings
    slippage_percent: float = 0.1
    fee_percent: float = 0.0  # Polymarket has no fees
    min_spread_percent: float = 0.5  # Minimum spread to consider trade valid

    # Data resolution
    data_resolution_seconds: int = 1  # 1 second resolution

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "token_ids": self.token_ids,
            "spike_threshold_percent": self.spike_threshold_percent,
            "spike_window_seconds": self.spike_window_seconds,
            "limit_order_offset_percent": self.limit_order_offset_percent,
            "take_profit_percent": self.take_profit_percent,
            "stop_loss_percent": self.stop_loss_percent,
            "position_timeout_seconds": self.position_timeout_seconds,
            "initial_capital": self.initial_capital,
            "position_size_usdc": self.position_size_usdc,
            "max_positions": self.max_positions,
            "slippage_percent": self.slippage_percent,
            "fee_percent": self.fee_percent,
        }


@dataclass
class EquityPoint:
    """Point in equity curve."""
    timestamp: datetime
    equity: float
    cash: float
    positions_value: float
    drawdown: float = 0.0
    drawdown_percent: float = 0.0


@dataclass
class BacktestResult:
    """
    Complete results of a backtest run.

    Contains all trades, metrics, and analysis.
    """
    # Configuration
    config: BacktestConfig

    # Execution info
    run_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_time_seconds: float = 0.0
    data_points_processed: int = 0

    # Trades
    trades: list[Trade] = field(default_factory=list)

    # Portfolio state
    final_equity: float = 0.0
    final_cash: float = 0.0

    # Equity curve
    equity_curve: list[EquityPoint] = field(default_factory=list)

    # Performance metrics (populated by PerformanceMetrics)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def total_trades(self) -> int:
        """Total number of trades."""
        return len([t for t in self.trades if t.is_closed])

    @property
    def winning_trades(self) -> int:
        """Number of winning trades."""
        return len([t for t in self.trades if t.is_closed and t.pnl > 0])

    @property
    def losing_trades(self) -> int:
        """Number of losing trades."""
        return len([t for t in self.trades if t.is_closed and t.pnl < 0])

    @property
    def win_rate(self) -> float:
        """Win rate percentage."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    @property
    def total_pnl(self) -> float:
        """Total realized P&L."""
        return sum(t.pnl for t in self.trades if t.is_closed)

    @property
    def total_return_percent(self) -> float:
        """Total return as percentage of initial capital."""
        if self.config.initial_capital == 0:
            return 0.0
        return (self.total_pnl / self.config.initial_capital) * 100

    def get_summary(self) -> dict[str, Any]:
        """Get summary of backtest results."""
        return {
            "period": {
                "start": self.config.start_date.isoformat(),
                "end": self.config.end_date.isoformat(),
            },
            "performance": {
                "total_return": f"{self.total_return_percent:.2f}%",
                "total_pnl": f"${self.total_pnl:.2f}",
                "final_equity": f"${self.final_equity:.2f}",
            },
            "trades": {
                "total": self.total_trades,
                "winning": self.winning_trades,
                "losing": self.losing_trades,
                "win_rate": f"{self.win_rate:.1f}%",
            },
            "execution": {
                "data_points": self.data_points_processed,
                "time_seconds": f"{self.execution_time_seconds:.2f}",
            },
            **self.metrics,
        }


@dataclass
class ParameterSet:
    """A set of strategy parameters for optimization."""
    spike_threshold_percent: float
    spike_window_seconds: float
    take_profit_percent: float
    stop_loss_percent: float
    limit_order_offset_percent: float = 0.5
    position_size_usdc: float = 100.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "spike_threshold_percent": self.spike_threshold_percent,
            "spike_window_seconds": self.spike_window_seconds,
            "take_profit_percent": self.take_profit_percent,
            "stop_loss_percent": self.stop_loss_percent,
            "limit_order_offset_percent": self.limit_order_offset_percent,
            "position_size_usdc": self.position_size_usdc,
        }

    def __hash__(self):
        return hash((
            self.spike_threshold_percent,
            self.spike_window_seconds,
            self.take_profit_percent,
            self.stop_loss_percent,
            self.limit_order_offset_percent,
            self.position_size_usdc,
        ))


@dataclass
class OptimizationResult:
    """Results of parameter optimization."""
    best_params: ParameterSet
    best_result: BacktestResult
    all_results: list[tuple[ParameterSet, BacktestResult]] = field(default_factory=list)

    # Optimization stats
    total_combinations: int = 0
    optimization_time_seconds: float = 0.0

    # Parameter sensitivity analysis
    parameter_sensitivity: dict[str, list[tuple[float, float]]] = field(default_factory=dict)

    def get_top_n(self, n: int = 10, metric: str = "total_return_percent") -> list[tuple[ParameterSet, BacktestResult]]:
        """Get top N parameter sets by a given metric."""
        def get_metric(result: BacktestResult) -> float:
            if metric == "total_return_percent":
                return result.total_return_percent
            elif metric == "sharpe_ratio":
                return result.metrics.get("sharpe_ratio", 0.0)
            elif metric == "win_rate":
                return result.win_rate
            elif metric == "profit_factor":
                return result.metrics.get("profit_factor", 0.0)
            else:
                return result.total_pnl

        sorted_results = sorted(self.all_results, key=lambda x: get_metric(x[1]), reverse=True)
        return sorted_results[:n]


@dataclass
class MarketData:
    """Container for market historical data."""
    token_id: str
    market_name: str
    condition_id: str
    start_date: datetime
    end_date: datetime
    prices: list[PriceData] = field(default_factory=list)
    ohlcv: list[OHLCV] = field(default_factory=list)

    # Market metadata
    total_volume: float = 0.0
    avg_daily_volume: float = 0.0
    avg_spread: float = 0.0
    liquidity_score: float = 0.0

    @property
    def data_points(self) -> int:
        """Total number of data points."""
        return len(self.prices) + len(self.ohlcv)

    @property
    def duration_days(self) -> float:
        """Duration of data in days."""
        return (self.end_date - self.start_date).total_seconds() / 86400

    def get_price_at(self, timestamp: datetime) -> float | None:
        """Get price at or before a given timestamp."""
        if not self.prices:
            return None

        for price in reversed(self.prices):
            if price.timestamp <= timestamp:
                return price.price
        return None
