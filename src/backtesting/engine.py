"""
Backtesting Engine - Core simulation engine for Polymarket Spike Bot

The heart of the backtesting system. Simulates the spike hunting strategy
on historical data with realistic execution modeling.

Features:
- Realistic order fill simulation with slippage
- Position management with stop loss and take profit
- Multi-market support
- Detailed trade logging
- Equity curve tracking
"""

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

import structlog

from .models import (
    BacktestConfig,
    BacktestResult,
    Trade,
    TradeSide,
    TradeType,
    SignalReason,
    Position,
    PriceData,
    MarketData,
    EquityPoint,
)


logger = structlog.get_logger(__name__)


@dataclass
class PriceWindow:
    """Rolling window of prices for spike detection."""
    prices: deque = field(default_factory=lambda: deque(maxlen=1000))

    def add(self, timestamp: datetime, price: float) -> None:
        """Add a price point."""
        self.prices.append((timestamp, price))

    def get_oldest_in_window(self, current_time: datetime, window_seconds: float) -> float | None:
        """Get the oldest price within the time window."""
        window_start = current_time - timedelta(seconds=window_seconds)

        for ts, price in self.prices:
            if ts >= window_start:
                return price
        return None

    def get_price_n_points_ago(self, n: int = 1) -> float | None:
        """Get the price N data points ago (for minute-level backtesting)."""
        if len(self.prices) <= n:
            return None
        return self.prices[-(n + 1)][1]

    def get_data_interval_seconds(self) -> float | None:
        """Estimate the interval between data points."""
        if len(self.prices) < 2:
            return None
        # Use the last two points to estimate interval
        ts1, _ = self.prices[-2]
        ts2, _ = self.prices[-1]
        return (ts2 - ts1).total_seconds()

    def clear(self) -> None:
        """Clear the window."""
        self.prices.clear()


class BacktestEngine:
    """
    Core backtesting engine for spike hunting strategy.

    Simulates the strategy on historical data with realistic execution.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

        # Portfolio state
        self.cash = config.initial_capital
        self.initial_capital = config.initial_capital

        # Positions and trades
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.pending_trades: dict[str, Trade] = {}  # Trades awaiting fill

        # Price tracking for spike detection
        self.price_windows: dict[str, PriceWindow] = {}

        # Equity tracking
        self.equity_curve: list[EquityPoint] = []
        self.peak_equity = config.initial_capital

        # Statistics
        self.data_points_processed = 0
        self.signals_generated = 0
        self.trades_executed = 0
        self.trades_per_day: dict[str, int] = {}  # date -> count

        # Cooldowns (token_id -> last trade timestamp)
        self.cooldowns: dict[str, datetime] = {}
        self.cooldown_seconds = 30.0  # Cooldown between trades on same token

        # Callbacks for custom logic
        self.on_signal_callbacks: list[Callable] = []
        self.on_trade_callbacks: list[Callable] = []

        logger.info(
            "BacktestEngine initialized",
            initial_capital=config.initial_capital,
            spike_threshold=f"{config.spike_threshold_percent}%",
            spike_window=f"{config.spike_window_seconds}s",
        )

    def reset(self) -> None:
        """Reset engine state for a new backtest."""
        self.cash = self.config.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.pending_trades.clear()
        self.price_windows.clear()
        self.equity_curve.clear()
        self.peak_equity = self.config.initial_capital
        self.data_points_processed = 0
        self.signals_generated = 0
        self.trades_executed = 0
        self.trades_per_day.clear()
        self.cooldowns.clear()

    def run(self, market_data: dict[str, MarketData]) -> BacktestResult:
        """
        Run backtest on provided market data.

        Args:
            market_data: Dictionary of token_id -> MarketData

        Returns:
            BacktestResult with all trades and metrics
        """
        import time
        start_time = time.time()

        self.reset()

        # Combine all price data and sort by timestamp
        all_prices = []
        for token_id, data in market_data.items():
            for price in data.prices:
                all_prices.append((price.timestamp, token_id, price))

        all_prices.sort(key=lambda x: x[0])

        if not all_prices:
            logger.warning("No price data to backtest")
            return self._create_result(0.0)

        logger.info(
            "Starting backtest",
            tokens=len(market_data),
            data_points=len(all_prices),
            start=all_prices[0][0].isoformat(),
            end=all_prices[-1][0].isoformat(),
        )

        # Process each price point
        last_equity_update = None
        equity_update_interval = timedelta(minutes=15)

        for timestamp, token_id, price_data in all_prices:
            self._process_price_update(timestamp, token_id, price_data)
            self.data_points_processed += 1

            # Update equity curve periodically
            if last_equity_update is None or timestamp - last_equity_update >= equity_update_interval:
                self._update_equity_curve(timestamp)
                last_equity_update = timestamp

        # Close any remaining positions at end of data
        self._close_all_positions(all_prices[-1][0], SignalReason.END_OF_DATA)

        # Final equity update
        if all_prices:
            self._update_equity_curve(all_prices[-1][0])

        execution_time = time.time() - start_time

        result = self._create_result(execution_time)

        logger.info(
            "Backtest completed",
            total_trades=len(self.trades),
            final_equity=f"${result.final_equity:.2f}",
            total_return=f"{result.total_return_percent:.2f}%",
            execution_time=f"{execution_time:.2f}s",
        )

        return result

    def _process_price_update(
        self,
        timestamp: datetime,
        token_id: str,
        price_data: PriceData,
    ) -> None:
        """Process a single price update."""
        price = price_data.price

        # Initialize price window if needed
        if token_id not in self.price_windows:
            self.price_windows[token_id] = PriceWindow()

        # Add to price window
        self.price_windows[token_id].add(timestamp, price)

        # Update existing position
        if token_id in self.positions:
            self._update_position(token_id, price, timestamp)

        # Check for spike signal (only if no position and not on cooldown)
        elif self._can_trade(token_id, timestamp):
            signal = self._detect_spike(token_id, price, timestamp)
            if signal:
                self._execute_signal(signal, token_id, price, timestamp)

    def _detect_spike(
        self,
        token_id: str,
        current_price: float,
        timestamp: datetime,
    ) -> dict | None:
        """
        Detect if a price spike has occurred.

        Returns signal dict if spike detected, None otherwise.
        """
        window = self.price_windows.get(token_id)
        if not window or len(window.prices) < 2:
            return None

        # Check data interval to determine comparison method
        data_interval = window.get_data_interval_seconds()

        if data_interval is not None and data_interval > self.config.spike_window_seconds:
            # Data is at lower frequency (e.g., minute bars)
            # Use point-based comparison: compare to previous N data points
            # spike_window_seconds becomes "number of bars to look back"
            lookback_points = max(1, int(self.config.spike_window_seconds))
            reference_price = window.get_price_n_points_ago(lookback_points)
        else:
            # High-frequency data (tick level) - use time-based window
            reference_price = window.get_oldest_in_window(
                timestamp,
                self.config.spike_window_seconds,
            )

        if reference_price is None or reference_price <= 0:
            return None

        # Calculate percentage change
        change_percent = ((current_price - reference_price) / reference_price) * 100

        # Check if change exceeds threshold
        if abs(change_percent) < self.config.spike_threshold_percent:
            return None

        self.signals_generated += 1

        # Determine signal direction
        if change_percent > 0:
            # Price spiked UP -> expect reversion DOWN -> SELL (short)
            return {
                "type": SignalReason.SPIKE_UP,
                "side": TradeSide.SELL,
                "change_percent": change_percent,
                "reference_price": reference_price,
            }
        else:
            # Price spiked DOWN -> expect reversion UP -> BUY
            return {
                "type": SignalReason.SPIKE_DOWN,
                "side": TradeSide.BUY,
                "change_percent": change_percent,
                "reference_price": reference_price,
            }

    def _execute_signal(
        self,
        signal: dict,
        token_id: str,
        current_price: float,
        timestamp: datetime,
    ) -> None:
        """Execute a trading signal."""
        # Check position limit
        if len(self.positions) >= self.config.max_positions:
            return

        # Check daily trade limit
        date_key = timestamp.date().isoformat()
        daily_trades = self.trades_per_day.get(date_key, 0)
        if daily_trades >= self.config.max_trades_per_day:
            return

        # Calculate entry price with slippage
        slippage = current_price * (self.config.slippage_percent / 100)
        if signal["side"] == TradeSide.BUY:
            entry_price = current_price + slippage  # Pay more when buying
        else:
            entry_price = current_price - slippage  # Get less when selling

        # Calculate position size
        position_value = min(self.config.position_size_usdc, self.cash * 0.95)
        if position_value < 1:
            return  # Not enough capital

        size = position_value / entry_price

        # Calculate fees
        fees = position_value * (self.config.fee_percent / 100)

        # Check if we have enough cash
        total_cost = position_value + fees
        if total_cost > self.cash:
            return

        # Create trade
        trade_id = str(uuid.uuid4())[:8]
        trade = Trade(
            id=trade_id,
            timestamp=timestamp,
            token_id=token_id,
            side=signal["side"],
            trade_type=TradeType.ENTRY,
            signal_reason=signal["type"],
            entry_price=entry_price,
            size=size,
            position_value=position_value,
            fees=fees,
            slippage=abs(entry_price - current_price) * size,
            metadata={
                "change_percent": signal["change_percent"],
                "reference_price": signal["reference_price"],
            },
        )

        # Create position
        position = Position(
            token_id=token_id,
            side=signal["side"],
            entry_price=entry_price,
            size=size,
            entry_timestamp=timestamp,
            trade_id=trade_id,
            current_price=current_price,
        )

        # Update state
        self.cash -= total_cost
        self.positions[token_id] = position
        self.trades.append(trade)
        self.cooldowns[token_id] = timestamp
        self.trades_per_day[date_key] = daily_trades + 1
        self.trades_executed += 1

        # Callbacks
        for callback in self.on_trade_callbacks:
            try:
                callback(trade, position)
            except Exception:
                pass

        logger.debug(
            "Trade executed",
            trade_id=trade_id,
            token_id=token_id,
            side=signal["side"].value,
            entry_price=f"{entry_price:.4f}",
            size=f"{size:.2f}",
            signal=signal["type"].value,
        )

    def _update_position(
        self,
        token_id: str,
        current_price: float,
        timestamp: datetime,
    ) -> None:
        """Update existing position and check exit conditions."""
        position = self.positions.get(token_id)
        if not position:
            return

        position.update_price(current_price)

        # Check stop loss
        if position.unrealized_pnl_percent <= -self.config.stop_loss_percent:
            self._close_position(token_id, current_price, timestamp, SignalReason.STOP_LOSS)
            return

        # Check take profit
        if position.unrealized_pnl_percent >= self.config.take_profit_percent:
            self._close_position(token_id, current_price, timestamp, SignalReason.TAKE_PROFIT)
            return

        # Check position timeout
        hold_duration = (timestamp - position.entry_timestamp).total_seconds()
        if hold_duration >= self.config.position_timeout_seconds:
            self._close_position(token_id, current_price, timestamp, SignalReason.TIMEOUT)
            return

    def _close_position(
        self,
        token_id: str,
        current_price: float,
        timestamp: datetime,
        reason: SignalReason,
    ) -> None:
        """Close a position."""
        position = self.positions.get(token_id)
        if not position:
            return

        # Calculate exit price with slippage
        slippage = current_price * (self.config.slippage_percent / 100)
        if position.side == TradeSide.BUY:
            exit_price = current_price - slippage  # Get less when selling
        else:
            exit_price = current_price + slippage  # Pay more when covering

        # Calculate fees
        exit_value = position.size * exit_price
        fees = exit_value * (self.config.fee_percent / 100)

        # Find the entry trade and close it
        for trade in self.trades:
            if trade.id == position.trade_id and not trade.is_closed:
                trade.close(exit_price, timestamp, fees)
                trade.signal_reason = reason  # Update with exit reason
                break

        # Update cash
        if position.side == TradeSide.BUY:
            # We sold the position
            self.cash += exit_value - fees
        else:
            # We covered the short
            # For shorts: profit if exit < entry
            pnl = (position.entry_price - exit_price) * position.size
            self.cash += position.size * position.entry_price + pnl - fees

        # Remove position
        del self.positions[token_id]

        logger.debug(
            "Position closed",
            token_id=token_id,
            reason=reason.value,
            entry_price=f"{position.entry_price:.4f}",
            exit_price=f"{exit_price:.4f}",
            pnl_percent=f"{position.unrealized_pnl_percent:.2f}%",
        )

    def _close_all_positions(self, timestamp: datetime, reason: SignalReason) -> None:
        """Close all open positions."""
        for token_id in list(self.positions.keys()):
            position = self.positions[token_id]
            self._close_position(token_id, position.current_price, timestamp, reason)

    def _can_trade(self, token_id: str, timestamp: datetime) -> bool:
        """Check if trading is allowed for a token."""
        # Check cooldown
        if token_id in self.cooldowns:
            cooldown_end = self.cooldowns[token_id] + timedelta(seconds=self.cooldown_seconds)
            if timestamp < cooldown_end:
                return False

        # Check if we already have a position
        if token_id in self.positions:
            return False

        return True

    def _get_equity(self) -> float:
        """Calculate current total equity."""
        positions_value = sum(
            p.size * p.current_price for p in self.positions.values()
        )
        return self.cash + positions_value

    def _update_equity_curve(self, timestamp: datetime) -> None:
        """Update the equity curve."""
        equity = self._get_equity()
        positions_value = sum(
            p.size * p.current_price for p in self.positions.values()
        )

        # Calculate drawdown
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = self.peak_equity - equity
        drawdown_percent = (drawdown / self.peak_equity * 100) if self.peak_equity > 0 else 0

        self.equity_curve.append(EquityPoint(
            timestamp=timestamp,
            equity=equity,
            cash=self.cash,
            positions_value=positions_value,
            drawdown=drawdown,
            drawdown_percent=drawdown_percent,
        ))

    def _create_result(self, execution_time: float) -> BacktestResult:
        """Create the backtest result."""
        final_equity = self._get_equity()

        result = BacktestResult(
            config=self.config,
            execution_time_seconds=execution_time,
            data_points_processed=self.data_points_processed,
            trades=self.trades.copy(),
            final_equity=final_equity,
            final_cash=self.cash,
            equity_curve=self.equity_curve.copy(),
        )

        return result


class MultiTokenBacktestEngine(BacktestEngine):
    """
    Extended backtest engine optimized for multiple tokens.

    Adds features for:
    - Correlation analysis between tokens
    - Portfolio-level risk management
    - Token selection based on historical performance
    """

    def __init__(self, config: BacktestConfig):
        super().__init__(config)
        self.token_stats: dict[str, dict] = {}

    def run_with_analysis(
        self,
        market_data: dict[str, MarketData],
    ) -> tuple[BacktestResult, dict]:
        """
        Run backtest with additional per-token analysis.

        Returns:
            Tuple of (BacktestResult, token_analysis_dict)
        """
        result = self.run(market_data)

        # Analyze per-token performance
        token_analysis = {}
        for token_id in market_data.keys():
            token_trades = [t for t in result.trades if t.token_id == token_id]

            if not token_trades:
                continue

            winning = [t for t in token_trades if t.is_closed and t.pnl > 0]
            losing = [t for t in token_trades if t.is_closed and t.pnl < 0]

            total_pnl = sum(t.pnl for t in token_trades if t.is_closed)
            avg_hold_time = (
                sum(t.hold_duration_seconds for t in token_trades if t.is_closed)
                / len(token_trades)
                if token_trades else 0
            )

            token_analysis[token_id] = {
                "total_trades": len(token_trades),
                "winning_trades": len(winning),
                "losing_trades": len(losing),
                "win_rate": len(winning) / len(token_trades) * 100 if token_trades else 0,
                "total_pnl": total_pnl,
                "avg_pnl_per_trade": total_pnl / len(token_trades) if token_trades else 0,
                "avg_hold_time_seconds": avg_hold_time,
                "market_name": market_data[token_id].market_name,
            }

        return result, token_analysis


def run_quick_backtest(
    market_data: dict[str, MarketData],
    spike_threshold: float = 3.0,
    spike_window: float = 2.0,
    take_profit: float = 2.0,
    stop_loss: float = 5.0,
    initial_capital: float = 10000.0,
    position_size: float = 100.0,
) -> BacktestResult:
    """
    Quick helper function to run a backtest with common parameters.

    Args:
        market_data: Dictionary of token_id -> MarketData
        spike_threshold: Spike detection threshold (%)
        spike_window: Spike detection window (seconds)
        take_profit: Take profit threshold (%)
        stop_loss: Stop loss threshold (%)
        initial_capital: Starting capital
        position_size: Size per position in USDC

    Returns:
        BacktestResult
    """
    # Determine date range from data
    all_timestamps = []
    for data in market_data.values():
        all_timestamps.extend([p.timestamp for p in data.prices])

    if not all_timestamps:
        raise ValueError("No price data provided")

    start_date = min(all_timestamps)
    end_date = max(all_timestamps)

    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        token_ids=list(market_data.keys()),
        spike_threshold_percent=spike_threshold,
        spike_window_seconds=spike_window,
        take_profit_percent=take_profit,
        stop_loss_percent=stop_loss,
        initial_capital=initial_capital,
        position_size_usdc=position_size,
    )

    engine = BacktestEngine(config)
    return engine.run(market_data)
