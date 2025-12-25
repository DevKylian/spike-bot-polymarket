"""
Risk Manager Module - Polymarket Spike Bot

Critical module for managing trading risk:
- Position sizing limits
- Stop loss enforcement
- Cooldown management
- Daily loss limits
- Volatility circuit breakers
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import structlog

from .config import BotConfig, RiskConfig
from .market_connector import MarketConnector, OrderSide, Position


logger = structlog.get_logger(__name__)


class RiskStatus(str, Enum):
    """Risk check status."""
    APPROVED = "approved"
    REJECTED = "rejected"
    PAUSED = "paused"


@dataclass
class RiskCheckResult:
    """Result of a risk check."""
    status: RiskStatus
    reason: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    timestamp: datetime
    market_id: str
    side: OrderSide
    entry_price: float
    exit_price: float | None = None
    size: float = 0.0
    pnl: float = 0.0
    is_closed: bool = False


@dataclass
class PriceSnapshot:
    """Price snapshot for volatility tracking."""
    timestamp: datetime
    price: float


class RiskManager:
    """
    Manages all trading risk controls.

    Enforces:
    - Maximum portfolio allocation per trade
    - Maximum open positions
    - Stop loss per position
    - Trade cooldowns per market
    - Daily loss limits
    - Volatility circuit breakers
    """

    def __init__(self, config: BotConfig, connector: MarketConnector):
        self.config = config
        self.risk_config: RiskConfig = config.risk
        self.connector = connector

        # Portfolio tracking
        self._portfolio_value: float = 0.0
        self._available_balance: float = 0.0

        # Position tracking
        self._positions: dict[str, Position] = {}

        # Trade history
        self._trade_history: deque[TradeRecord] = deque(maxlen=1000)
        self._trades_this_hour: deque[datetime] = deque(maxlen=100)

        # Cooldown tracking (market_id -> last trade timestamp)
        self._cooldowns: dict[str, datetime] = {}

        # Daily P&L tracking
        self._daily_pnl: float = 0.0
        self._daily_reset_date: datetime = datetime.now(timezone.utc).date()

        # Volatility tracking (token_id -> price history)
        self._price_history: dict[str, deque[PriceSnapshot]] = {}
        self._volatility_pause_until: datetime | None = None

        # Global trading pause
        self._trading_paused: bool = False
        self._pause_reason: str = ""

        logger.info(
            "RiskManager initialized",
            max_portfolio_pct=self.risk_config.max_portfolio_percent,
            stop_loss_pct=self.risk_config.stop_loss_percent,
            daily_loss_limit=self.risk_config.daily_loss_limit_usdc,
        )

    async def initialize(self) -> None:
        """Initialize risk manager with current account state."""
        try:
            # Fetch current balance
            balance = await self.connector.get_balance()
            self._available_balance = float(balance.get("available", 0))
            self._portfolio_value = float(balance.get("total", self._available_balance))

            # Fetch current positions
            positions = await self.connector.get_positions()
            for pos in positions:
                position = Position(
                    market_id=pos.get("market", ""),
                    token_id=pos.get("tokenId", ""),
                    side=OrderSide.BUY if float(pos.get("size", 0)) > 0 else OrderSide.SELL,
                    size=abs(float(pos.get("size", 0))),
                    entry_price=float(pos.get("avgPrice", 0)),
                    current_price=float(pos.get("currentPrice", 0)),
                )
                self._positions[position.token_id] = position

            logger.info(
                "Risk manager initialized",
                portfolio_value=self._portfolio_value,
                available_balance=self._available_balance,
                open_positions=len(self._positions),
            )

        except Exception as e:
            logger.warning("Could not fetch balance from API", error=str(e))
            # Use fallback balance - in live mode, we'll use order amount as reference
            if self.config.is_paper_trading:
                self._portfolio_value = 10000.0  # Paper trading balance
                self._available_balance = 10000.0
                logger.info("Using paper trading balance", balance=10000.0)
            else:
                # In live mode, use a reasonable default based on order amount
                # This allows trading to proceed even if balance API fails
                fallback = self.config.trading.order_amount_usdc * 20  # Assume 20x order size
                self._portfolio_value = fallback
                self._available_balance = fallback
                logger.info("Using fallback balance for live trading", balance=fallback)

    def _reset_daily_pnl_if_needed(self) -> None:
        """Reset daily P&L at midnight UTC."""
        today = datetime.now(timezone.utc).date()
        if today > self._daily_reset_date:
            logger.info(
                "Resetting daily P&L",
                previous_pnl=self._daily_pnl,
                previous_date=self._daily_reset_date,
            )
            self._daily_pnl = 0.0
            self._daily_reset_date = today

    def update_price(self, token_id: str, price: float) -> None:
        """Update price history for volatility tracking."""
        now = datetime.now(timezone.utc)

        if token_id not in self._price_history:
            self._price_history[token_id] = deque(maxlen=120)  # 2 minutes at 1/sec

        self._price_history[token_id].append(PriceSnapshot(timestamp=now, price=price))

        # Check volatility
        self._check_volatility(token_id)

    def _check_volatility(self, token_id: str) -> None:
        """Check if price volatility triggers a pause."""
        history = self._price_history.get(token_id)
        if not history or len(history) < 10:
            return

        now = datetime.now(timezone.utc)
        one_minute_ago = now - timedelta(minutes=1)

        # Get prices from last minute
        recent_prices = [s.price for s in history if s.timestamp >= one_minute_ago]
        if len(recent_prices) < 5:
            return

        min_price = min(recent_prices)
        max_price = max(recent_prices)

        if min_price <= 0:
            return

        volatility_pct = ((max_price - min_price) / min_price) * 100

        if volatility_pct >= self.risk_config.volatility_pause_threshold:
            pause_duration = timedelta(seconds=self.risk_config.volatility_pause_duration_seconds)
            self._volatility_pause_until = now + pause_duration

            logger.warning(
                "Volatility circuit breaker triggered",
                token_id=token_id,
                volatility_pct=f"{volatility_pct:.2f}%",
                threshold=f"{self.risk_config.volatility_pause_threshold}%",
                pause_until=self._volatility_pause_until.isoformat(),
            )

    def can_trade(self) -> RiskCheckResult:
        """
        Check if trading is currently allowed.

        Returns overall trading status without checking a specific trade.
        """
        self._reset_daily_pnl_if_needed()

        # Check global pause
        if self._trading_paused:
            return RiskCheckResult(
                status=RiskStatus.PAUSED,
                reason=f"Trading paused: {self._pause_reason}",
            )

        # Check volatility pause
        now = datetime.now(timezone.utc)
        if self._volatility_pause_until and now < self._volatility_pause_until:
            remaining = (self._volatility_pause_until - now).total_seconds()
            return RiskCheckResult(
                status=RiskStatus.PAUSED,
                reason=f"Volatility pause active ({remaining:.0f}s remaining)",
            )

        # Check daily loss limit
        if abs(self._daily_pnl) >= self.risk_config.daily_loss_limit_usdc and self._daily_pnl < 0:
            return RiskCheckResult(
                status=RiskStatus.REJECTED,
                reason="Daily loss limit reached",
                details={"daily_pnl": self._daily_pnl, "limit": self.risk_config.daily_loss_limit_usdc},
            )

        # Check hourly trade limit
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)
        recent_trades = [t for t in self._trades_this_hour if t >= hour_ago]
        self._trades_this_hour = deque(recent_trades, maxlen=100)

        if len(recent_trades) >= self.risk_config.max_trades_per_hour:
            return RiskCheckResult(
                status=RiskStatus.REJECTED,
                reason="Hourly trade limit reached",
                details={"trades_this_hour": len(recent_trades), "limit": self.risk_config.max_trades_per_hour},
            )

        return RiskCheckResult(status=RiskStatus.APPROVED)

    def check_trade(
        self,
        token_id: str,
        side: OrderSide,
        size: float,
        price: float,
    ) -> RiskCheckResult:
        """
        Check if a specific trade is allowed.

        Validates:
        - Overall trading status
        - Position sizing
        - Maximum positions
        - Cooldown
        """
        # First check overall trading status
        overall_check = self.can_trade()
        if overall_check.status != RiskStatus.APPROVED:
            return overall_check

        # Check position sizing
        trade_value = size * price
        max_trade_value = self._portfolio_value * (self.risk_config.max_portfolio_percent / 100)

        if trade_value > max_trade_value:
            return RiskCheckResult(
                status=RiskStatus.REJECTED,
                reason="Trade exceeds position size limit",
                details={
                    "trade_value": trade_value,
                    "max_allowed": max_trade_value,
                    "portfolio_value": self._portfolio_value,
                },
            )

        # Check balance
        if trade_value > self._available_balance:
            return RiskCheckResult(
                status=RiskStatus.REJECTED,
                reason="Insufficient balance",
                details={"trade_value": trade_value, "available": self._available_balance},
            )

        # Check maximum open positions
        if len(self._positions) >= self.risk_config.max_open_positions:
            # Allow if this is closing an existing position
            if token_id not in self._positions:
                return RiskCheckResult(
                    status=RiskStatus.REJECTED,
                    reason="Maximum open positions reached",
                    details={
                        "open_positions": len(self._positions),
                        "max_allowed": self.risk_config.max_open_positions,
                    },
                )

        # Check cooldown
        now = datetime.now(timezone.utc)
        if token_id in self._cooldowns:
            last_trade = self._cooldowns[token_id]
            cooldown_end = last_trade + timedelta(seconds=self.risk_config.cooldown_seconds)

            if now < cooldown_end:
                remaining = (cooldown_end - now).total_seconds()
                return RiskCheckResult(
                    status=RiskStatus.REJECTED,
                    reason=f"Cooldown active ({remaining:.1f}s remaining)",
                    details={"last_trade": last_trade.isoformat(), "cooldown_ends": cooldown_end.isoformat()},
                )

        return RiskCheckResult(status=RiskStatus.APPROVED)

    def record_trade(
        self,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
    ) -> None:
        """Record a trade execution."""
        now = datetime.now(timezone.utc)

        # Update cooldown
        self._cooldowns[token_id] = now

        # Record trade timestamp for hourly limit
        self._trades_this_hour.append(now)

        # Create trade record
        trade = TradeRecord(
            timestamp=now,
            market_id=token_id,
            side=side,
            entry_price=price,
            size=size,
        )
        self._trade_history.append(trade)

        # Update position
        if token_id in self._positions:
            existing = self._positions[token_id]
            if existing.side != side:
                # Closing or reducing position
                # This is a simplification - real implementation would handle partial fills
                del self._positions[token_id]
        else:
            # New position
            self._positions[token_id] = Position(
                market_id=token_id,
                token_id=token_id,
                side=side,
                size=size,
                entry_price=price,
                current_price=price,
            )

        logger.debug(
            "Trade recorded",
            token_id=token_id,
            side=side.value,
            price=price,
            size=size,
            open_positions=len(self._positions),
        )

    def check_stop_loss(self, token_id: str, current_price: float) -> bool:
        """
        Check if stop loss is triggered for a position.

        Returns True if stop loss should be executed.
        """
        position = self._positions.get(token_id)
        if not position:
            return False

        # Calculate P&L percentage
        if position.side == OrderSide.BUY:
            pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
        else:
            pnl_pct = ((position.entry_price - current_price) / position.entry_price) * 100

        # Update position current price
        position.current_price = current_price
        position.unrealized_pnl = pnl_pct

        # Check stop loss
        if pnl_pct <= -self.risk_config.stop_loss_percent:
            logger.warning(
                "Stop loss triggered",
                token_id=token_id,
                entry_price=position.entry_price,
                current_price=current_price,
                pnl_pct=f"{pnl_pct:.2f}%",
                stop_loss_pct=f"-{self.risk_config.stop_loss_percent}%",
            )
            return True

        return False

    def check_take_profit(self, token_id: str, current_price: float) -> bool:
        """
        Check if take profit is triggered for a position.

        Returns True if take profit should be executed.
        """
        position = self._positions.get(token_id)
        if not position:
            return False

        # Calculate P&L percentage
        if position.side == OrderSide.BUY:
            pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
        else:
            pnl_pct = ((position.entry_price - current_price) / position.entry_price) * 100

        take_profit_pct = self.config.trading.take_profit_percent

        if pnl_pct >= take_profit_pct:
            logger.info(
                "Take profit triggered",
                token_id=token_id,
                entry_price=position.entry_price,
                current_price=current_price,
                pnl_pct=f"{pnl_pct:.2f}%",
                target_pct=f"{take_profit_pct}%",
            )
            return True

        return False

    def close_position(self, token_id: str, exit_price: float) -> float:
        """
        Close a position and calculate P&L.

        Returns the realized P&L.
        """
        position = self._positions.get(token_id)
        if not position:
            logger.warning("No position to close", token_id=token_id)
            return 0.0

        # Calculate P&L
        if position.side == OrderSide.BUY:
            pnl = (exit_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - exit_price) * position.size

        # Update daily P&L
        self._daily_pnl += pnl

        # Remove position
        del self._positions[token_id]

        # Update trade history
        for trade in reversed(self._trade_history):
            if trade.market_id == token_id and not trade.is_closed:
                trade.exit_price = exit_price
                trade.pnl = pnl
                trade.is_closed = True
                break

        logger.info(
            "Position closed",
            token_id=token_id,
            entry_price=position.entry_price,
            exit_price=exit_price,
            pnl=f"${pnl:.2f}",
            daily_pnl=f"${self._daily_pnl:.2f}",
        )

        return pnl

    def pause_trading(self, reason: str) -> None:
        """Pause all trading."""
        self._trading_paused = True
        self._pause_reason = reason
        logger.warning("Trading paused", reason=reason)

    def resume_trading(self) -> None:
        """Resume trading."""
        self._trading_paused = False
        self._pause_reason = ""
        logger.info("Trading resumed")

    def get_position(self, token_id: str) -> Position | None:
        """Get position for a token."""
        return self._positions.get(token_id)

    def get_all_positions(self) -> dict[str, Position]:
        """Get all open positions."""
        return self._positions.copy()

    def get_daily_pnl(self) -> float:
        """Get current daily P&L."""
        return self._daily_pnl

    def get_risk_status(self) -> dict[str, Any]:
        """Get current risk status summary."""
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)
        trades_last_hour = sum(1 for t in self._trades_this_hour if t >= hour_ago)

        volatility_pause_remaining = None
        if self._volatility_pause_until and now < self._volatility_pause_until:
            volatility_pause_remaining = (self._volatility_pause_until - now).total_seconds()

        return {
            "trading_enabled": not self._trading_paused,
            "pause_reason": self._pause_reason if self._trading_paused else None,
            "portfolio_value": self._portfolio_value,
            "available_balance": self._available_balance,
            "open_positions": len(self._positions),
            "max_positions": self.risk_config.max_open_positions,
            "daily_pnl": self._daily_pnl,
            "daily_loss_limit": self.risk_config.daily_loss_limit_usdc,
            "trades_last_hour": trades_last_hour,
            "max_trades_per_hour": self.risk_config.max_trades_per_hour,
            "volatility_pause_remaining": volatility_pause_remaining,
        }

    def calculate_position_size(self, price: float) -> float:
        """
        Calculate optimal position size based on risk parameters.

        Returns size in USDC.
        """
        # Use configured order amount or calculate from portfolio
        order_amount = self.config.trading.order_amount_usdc

        # Cap at maximum portfolio percentage
        max_amount = self._portfolio_value * (self.risk_config.max_portfolio_percent / 100)
        order_amount = min(order_amount, max_amount)

        # Cap at available balance
        order_amount = min(order_amount, self._available_balance)

        # Convert to size at given price
        if price > 0:
            size = order_amount / price
        else:
            size = 0

        return size
