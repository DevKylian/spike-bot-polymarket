"""
Logging Module - Polymarket Spike Bot

Provides structured logging with file rotation and console output.
Uses structlog for structured logging with context binding.
"""

import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import structlog

from .config import BotConfig, LogLevel


def setup_logging(config: BotConfig) -> structlog.BoundLogger:
    """
    Configure logging for the bot.

    Sets up both console and file handlers with rotation.
    Returns a structlog logger with context binding capabilities.
    """
    log_config = config.logging

    # Ensure log directory exists
    log_path = Path(log_config.file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert LogLevel enum to logging level
    level = getattr(logging, log_config.level.value)

    # Configure standard library logging
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_config.file_path,
        maxBytes=log_config.max_bytes,
        backupCount=log_config.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)

    # Formatters
    if log_config.json_format:
        # JSON format for structured log aggregation
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s"}'
        )
    else:
        # Human-readable format
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Create and return the main logger
    logger = structlog.get_logger(config.bot_name)
    logger.info(
        "Logging initialized",
        level=log_config.level.value,
        file=log_config.file_path,
        json_format=log_config.json_format,
    )

    return logger


class TradingLogger:
    """
    Specialized logger for trading events.

    Provides structured logging for trades, positions, and market events.
    """

    def __init__(self, config: BotConfig):
        self.config = config
        self.logger = structlog.get_logger(f"{config.bot_name}.trading")
        self._trade_count = 0

    def log_trade_signal(
        self,
        market_id: str,
        signal_type: str,
        price: float,
        spike_percent: float,
        **extra: Any,
    ) -> None:
        """Log a trade signal detection."""
        self.logger.info(
            "Trade signal detected",
            market_id=market_id,
            signal_type=signal_type,
            price=price,
            spike_percent=f"{spike_percent:.2f}%",
            **extra,
        )

    def log_order_placed(
        self,
        market_id: str,
        order_id: str,
        side: str,
        price: float,
        size: float,
        is_paper: bool = False,
        **extra: Any,
    ) -> None:
        """Log an order placement."""
        self._trade_count += 1
        mode = "[PAPER]" if is_paper else "[LIVE]"
        self.logger.info(
            f"{mode} Order placed",
            market_id=market_id,
            order_id=order_id,
            side=side,
            price=price,
            size=size,
            trade_number=self._trade_count,
            **extra,
        )

    def log_order_filled(
        self,
        market_id: str,
        order_id: str,
        fill_price: float,
        fill_size: float,
        pnl: float | None = None,
        **extra: Any,
    ) -> None:
        """Log an order fill."""
        self.logger.info(
            "Order filled",
            market_id=market_id,
            order_id=order_id,
            fill_price=fill_price,
            fill_size=fill_size,
            pnl=f"${pnl:.2f}" if pnl is not None else None,
            **extra,
        )

    def log_order_cancelled(
        self,
        market_id: str,
        order_id: str,
        reason: str,
        **extra: Any,
    ) -> None:
        """Log an order cancellation."""
        self.logger.warning(
            "Order cancelled",
            market_id=market_id,
            order_id=order_id,
            reason=reason,
            **extra,
        )

    def log_position_opened(
        self,
        market_id: str,
        side: str,
        entry_price: float,
        size: float,
        **extra: Any,
    ) -> None:
        """Log a new position."""
        self.logger.info(
            "Position opened",
            market_id=market_id,
            side=side,
            entry_price=entry_price,
            size=size,
            **extra,
        )

    def log_position_closed(
        self,
        market_id: str,
        exit_price: float,
        pnl: float,
        pnl_percent: float,
        reason: str,
        **extra: Any,
    ) -> None:
        """Log position closure."""
        log_method = self.logger.info if pnl >= 0 else self.logger.warning
        log_method(
            "Position closed",
            market_id=market_id,
            exit_price=exit_price,
            pnl=f"${pnl:.2f}",
            pnl_percent=f"{pnl_percent:.2f}%",
            reason=reason,
            **extra,
        )

    def log_stop_loss_triggered(
        self,
        market_id: str,
        entry_price: float,
        exit_price: float,
        loss: float,
        **extra: Any,
    ) -> None:
        """Log a stop loss trigger."""
        self.logger.error(
            "STOP LOSS TRIGGERED",
            market_id=market_id,
            entry_price=entry_price,
            exit_price=exit_price,
            loss=f"-${abs(loss):.2f}",
            **extra,
        )

    def log_risk_limit_hit(
        self,
        limit_type: str,
        current_value: float,
        limit_value: float,
        **extra: Any,
    ) -> None:
        """Log a risk limit breach."""
        self.logger.error(
            "Risk limit hit - trading paused",
            limit_type=limit_type,
            current_value=current_value,
            limit_value=limit_value,
            **extra,
        )

    def log_market_update(
        self,
        market_id: str,
        best_bid: float,
        best_ask: float,
        mid_price: float,
        **extra: Any,
    ) -> None:
        """Log market data update (debug level)."""
        self.logger.debug(
            "Market update",
            market_id=market_id,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price,
            spread=f"{(best_ask - best_bid) * 100:.2f}%",
            **extra,
        )

    def log_connection_event(
        self,
        event_type: str,
        endpoint: str,
        **extra: Any,
    ) -> None:
        """Log connection events."""
        self.logger.info(
            f"Connection {event_type}",
            endpoint=endpoint,
            **extra,
        )

    def log_error(
        self,
        error_type: str,
        message: str,
        exception: Exception | None = None,
        **extra: Any,
    ) -> None:
        """Log an error."""
        self.logger.error(
            f"Error: {error_type}",
            message=message,
            exception=str(exception) if exception else None,
            **extra,
        )

    def log_startup(self) -> None:
        """Log bot startup."""
        mode = "PAPER TRADING" if self.config.is_paper_trading else "LIVE TRADING"
        self.logger.info(
            "=" * 60,
        )
        self.logger.info(
            f"Bot starting in {mode} mode",
            timestamp=datetime.now(timezone.utc).isoformat(),
            config_summary={
                "spike_threshold": f"{self.config.trading.spike_threshold_percent}%",
                "spike_window": f"{self.config.trading.spike_window_seconds}s",
                "order_amount": f"${self.config.trading.order_amount_usdc}",
                "stop_loss": f"{self.config.risk.stop_loss_percent}%",
            },
        )

    def log_shutdown(self, reason: str = "normal") -> None:
        """Log bot shutdown."""
        self.logger.info(
            "Bot shutting down",
            reason=reason,
            total_trades=self._trade_count,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.logger.info("=" * 60)
