"""
Configuration Module - Polymarket Spike Bot

Loads all environment variables and trading parameters using pydantic-settings.
Provides type validation and defaults for all configuration values.
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Trading environment."""
    PRODUCTION = "production"
    TESTNET = "testnet"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class PolymarketCredentials(BaseSettings):
    """
    Polymarket API credentials.

    These are loaded from environment variables prefixed with POLY_.
    """
    model_config = SettingsConfigDict(
        env_prefix="POLY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Required credentials
    private_key: SecretStr = Field(
        ...,
        description="Ethereum private key for signing transactions (with or without 0x prefix)"
    )

    # Optional API credentials (for authenticated endpoints)
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="Polymarket API key"
    )
    api_secret: Optional[SecretStr] = Field(
        default=None,
        description="Polymarket API secret"
    )
    api_passphrase: Optional[SecretStr] = Field(
        default=None,
        description="Polymarket API passphrase"
    )

    @field_validator("private_key", mode="before")
    @classmethod
    def validate_private_key(cls, v: str) -> str:
        """Ensure private key is properly formatted."""
        if not v:
            raise ValueError("Private key is required")
        # Remove 0x prefix if present for consistency
        key = v.strip()
        if key.startswith("0x"):
            key = key[2:]
        # Validate hex string
        if len(key) != 64:
            raise ValueError("Private key must be 64 hex characters (32 bytes)")
        try:
            int(key, 16)
        except ValueError:
            raise ValueError("Private key must be a valid hex string")
        return key


class TradingConfig(BaseSettings):
    """
    Trading strategy parameters.

    Loaded from environment variables prefixed with TRADING_.
    """
    model_config = SettingsConfigDict(
        env_prefix="TRADING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Mode
    paper_trading: bool = Field(
        default=True,
        description="If True, simulate orders without real execution (DRY RUN mode)"
    )

    # Strategy Parameters - Spike Detection
    spike_threshold_percent: float = Field(
        default=3.0,
        ge=0.5,
        le=20.0,
        description="Percentage change to trigger spike detection (e.g., 3.0 = 3%)"
    )
    spike_window_seconds: float = Field(
        default=2.0,
        ge=0.5,
        le=30.0,
        description="Time window for spike detection in seconds"
    )

    # Order Parameters
    order_amount_usdc: float = Field(
        default=10.0,
        ge=1.0,
        le=10000.0,
        description="Amount in USDC per trade"
    )
    limit_order_offset_percent: float = Field(
        default=0.5,
        ge=0.1,
        le=5.0,
        description="Offset from mid-price for limit orders (catching the reversion)"
    )

    # Take Profit / Target
    take_profit_percent: float = Field(
        default=2.0,
        ge=0.5,
        le=20.0,
        description="Take profit threshold as percentage"
    )

    # Markets to monitor (comma-separated token IDs or condition IDs)
    target_markets: str = Field(
        default="",
        description="Comma-separated list of market condition IDs to monitor"
    )

    @property
    def target_market_list(self) -> list[str]:
        """Parse target markets into a list."""
        if not self.target_markets:
            return []
        return [m.strip() for m in self.target_markets.split(",") if m.strip()]


class RiskConfig(BaseSettings):
    """
    Risk management parameters.

    Loaded from environment variables prefixed with RISK_.
    """
    model_config = SettingsConfigDict(
        env_prefix="RISK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Position Sizing
    max_portfolio_percent: float = Field(
        default=5.0,
        ge=1.0,
        le=50.0,
        description="Maximum percentage of portfolio to risk per trade"
    )
    max_open_positions: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum number of concurrent open positions"
    )

    # Stop Loss
    stop_loss_percent: float = Field(
        default=5.0,
        ge=1.0,
        le=50.0,
        description="Hard stop loss percentage per position"
    )

    # Cooldown
    cooldown_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=600.0,
        description="Minimum seconds between trades on same market"
    )
    max_trades_per_hour: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum trades per hour across all markets"
    )

    # Global limits
    daily_loss_limit_usdc: float = Field(
        default=100.0,
        ge=10.0,
        le=10000.0,
        description="Maximum daily loss in USDC before stopping"
    )

    # Volatility circuit breaker
    volatility_pause_threshold: float = Field(
        default=10.0,
        ge=5.0,
        le=50.0,
        description="If price moves more than X% in 1 minute, pause trading"
    )
    volatility_pause_duration_seconds: float = Field(
        default=300.0,
        ge=60.0,
        le=3600.0,
        description="Duration to pause after volatility trigger"
    )


class ConnectionConfig(BaseSettings):
    """
    Connection and network parameters.

    Loaded from environment variables prefixed with CONN_.
    """
    model_config = SettingsConfigDict(
        env_prefix="CONN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Environment = Field(
        default=Environment.PRODUCTION,
        description="Trading environment (production or testnet)"
    )

    # Polymarket endpoints
    clob_api_url: str = Field(
        default="https://clob.polymarket.com",
        description="Polymarket CLOB REST API URL"
    )
    ws_url: str = Field(
        default="wss://ws-subscriptions-clob.polymarket.com/ws",
        description="Polymarket WebSocket URL"
    )

    # Retry configuration
    max_retries: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum retry attempts for failed requests"
    )
    retry_base_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Base delay for exponential backoff"
    )
    retry_max_delay_seconds: float = Field(
        default=60.0,
        ge=10.0,
        le=300.0,
        description="Maximum delay between retries"
    )

    # Timeouts
    http_timeout_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="HTTP request timeout"
    )
    ws_ping_interval_seconds: float = Field(
        default=30.0,
        ge=10.0,
        le=120.0,
        description="WebSocket ping interval"
    )
    ws_reconnect_delay_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=60.0,
        description="Delay before WebSocket reconnection attempt"
    )

    # Chain ID (Polygon for Polymarket)
    chain_id: int = Field(
        default=137,
        description="Blockchain chain ID (137 for Polygon mainnet)"
    )


class LoggingConfig(BaseSettings):
    """
    Logging configuration.

    Loaded from environment variables prefixed with LOG_.
    """
    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )
    file_path: str = Field(
        default="logs/bot.log",
        description="Log file path"
    )
    max_bytes: int = Field(
        default=10_000_000,  # 10 MB
        ge=1_000_000,
        le=100_000_000,
        description="Maximum log file size before rotation"
    )
    backup_count: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of backup log files to keep"
    )
    json_format: bool = Field(
        default=False,
        description="Use JSON format for log output"
    )


class BotConfig(BaseSettings):
    """
    Main configuration aggregating all sub-configs.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Bot identification
    bot_name: str = Field(
        default="PolymarketSpikeBot",
        description="Bot identifier for logging"
    )

    # Sub-configurations
    credentials: PolymarketCredentials = Field(default_factory=PolymarketCredentials)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    connection: ConnectionConfig = Field(default_factory=ConnectionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def __init__(self, **kwargs):
        """Initialize with sub-configurations."""
        super().__init__(**kwargs)
        # Ensure log directory exists
        log_path = Path(self.logging.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def is_paper_trading(self) -> bool:
        """Convenience property for paper trading mode."""
        return self.trading.paper_trading

    def display_config(self) -> str:
        """Return a safe string representation (no secrets)."""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                    {self.bot_name} Configuration
╠══════════════════════════════════════════════════════════════╣
║ Mode: {'PAPER TRADING (DRY RUN)' if self.is_paper_trading else 'LIVE TRADING'}
║ Environment: {self.connection.environment.value}
╠══════════════════════════════════════════════════════════════╣
║ STRATEGY PARAMETERS:
║   Spike Threshold: {self.trading.spike_threshold_percent}%
║   Spike Window: {self.trading.spike_window_seconds}s
║   Order Amount: ${self.trading.order_amount_usdc} USDC
║   Take Profit: {self.trading.take_profit_percent}%
╠══════════════════════════════════════════════════════════════╣
║ RISK PARAMETERS:
║   Max Portfolio %: {self.risk.max_portfolio_percent}%
║   Stop Loss: {self.risk.stop_loss_percent}%
║   Cooldown: {self.risk.cooldown_seconds}s
║   Max Open Positions: {self.risk.max_open_positions}
║   Daily Loss Limit: ${self.risk.daily_loss_limit_usdc}
╠══════════════════════════════════════════════════════════════╣
║ CONNECTION:
║   CLOB API: {self.connection.clob_api_url}
║   WebSocket: {self.connection.ws_url}
╚══════════════════════════════════════════════════════════════╝
"""


@lru_cache()
def get_config() -> BotConfig:
    """
    Get cached configuration singleton.

    Uses LRU cache to ensure config is only loaded once.
    Call get_config.cache_clear() to reload.
    """
    return BotConfig()


def reload_config() -> BotConfig:
    """Force reload configuration from environment."""
    get_config.cache_clear()
    return get_config()
