"""
Main Module - Polymarket Spike Bot

Entry point that orchestrates all components:
- Configuration loading
- WebSocket connection and reconnection
- Strategy execution
- Web dashboard for monitoring
- Graceful shutdown handling
"""

import argparse
import asyncio
import os
import signal
import sys
from datetime import datetime, timezone
from typing import NoReturn

import structlog
import uvicorn

from .config import get_config, BotConfig
from .logger import setup_logging, TradingLogger
from .market_connector import MarketConnector, OrderBook
from .risk_manager import RiskManager
from .strategy_engine import MultiMarketStrategy, Signal
from .web import WebDashboard


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Polymarket Spike Bot - Mean reversion trading bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "market_url",
        nargs="?",
        type=str,
        help="Polymarket market URL (e.g., https://polymarket.com/event/sol-updown-15m-...)"
    )
    parser.add_argument(
        "--token-id", "-t",
        type=str,
        help="Token ID to trade (overrides TRADING_TARGET_MARKETS env var)"
    )
    parser.add_argument(
        "--web", "-w",
        action="store_true",
        help="Enable web dashboard"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Web dashboard port (default: 8080)"
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Force paper trading mode"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Force live trading mode"
    )
    return parser.parse_args()


# Global state for signal handling
_shutdown_event: asyncio.Event | None = None
_bot_instance: "SpikeBot | None" = None


class SpikeBot:
    """
    Main bot orchestrator.

    Manages the lifecycle of all components and coordinates
    the overall trading operation.
    """

    def __init__(self, config: BotConfig, enable_web: bool = True, web_port: int = 8080):
        self.config = config
        self.logger = TradingLogger(config)
        self.enable_web = enable_web
        self.web_port = web_port

        # Components (initialized in start())
        self.connector: MarketConnector | None = None
        self.risk_manager: RiskManager | None = None
        self.strategy: MultiMarketStrategy | None = None

        # Web dashboard
        self.dashboard: WebDashboard | None = None
        if enable_web:
            self.dashboard = WebDashboard(config)

        # State
        self._running = False
        self._shutdown_requested = False

        # Tasks
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Initialize and start the bot."""
        self.logger.log_startup()
        print(self.config.display_config())

        try:
            # Initialize connector
            self.connector = MarketConnector(self.config)
            await self.connector.connect()

            # Initialize risk manager
            self.risk_manager = RiskManager(self.config, self.connector)
            await self.risk_manager.initialize()

            # Initialize strategy
            self.strategy = MultiMarketStrategy(
                self.config,
                self.connector,
                self.risk_manager,
            )

            # Get markets to monitor
            target_markets = self.config.trading.target_market_list
            if not target_markets:
                # If no specific markets configured, fetch active markets
                structlog.get_logger().info("No target markets configured, fetching active markets...")
                try:
                    markets = await self.connector.get_markets(active_only=True, limit=10)
                    structlog.get_logger().debug("Fetched markets", count=len(markets))

                    # Extract token IDs from markets (Polymarket uses tokens for trading)
                    target_markets = []
                    for market in markets[:5]:
                        # Try different possible ID fields
                        token_id = (
                            market.get("condition_id") or
                            market.get("token_id") or
                            market.get("id") or
                            ""
                        )
                        # Also check for tokens array in market
                        if not token_id and "tokens" in market:
                            tokens = market.get("tokens", [])
                            if tokens and isinstance(tokens, list):
                                token_id = tokens[0].get("token_id", "")

                        if token_id:
                            target_markets.append(token_id)
                            structlog.get_logger().debug(
                                "Found market",
                                token_id=token_id,
                                question=market.get("question", "")[:50],
                            )
                except Exception as e:
                    structlog.get_logger().error("Failed to fetch markets", error=str(e))

            if not target_markets:
                structlog.get_logger().error("No markets available to monitor")
                return

            structlog.get_logger().info(
                "Starting market monitoring",
                markets=target_markets,
                count=len(target_markets),
            )

            # Connect WebSocket
            await self.connector.connect_websocket()

            # Start monitoring markets
            await self.strategy.start(target_markets)

            # Setup dashboard callbacks
            if self.dashboard:
                self.connector.on_orderbook_update(self._on_orderbook_for_dashboard)
                self.strategy.engine.on_signal(self._on_signal_for_dashboard)
                self.dashboard.on_start_trading(self._on_start_trading_from_dashboard)
                self.dashboard.on_stop_trading(self._on_stop_trading_from_dashboard)
                self.dashboard.on_config_change(self._on_config_change_from_dashboard)

                # Auto-set dashboard to trading mode with the first token
                if target_markets:
                    first_token = target_markets[0]
                    self.dashboard.current_token_id = first_token
                    self.dashboard.is_trading = True
                    self.dashboard.bot_status = "running"
                    self.dashboard.current_market = {
                        "token_id": first_token,
                        "question": f"Trading on token {first_token[:20]}...",
                        "status": "active",
                    }
                    structlog.get_logger().info(
                        "Dashboard auto-configured",
                        token_id=first_token[:30] + "...",
                    )

                self.dashboard.start()

            # Start background tasks
            self._running = True
            self._tasks = [
                asyncio.create_task(self._run_websocket_listener()),
                asyncio.create_task(self._run_maintenance()),
                asyncio.create_task(self._run_health_check()),
            ]

            # Start web server if enabled
            if self.enable_web and self.dashboard:
                self._tasks.append(asyncio.create_task(self._run_web_server()))

            structlog.get_logger().info(
                "Bot started successfully",
                paper_trading=self.config.is_paper_trading,
                markets_monitored=len(target_markets),
                web_dashboard=f"http://localhost:{self.web_port}" if self.enable_web else "disabled",
            )

            # Wait for shutdown
            await self._wait_for_shutdown()

        except Exception as e:
            structlog.get_logger().error("Bot startup failed", error=str(e))
            raise
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the bot gracefully."""
        if not self._running:
            return

        self._running = False
        self._shutdown_requested = True

        structlog.get_logger().info("Stopping bot...")

        # Cancel all orders before shutdown
        if self.connector:
            try:
                cancelled = await self.connector.cancel_all_orders()
                structlog.get_logger().info("Cancelled pending orders", count=cancelled)
            except Exception as e:
                structlog.get_logger().error("Error cancelling orders", error=str(e))

        # Cancel tasks
        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Close connections
        if self.connector:
            await self.connector.disconnect()

        self.logger.log_shutdown()

    async def _run_websocket_listener(self) -> None:
        """Run WebSocket listener with reconnection."""
        while self._running:
            try:
                await self.connector.listen_websocket()
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._shutdown_requested:
                    structlog.get_logger().error(
                        "WebSocket error, reconnecting",
                        error=str(e),
                    )
                    await asyncio.sleep(self.config.connection.ws_reconnect_delay_seconds)

    async def _run_maintenance(self) -> None:
        """Run periodic maintenance tasks."""
        while self._running:
            try:
                if self.strategy:
                    await self.strategy.run_maintenance()
            except asyncio.CancelledError:
                break
            except Exception as e:
                structlog.get_logger().error("Maintenance error", error=str(e))
                await asyncio.sleep(10)

    async def _run_health_check(self) -> None:
        """Run periodic health checks and status logging."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Every minute

                if self.risk_manager and self.strategy:
                    risk_status = self.risk_manager.get_risk_status()
                    strategy_stats = self.strategy.engine.get_statistics()

                    structlog.get_logger().info(
                        "Health check",
                        uptime="running",
                        positions=risk_status["open_positions"],
                        daily_pnl=f"${risk_status['daily_pnl']:.2f}",
                        signals=strategy_stats["signals_generated"],
                        orders=strategy_stats["orders_placed"],
                        active_orders=strategy_stats["active_orders"],
                        trading_enabled=risk_status["trading_enabled"],
                    )

                    # Update dashboard stats
                    if self.dashboard:
                        await self.dashboard.update_stats({
                            "daily_pnl": risk_status["daily_pnl"],
                            "total_trades": strategy_stats["orders_placed"],
                            "signals_detected": strategy_stats["signals_generated"],
                            "active_orders": strategy_stats["active_orders"],
                        })

            except asyncio.CancelledError:
                break
            except Exception as e:
                structlog.get_logger().error("Health check error", error=str(e))

    async def _on_orderbook_for_dashboard(self, orderbook: OrderBook) -> None:
        """Forward orderbook updates to dashboard."""
        if self.dashboard and orderbook.mid_price:
            await self.dashboard.update_market(orderbook.token_id, {
                "best_bid": orderbook.best_bid,
                "best_ask": orderbook.best_ask,
                "mid_price": orderbook.mid_price,
                "spread": orderbook.spread_percent,
            })

    async def _on_signal_for_dashboard(self, signal: Signal) -> None:
        """Forward signals to dashboard."""
        if self.dashboard:
            await self.dashboard.add_signal({
                "type": signal.type.value,
                "token_id": signal.token_id,
                "current_price": signal.current_price,
                "change_percent": signal.change_percent,
                "suggested_side": signal.suggested_side.value,
                "confidence": signal.confidence,
            })

    async def _on_start_trading_from_dashboard(self, token_id: str, config: dict) -> bool:
        """Handle trading start request from dashboard."""
        try:
            structlog.get_logger().info(
                "Starting trading from dashboard",
                token_id=token_id,
                config=config,
            )

            # Update config if provided
            if config:
                if "spike_threshold" in config:
                    self.config.trading.spike_threshold_percent = config["spike_threshold"]
                if "spike_window" in config:
                    self.config.trading.spike_window_seconds = config["spike_window"]
                if "order_amount" in config:
                    self.config.trading.order_amount_usdc = config["order_amount"]
                if "stop_loss" in config:
                    self.config.risk.stop_loss_percent = config["stop_loss"]
                if "take_profit" in config:
                    self.config.trading.take_profit_percent = config["take_profit"]

            # Subscribe to the new token
            if self.connector:
                await self.connector.subscribe_orderbook(token_id)

            # Add to strategy monitoring
            if self.strategy:
                await self.strategy.add_market(token_id)

            return True
        except Exception as e:
            structlog.get_logger().error("Failed to start trading", error=str(e))
            return False

    async def _on_stop_trading_from_dashboard(self) -> bool:
        """Handle trading stop request from dashboard."""
        try:
            structlog.get_logger().info("Stopping trading from dashboard")

            # Cancel all orders
            if self.connector:
                await self.connector.cancel_all_orders()

            return True
        except Exception as e:
            structlog.get_logger().error("Failed to stop trading", error=str(e))
            return False

    async def _on_config_change_from_dashboard(self, config: dict) -> None:
        """Handle configuration change from dashboard."""
        structlog.get_logger().info("Configuration updated from dashboard", config=config)

        # Update bot config
        if "spike_threshold" in config:
            self.config.trading.spike_threshold_percent = config["spike_threshold"]
        if "spike_window" in config:
            self.config.trading.spike_window_seconds = config["spike_window"]
        if "order_amount" in config:
            self.config.trading.order_amount_usdc = config["order_amount"]
        if "stop_loss" in config:
            self.config.risk.stop_loss_percent = config["stop_loss"]
        if "take_profit" in config:
            self.config.trading.take_profit_percent = config["take_profit"]
        if "max_positions" in config:
            self.config.risk.max_open_positions = config["max_positions"]

        # Update strategy engine if exists
        if self.strategy and hasattr(self.strategy, 'engine'):
            self.strategy.engine.config = self.config

    async def _run_web_server(self) -> None:
        """Run the web dashboard server."""
        if not self.dashboard:
            return

        config = uvicorn.Config(
            self.dashboard.app,
            host="0.0.0.0",
            port=self.web_port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)

        structlog.get_logger().info(
            "Web dashboard starting",
            url=f"http://localhost:{self.web_port}",
        )

        try:
            await server.serve()
        except asyncio.CancelledError:
            pass

    async def _wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        global _shutdown_event
        if _shutdown_event:
            await _shutdown_event.wait()


def setup_signal_handlers() -> asyncio.Event:
    """Setup signal handlers for graceful shutdown."""
    global _shutdown_event, _bot_instance

    _shutdown_event = asyncio.Event()

    def signal_handler(sig: int, frame) -> None:
        print(f"\nReceived signal {sig}, initiating shutdown...")
        if _shutdown_event:
            _shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    return _shutdown_event


async def fetch_token_from_url(url: str) -> str | None:
    """Fetch token ID from a Polymarket market URL."""
    import aiohttp
    import re

    # Extract slug from URL
    # Format: https://polymarket.com/event/sol-updown-15m-1766620800
    match = re.search(r'/event/([^/?]+)', url)
    if not match:
        print(f"Could not extract event slug from URL: {url}")
        return None

    slug = match.group(1)
    print(f"Fetching market info for: {slug}")

    try:
        async with aiohttp.ClientSession() as session:
            # Try Gamma API
            async with session.get(
                f"https://gamma-api.polymarket.com/events?slug={slug}",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        event = data[0]
                        markets = event.get("markets", [])
                        if markets:
                            # Get the first market's token
                            market = markets[0]
                            tokens = market.get("clobTokenIds", [])
                            if tokens:
                                token_id = tokens[0]
                                print(f"Found token ID: {token_id[:30]}...")
                                return token_id
                            # Try alternative field names
                            token_id = market.get("token_id") or market.get("tokenId")
                            if token_id:
                                print(f"Found token ID: {token_id[:30]}...")
                                return token_id
    except Exception as e:
        print(f"Error fetching market info: {e}")

    print("Could not find token ID for this market URL")
    return None


async def main(args=None) -> None:
    """Main entry point."""
    global _bot_instance

    # Apply command line arguments to environment before loading config
    if args:
        # Handle URL argument - fetch token ID from Polymarket
        if args.market_url and args.market_url.startswith("http"):
            token_id = await fetch_token_from_url(args.market_url)
            if token_id:
                os.environ["TRADING_TARGET_MARKETS"] = token_id
            else:
                print("Failed to get token ID from URL. Please provide --token-id directly.")
                sys.exit(1)
        elif args.token_id:
            os.environ["TRADING_TARGET_MARKETS"] = args.token_id
        if args.paper:
            os.environ["TRADING_PAPER_TRADING"] = "true"
        if args.live:
            os.environ["TRADING_PAPER_TRADING"] = "false"
        if args.web:
            os.environ["WEB_ENABLED"] = "true"
        if args.port:
            os.environ["WEB_PORT"] = str(args.port)

    # Load configuration
    config = get_config()

    # Setup logging
    setup_logging(config)

    # Setup signal handlers
    setup_signal_handlers()

    # Create and run bot
    _bot_instance = SpikeBot(config, enable_web=args.web if args else False, web_port=args.port if args else 8080)

    try:
        await _bot_instance.start()
    except KeyboardInterrupt:
        structlog.get_logger().info("Keyboard interrupt received")
    except Exception as e:
        structlog.get_logger().error("Fatal error", error=str(e))
        sys.exit(1)


def run() -> None:
    """
    Synchronous entry point for running the bot.

    Can be called from command line or as a module.
    """
    args = parse_args()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
