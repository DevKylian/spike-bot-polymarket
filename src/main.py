"""
Main Module - Polymarket Spike Bot

Entry point that orchestrates all components:
- Configuration loading
- WebSocket connection and reconnection
- Strategy execution
- Web dashboard for monitoring
- Graceful shutdown handling
"""

import asyncio
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


async def main() -> None:
    """Main entry point."""
    global _bot_instance

    # Load configuration
    config = get_config()

    # Setup logging
    setup_logging(config)

    # Setup signal handlers
    setup_signal_handlers()

    # Create and run bot
    _bot_instance = SpikeBot(config)

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
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
