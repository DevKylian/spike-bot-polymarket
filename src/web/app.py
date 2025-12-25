"""
Web Dashboard - Polymarket Spike Bot

FastAPI-based web interface for monitoring and controlling the bot.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import structlog

from ..config import BotConfig
from ..market_analyzer import MarketAnalyzer, AnalysisResult

logger = structlog.get_logger(__name__)


class AnalyzeRequest(BaseModel):
    """Request to analyze a market URL."""
    url: str


class ConfigUpdate(BaseModel):
    """Configuration update request."""
    spike_threshold: float | None = None
    spike_window: float | None = None
    order_amount: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    max_positions: int | None = None


class TradingStartRequest(BaseModel):
    """Request to start trading."""
    token_id: str
    market_info: dict | None = None

# Path to templates and static files
WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket client connected", total=len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WebSocket client disconnected", total=len(self.active_connections))

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        if not self.active_connections:
            return

        message_json = json.dumps(message)
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)


class WebDashboard:
    """
    Web dashboard for the trading bot.

    Provides:
    - Real-time market data visualization
    - Position and P&L monitoring
    - Bot control (start/stop/pause)
    - Trade history
    - Market URL analysis
    - Real-time configuration
    """

    def __init__(self, config: BotConfig):
        self.config = config
        self.app = create_app(self)
        self.manager = ConnectionManager()
        self.analyzer = MarketAnalyzer()

        # Bot state (will be updated by main bot)
        self.bot_status = "stopped"
        self.is_paper_trading = config.is_paper_trading
        self.is_trading = False

        # Current market focus (single market mode)
        self.current_market: dict | None = None
        self.current_token_id: str | None = None

        # Market data
        self.markets: dict[str, dict] = {}
        self.positions: dict[str, dict] = {}

        # Dynamic configuration (can be modified at runtime)
        self.runtime_config = {
            "spike_threshold": config.trading.spike_threshold_percent,
            "spike_window": config.trading.spike_window_seconds,
            "order_amount": config.trading.order_amount_usdc,
            "stop_loss": config.risk.stop_loss_percent,
            "take_profit": config.trading.take_profit_percent,
            "max_positions": config.risk.max_open_positions,
        }

        # Statistics
        self.stats = {
            "daily_pnl": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "signals_detected": 0,
            "active_orders": 0,
            "uptime_start": None,
        }

        # Trade history
        self.trade_history: list[dict] = []

        # Price history for charts (token_id -> list of {timestamp, price})
        self.price_history: dict[str, list[dict]] = {}

        # Callbacks for trading control
        self._on_start_trading: Callable[[str, dict], Coroutine[Any, Any, bool]] | None = None
        self._on_stop_trading: Callable[[], Coroutine[Any, Any, bool]] | None = None
        self._on_config_change: Callable[[dict], Coroutine[Any, Any, None]] | None = None

    def on_start_trading(self, callback: Callable[[str, dict], Coroutine[Any, Any, bool]]) -> None:
        """Register callback for when trading starts."""
        self._on_start_trading = callback

    def on_stop_trading(self, callback: Callable[[], Coroutine[Any, Any, bool]]) -> None:
        """Register callback for when trading stops."""
        self._on_stop_trading = callback

    def on_config_change(self, callback: Callable[[dict], Coroutine[Any, Any, None]]) -> None:
        """Register callback for configuration changes."""
        self._on_config_change = callback

    async def analyze_market_url(self, url: str) -> dict:
        """Analyze a Polymarket URL and return market info with viability."""
        try:
            result = await self.analyzer.analyze_from_url(url)
            if not result:
                return {"error": "Could not analyze market. Check the URL."}

            # Normalize tokens to always have token_id field
            normalized_tokens = []
            raw_tokens = result.market_info.tokens or []

            for i, token in enumerate(raw_tokens):
                # Try multiple possible field names for token ID
                token_id = (
                    token.get("token_id", "") or
                    token.get("tokenId", "") or
                    token.get("id", "") or
                    ""
                )
                outcome = token.get("outcome", "YES" if i == 0 else "NO")
                price = float(token.get("price", 0) or 0)

                normalized_tokens.append({
                    "token_id": token_id,
                    "outcome": outcome,
                    "price": price,
                })

            # If no token IDs found, try clobTokenIds from raw market data
            if not any(t["token_id"] for t in normalized_tokens):
                raw_market = result.market_info.raw_market or {}
                clob_token_ids = raw_market.get("clobTokenIds", [])

                # Parse if it's a JSON string
                if isinstance(clob_token_ids, str):
                    try:
                        clob_token_ids = json.loads(clob_token_ids)
                    except (json.JSONDecodeError, TypeError):
                        clob_token_ids = []

                if clob_token_ids and isinstance(clob_token_ids, list):
                    # Use clobTokenIds to populate token_id
                    for i, tid in enumerate(clob_token_ids):
                        if isinstance(tid, str) and len(tid) > 10:
                            if i < len(normalized_tokens):
                                normalized_tokens[i]["token_id"] = tid
                            else:
                                outcome = "YES" if i == 0 else "NO" if i == 1 else f"Option {i}"
                                price = result.market_info.yes_price if i == 0 else result.market_info.no_price
                                normalized_tokens.append({
                                    "token_id": tid,
                                    "outcome": outcome,
                                    "price": price,
                                })

            # Store as current market
            self.current_market = {
                "question": result.market_info.question,
                "category": result.market_info.category,
                "status": result.market_info.status.value,
                "condition_id": result.market_info.condition_id,
                "tokens": normalized_tokens,
                "yes_price": result.market_info.yes_price,
                "no_price": result.market_info.no_price,
                "volume": result.market_info.volume,
                "liquidity": result.market_info.liquidity,
                "viability": result.viability.value,
                "viability_score": result.viability_score,
                "liquidity_score": result.liquidity_score,
                "volume_score": result.volume_score,
                "spread_score": result.spread_score,
                "volatility_score": result.volatility_score,
                "warnings": result.warnings,
                "recommendations": result.recommendations,
                "suggested_order_size": result.suggested_order_size,
                "max_position": result.max_position,
                "suggested_spike_threshold": result.suggested_spike_threshold,
            }

            return self.current_market

        except Exception as e:
            logger.error("Market analysis failed", error=str(e))
            return {"error": f"Analysis failed: {str(e)}"}

    async def update_config(self, config: dict) -> dict:
        """Update runtime configuration."""
        for key, value in config.items():
            if key in self.runtime_config and value is not None:
                self.runtime_config[key] = value

        # Notify callback
        if self._on_config_change:
            await self._on_config_change(self.runtime_config)

        # Broadcast update
        await self.manager.broadcast({
            "type": "config_updated",
            "data": self.runtime_config,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return {"status": "ok", "config": self.runtime_config}

    async def start_trading_on_token(self, token_id: str, market_info: dict | None = None) -> dict:
        """Start trading on a specific token."""
        if self.is_trading:
            return {"error": "Already trading. Stop first."}

        self.current_token_id = token_id
        if market_info:
            self.current_market = market_info

        if self._on_start_trading:
            success = await self._on_start_trading(token_id, self.runtime_config)
            if success:
                self.is_trading = True
                self.bot_status = "running"
                await self.manager.broadcast({
                    "type": "trading_started",
                    "data": {"token_id": token_id},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                return {"status": "started", "token_id": token_id}
            else:
                return {"error": "Failed to start trading"}

        # If no callback, just update state
        self.is_trading = True
        self.bot_status = "running"
        return {"status": "started", "token_id": token_id}

    async def stop_trading_on_token(self) -> dict:
        """Stop trading."""
        if not self.is_trading:
            return {"error": "Not currently trading"}

        if self._on_stop_trading:
            await self._on_stop_trading()

        self.is_trading = False
        self.bot_status = "stopped"

        await self.manager.broadcast({
            "type": "trading_stopped",
            "data": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return {"status": "stopped"}

    def start(self):
        """Mark bot as started."""
        self.bot_status = "running"
        self.stats["uptime_start"] = datetime.now(timezone.utc).isoformat()

    def stop(self):
        """Mark bot as stopped."""
        self.bot_status = "stopped"

    def pause(self):
        """Mark bot as paused."""
        self.bot_status = "paused"

    async def update_market(self, token_id: str, data: dict):
        """Update market data and broadcast to clients."""
        self.markets[token_id] = {
            "token_id": token_id,
            "best_bid": data.get("best_bid"),
            "best_ask": data.get("best_ask"),
            "mid_price": data.get("mid_price"),
            "spread": data.get("spread"),
            "last_update": datetime.now(timezone.utc).isoformat(),
        }

        # Add to price history
        if token_id not in self.price_history:
            self.price_history[token_id] = []

        self.price_history[token_id].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": data.get("mid_price", 0),
        })

        # Keep only last 500 points per token
        if len(self.price_history[token_id]) > 500:
            self.price_history[token_id] = self.price_history[token_id][-500:]

        await self.broadcast_update("market", self.markets[token_id])

    async def update_position(self, token_id: str, position: dict):
        """Update position data."""
        self.positions[token_id] = position
        await self.broadcast_update("position", position)

    async def add_trade(self, trade: dict):
        """Add trade to history."""
        trade["timestamp"] = datetime.now(timezone.utc).isoformat()
        self.trade_history.insert(0, trade)

        # Keep only last 100 trades
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[:100]

        self.stats["total_trades"] += 1
        if trade.get("pnl", 0) > 0:
            self.stats["winning_trades"] += 1

        await self.broadcast_update("trade", trade)

    async def update_stats(self, stats: dict):
        """Update statistics."""
        self.stats.update(stats)
        await self.broadcast_update("stats", self.stats)

    async def add_signal(self, signal: dict):
        """Add signal detection event."""
        self.stats["signals_detected"] += 1
        await self.broadcast_update("signal", signal)

    async def broadcast_update(self, event_type: str, data: dict):
        """Broadcast update to all WebSocket clients."""
        message = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await self.manager.broadcast(message)

    def get_state(self) -> dict:
        """Get current state for initial page load."""
        return {
            "status": self.bot_status,
            "is_paper_trading": self.is_paper_trading,
            "is_trading": self.is_trading,
            "config": self.runtime_config,
            "current_market": self.current_market,
            "current_token_id": self.current_token_id,
            "stats": self.stats,
            "markets": self.markets,
            "positions": self.positions,
            "trade_history": self.trade_history[:20],
        }


def create_app(dashboard: "WebDashboard | None" = None) -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Polymarket Spike Bot",
        description="Trading bot dashboard",
        version="1.0.0",
    )

    # Ensure directories exist
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    # Mount static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Setup templates
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Serve main dashboard page."""
        state = dashboard.get_state() if dashboard else {}
        return templates.TemplateResponse(
            "dashboard.html",
            {"request": request, "state": state}
        )

    @app.get("/api/state")
    async def get_state():
        """Get current bot state."""
        if dashboard:
            return dashboard.get_state()
        return {"error": "Dashboard not initialized"}

    @app.get("/api/markets")
    async def get_markets():
        """Get all market data."""
        if dashboard:
            return {"markets": dashboard.markets}
        return {"markets": {}}

    @app.get("/api/positions")
    async def get_positions():
        """Get all positions."""
        if dashboard:
            return {"positions": dashboard.positions}
        return {"positions": {}}

    @app.get("/api/trades")
    async def get_trades():
        """Get trade history."""
        if dashboard:
            return {"trades": dashboard.trade_history}
        return {"trades": []}

    @app.get("/api/stats")
    async def get_stats():
        """Get statistics."""
        if dashboard:
            return {"stats": dashboard.stats}
        return {"stats": {}}

    @app.get("/api/price-history/{token_id}")
    async def get_price_history(token_id: str):
        """Get price history for a token."""
        if dashboard and token_id in dashboard.price_history:
            return {"history": dashboard.price_history[token_id]}
        return {"history": []}

    @app.post("/api/bot/pause")
    async def pause_bot():
        """Pause trading."""
        if dashboard:
            dashboard.pause()
            return {"status": "paused"}
        return {"error": "Dashboard not initialized"}

    @app.post("/api/bot/resume")
    async def resume_bot():
        """Resume trading."""
        if dashboard:
            dashboard.start()
            return {"status": "running"}
        return {"error": "Dashboard not initialized"}

    @app.post("/api/analyze")
    async def analyze_market(request: AnalyzeRequest):
        """Analyze a Polymarket URL."""
        if dashboard:
            return await dashboard.analyze_market_url(request.url)
        return {"error": "Dashboard not initialized"}

    @app.post("/api/config")
    async def update_config(request: ConfigUpdate):
        """Update trading configuration."""
        if dashboard:
            config_dict = request.model_dump(exclude_none=True)
            return await dashboard.update_config(config_dict)
        return {"error": "Dashboard not initialized"}

    @app.post("/api/trading/start")
    async def start_trading(request: TradingStartRequest):
        """Start trading on a specific token."""
        if dashboard:
            return await dashboard.start_trading_on_token(
                request.token_id,
                request.market_info
            )
        return {"error": "Dashboard not initialized"}

    @app.post("/api/trading/stop")
    async def stop_trading():
        """Stop trading."""
        if dashboard:
            return await dashboard.stop_trading_on_token()
        return {"error": "Dashboard not initialized"}

    @app.post("/api/backtest")
    async def run_backtest(request: Request):
        """Run a backtest with given parameters."""
        try:
            params = await request.json()

            token_id = params.get("token", "")
            days = params.get("days", 30)
            threshold = params.get("threshold", 3.0)
            window = params.get("window", 2.0)
            take_profit = params.get("take_profit", 2.0)
            stop_loss = params.get("stop_loss", 5.0)
            capital = params.get("capital", 10000)
            simulate = params.get("simulate", False)

            if not token_id:
                return {"error": "Token ID is required"}

            # Import backtesting modules
            from datetime import timedelta
            from src.backtesting.data_loader import HistoricalDataLoader
            from src.backtesting.engine import BacktestEngine, BacktestConfig
            from src.backtesting.models import MarketData

            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)

            async with HistoricalDataLoader(use_cache=True) as loader:
                is_simulated = False

                if simulate:
                    # Generate simulated data
                    prices = loader.generate_simulated_data(
                        start_date=start_date,
                        end_date=end_date,
                        initial_price=0.5,
                        volatility=0.02,
                        spike_probability=0.01,
                        spike_magnitude=0.15,
                        interval_minutes=1,
                    )
                    is_simulated = True
                else:
                    # Try to get real data with simulation fallback
                    prices, is_simulated = await loader.get_price_history_with_simulation(
                        token_id,
                        start_date,
                        end_date,
                        interval="1m",
                        use_simulation_if_no_data=True,
                    )

                if not prices:
                    return {"error": "No price data available"}

                # Create market data
                market_data = MarketData(
                    token_id=token_id,
                    market_name="Simulated Market" if is_simulated else "Market",
                    condition_id=token_id,
                    start_date=start_date,
                    end_date=end_date,
                    prices=prices,
                    ohlcv=[],
                    total_volume=sum(p.volume for p in prices),
                    avg_daily_volume=sum(p.volume for p in prices) / max(days, 1),
                    avg_spread=0.01,
                    liquidity_score=50 if is_simulated else 0,
                )

                # Create backtest config
                config = BacktestConfig(
                    start_date=start_date,
                    end_date=end_date,
                    token_ids=[token_id],
                    initial_capital=capital,
                    position_size_usdc=100.0,
                    spike_threshold_percent=threshold,
                    spike_window_seconds=window,
                    take_profit_percent=take_profit,
                    stop_loss_percent=stop_loss,
                    max_positions=3,
                    fee_percent=0.1,
                )

                # Run backtest
                engine = BacktestEngine(config)
                result = engine.run({token_id: market_data})

                # Build equity curve
                equity_curve = [capital]
                current = capital
                for trade in result.trades:
                    current += trade.pnl
                    equity_curve.append(current)

                return {
                    "total_return": result.metrics.get("total_return_percent", 0),
                    "total_trades": result.metrics.get("total_trades", 0),
                    "win_rate": result.metrics.get("win_rate_percent", 0),
                    "sharpe_ratio": result.metrics.get("sharpe_ratio", 0),
                    "total_pnl": result.metrics.get("total_pnl", 0),
                    "max_drawdown": result.metrics.get("max_drawdown_percent", 0),
                    "avg_win": result.metrics.get("avg_win", 0),
                    "avg_loss": result.metrics.get("avg_loss", 0),
                    "equity_curve": equity_curve,
                    "is_simulated": is_simulated,
                }

        except Exception as e:
            logger.error("Backtest error", error=str(e))
            return {"error": str(e)}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates."""
        if not dashboard:
            await websocket.close()
            return

        await dashboard.manager.connect(websocket)
        try:
            # Send initial state
            await websocket.send_json({
                "type": "init",
                "data": dashboard.get_state(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            # Keep connection alive and listen for commands
            while True:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=30.0
                    )
                    # Handle client commands if needed
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})

                except asyncio.TimeoutError:
                    # Send heartbeat
                    await websocket.send_json({"type": "heartbeat"})

        except WebSocketDisconnect:
            dashboard.manager.disconnect(websocket)
        except Exception as e:
            logger.error("WebSocket error", error=str(e))
            dashboard.manager.disconnect(websocket)

    return app
