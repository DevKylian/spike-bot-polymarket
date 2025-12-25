"""
Market Connector Module - Polymarket Spike Bot

Handles all communication with Polymarket CLOB API:
- REST API for orders and account info
- WebSocket for real-time order book updates
- Authentication using py-clob-client (official Polymarket client)
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine

import aiohttp
import structlog
from eth_account import Account

try:
    from aiohttp_socks import ProxyConnector
    SOCKS_AVAILABLE = True
except ImportError:
    SOCKS_AVAILABLE = False

# Official Polymarket CLOB client
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType as ClobOrderType
    from py_clob_client.order_builder.constants import BUY, SELL
    CLOB_CLIENT_AVAILABLE = True
except ImportError:
    CLOB_CLIENT_AVAILABLE = False
    ClobClient = None

from eth_account.messages import encode_defunct
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import BotConfig


logger = structlog.get_logger(__name__)


class OrderSide(str, Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type."""
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    GTC = "GTC"  # Good Till Cancelled
    FOK = "FOK"  # Fill or Kill
    GTD = "GTD"  # Good Till Date


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    MATCHED = "matched"
    LIVE = "live"


@dataclass
class OrderBookLevel:
    """Single level in the order book."""
    price: float
    size: float


@dataclass
class OrderBook:
    """Order book snapshot."""
    market_id: str
    token_id: str
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def best_bid(self) -> float | None:
        """Get best bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        """Get best ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> float | None:
        """Calculate mid price."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> float | None:
        """Calculate bid-ask spread."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_percent(self) -> float | None:
        """Calculate spread as percentage of mid price."""
        if self.spread is not None and self.mid_price is not None and self.mid_price > 0:
            return (self.spread / self.mid_price) * 100
        return None


@dataclass
class Order:
    """Order representation."""
    id: str
    market_id: str
    token_id: str
    side: OrderSide
    price: float
    size: float
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Position:
    """Position representation."""
    market_id: str
    token_id: str
    side: OrderSide
    size: float
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PolymarketAPIError(Exception):
    """Custom exception for API errors."""

    def __init__(self, message: str, status_code: int | None = None, response: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class MarketConnector:
    """
    Handles all Polymarket API interactions.

    Uses py-clob-client for authenticated operations (orders, positions)
    and direct REST/WebSocket for market data.
    """

    def __init__(self, config: BotConfig):
        self.config = config
        self.conn_config = config.connection
        self.credentials = config.credentials

        # Ethereum account for signing
        private_key = self.credentials.private_key.get_secret_value()
        if not private_key.startswith("0x"):
            private_key = f"0x{private_key}"
        self._private_key = private_key
        self.account = Account.from_key(private_key)
        self.address = self.account.address

        # Proxy configuration
        self._proxy_url: str | None = config.connection.proxy_url
        self._use_free_proxy: bool = config.connection.use_free_proxy

        # HTTP session for direct REST calls
        self._session: aiohttp.ClientSession | None = None

        # Official CLOB client (for authenticated operations)
        self._clob_client: ClobClient | None = None
        self._clob_initialized = False

        # WebSocket state
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._ws_connected = False
        self._ws_subscriptions: set[str] = set()
        self._reconnect_task: asyncio.Task | None = None

        # Callbacks for market data
        self._orderbook_callbacks: list[Callable[[OrderBook], Coroutine[Any, Any, None]]] = []

        # Local state
        self._orderbooks: dict[str, OrderBook] = {}
        self._open_orders: dict[str, Order] = {}

        # Track last orderbook update for debugging
        self._last_orderbook_update: datetime | None = None
        self._orderbook_update_count = 0

        logger.info(
            "MarketConnector initialized",
            address=self.address,
            api_url=self.conn_config.clob_api_url,
            proxy=self._proxy_url[:30] + "..." if self._proxy_url else "none",
            use_free_proxy=self._use_free_proxy,
            clob_client_available=CLOB_CLIENT_AVAILABLE,
            paper_trading=config.is_paper_trading,
        )

    async def __aenter__(self) -> "MarketConnector":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> None:
        """Initialize HTTP session and CLOB client."""
        if self._session is None:
            # Fetch free proxy if configured
            if self._use_free_proxy and not self._proxy_url:
                from .proxy_manager import get_working_proxy
                self._proxy_url = await get_working_proxy()
                if self._proxy_url:
                    logger.info("Using free proxy", proxy=self._proxy_url[:30] + "...")
                else:
                    logger.warning("No free proxy available, connecting directly")

            timeout = aiohttp.ClientTimeout(total=self.conn_config.http_timeout_seconds)

            # Use ProxyConnector for SOCKS proxies
            connector = None
            if self._proxy_url and (self._proxy_url.startswith("socks5://") or self._proxy_url.startswith("socks4://")):
                if SOCKS_AVAILABLE:
                    connector = ProxyConnector.from_url(self._proxy_url)
                    self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
                    self._is_socks_proxy = True
                else:
                    logger.warning("SOCKS proxy requested but aiohttp-socks not installed")
                    self._session = aiohttp.ClientSession(timeout=timeout)
                    self._is_socks_proxy = False
            else:
                self._session = aiohttp.ClientSession(timeout=timeout)
                self._is_socks_proxy = False

            logger.info("HTTP session created", proxy=self._proxy_url[:30] + "..." if self._proxy_url else "none")

        # Initialize CLOB client for real trading
        if not self.config.is_paper_trading and not self._clob_initialized:
            await self._initialize_clob_client()

    async def _initialize_clob_client(self) -> None:
        """Initialize the official Polymarket CLOB client."""
        if not CLOB_CLIENT_AVAILABLE:
            logger.error(
                "py-clob-client not available! Install with: pip install py-clob-client",
                available=False,
            )
            return

        try:
            # Chain ID: 137 for Polygon mainnet
            chain_id = self.conn_config.chain_id

            # Check if we have API credentials
            api_key = self.credentials.api_key.get_secret_value() if self.credentials.api_key else None
            api_secret = self.credentials.api_secret.get_secret_value() if self.credentials.api_secret else None
            api_passphrase = self.credentials.api_passphrase.get_secret_value() if self.credentials.api_passphrase else None

            if api_key and api_secret and api_passphrase:
                # Use existing API credentials
                creds = ApiCreds(
                    api_key=api_key,
                    api_secret=api_secret,
                    api_passphrase=api_passphrase,
                )
                self._clob_client = ClobClient(
                    host=self.conn_config.clob_api_url,
                    key=self._private_key,
                    chain_id=chain_id,
                    creds=creds,
                )
                logger.info("CLOB client initialized with API credentials")
            else:
                # Initialize without credentials - will need to derive/create them
                self._clob_client = ClobClient(
                    host=self.conn_config.clob_api_url,
                    key=self._private_key,
                    chain_id=chain_id,
                )

                # Try to derive API credentials
                try:
                    logger.info("Deriving API credentials from wallet...")
                    self._clob_client.set_api_creds(self._clob_client.derive_api_key())
                    logger.info("API credentials derived successfully")
                except Exception as e:
                    logger.warning(
                        "Could not derive API credentials. You may need to create them at https://clob.polymarket.com",
                        error=str(e),
                    )
                    # Try to create new credentials
                    try:
                        logger.info("Attempting to create new API credentials...")
                        self._clob_client.set_api_creds(self._clob_client.create_api_key())
                        logger.info("New API credentials created successfully")
                    except Exception as e2:
                        logger.error(
                            "Failed to create API credentials",
                            error=str(e2),
                        )

            self._clob_initialized = True

            # Verify connection
            try:
                # Test API connection
                ok = self._clob_client.get_ok()
                logger.info("CLOB API connection test", status="OK" if ok else "FAILED")
            except Exception as e:
                logger.warning("CLOB API test failed", error=str(e))

        except Exception as e:
            logger.error("Failed to initialize CLOB client", error=str(e))
            import traceback
            traceback.print_exc()

    async def disconnect(self) -> None:
        """Close all connections."""
        await self._close_websocket()
        if self._session:
            await self._session.close()
            self._session = None
            logger.info("HTTP session closed")

    def _generate_l1_signature(self, timestamp: int) -> str:
        """
        Generate L1 signature for API authentication.

        The L1 signature proves wallet ownership.
        """
        message = f"Polymarket Trading API\nTimestamp: {timestamp}"
        message_hash = encode_defunct(text=message)
        signed = self.account.sign_message(message_hash)
        return signed.signature.hex()

    def _get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers for API requests."""
        timestamp = int(time.time() * 1000)
        signature = self._generate_l1_signature(timestamp)

        headers = {
            "POLY_ADDRESS": self.address,
            "POLY_SIGNATURE": signature,
            "POLY_TIMESTAMP": str(timestamp),
            "POLY_NONCE": str(timestamp),
        }

        # Add API key if available
        if self.credentials.api_key:
            headers["POLY_API_KEY"] = self.credentials.api_key.get_secret_value()
        if self.credentials.api_passphrase:
            headers["POLY_PASSPHRASE"] = self.credentials.api_passphrase.get_secret_value()

        return headers

    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        reraise=True,
    )
    async def _request(
        self,
        method: str,
        endpoint: str,
        data: dict | None = None,
        authenticated: bool = True,
    ) -> dict:
        """
        Make HTTP request to Polymarket API with retry logic.
        """
        if not self._session:
            await self.connect()

        url = f"{self.conn_config.clob_api_url}{endpoint}"
        headers = {"Content-Type": "application/json"}

        if authenticated:
            headers.update(self._get_auth_headers())

        logger.debug("API request", method=method, url=url)

        try:
            # For SOCKS proxies, the connector handles the proxy - don't pass proxy param
            proxy_param = None if getattr(self, '_is_socks_proxy', False) else self._proxy_url
            async with self._session.request(
                method,
                url,
                json=data,
                headers=headers,
                proxy=proxy_param,
            ) as response:
                response_text = await response.text()

                if response.status >= 400:
                    logger.error(
                        "API error",
                        status=response.status,
                        response=response_text[:500],
                        url=url,
                    )
                    raise PolymarketAPIError(
                        f"API error: {response.status}",
                        status_code=response.status,
                        response={"raw": response_text},
                    )

                if response_text:
                    return json.loads(response_text)
                return {}

        except aiohttp.ClientError as e:
            logger.error("HTTP client error", error=str(e), url=url)
            raise
        except json.JSONDecodeError as e:
            logger.error("JSON decode error", error=str(e), response=response_text[:200])
            raise PolymarketAPIError(f"Invalid JSON response: {e}")

    # ==================== REST API Methods ====================

    async def get_markets(self, active_only: bool = True, limit: int = 100) -> list[dict]:
        """Get list of available markets."""
        endpoint = f"/markets?limit={limit}"
        if active_only:
            endpoint += "&active=true"

        response = await self._request("GET", endpoint, authenticated=False)

        # Handle different response formats
        if isinstance(response, list):
            return response
        elif isinstance(response, dict):
            # Polymarket returns paginated response with 'data' or 'markets' key
            if "data" in response:
                return response["data"]
            elif "markets" in response:
                return response["markets"]
            else:
                # Return as single-item list if it looks like a market
                logger.warning("Unexpected market response format", keys=list(response.keys()))
                return [response] if "condition_id" in response else []

        return []

    async def get_market(self, condition_id: str) -> dict:
        """Get market details by condition ID."""
        return await self._request("GET", f"/markets/{condition_id}", authenticated=False)

    async def get_orderbook(self, token_id: str) -> OrderBook:
        """Get order book snapshot for a token."""
        response = await self._request("GET", f"/book?token_id={token_id}", authenticated=False)

        bids = [
            OrderBookLevel(price=float(b["price"]), size=float(b["size"]))
            for b in response.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=float(a["price"]), size=float(a["size"]))
            for a in response.get("asks", [])
        ]

        # Sort bids descending, asks ascending
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        orderbook = OrderBook(
            market_id=response.get("market", ""),
            token_id=token_id,
            bids=bids,
            asks=asks,
        )

        self._orderbooks[token_id] = orderbook
        return orderbook

    async def get_balance(self) -> dict:
        """Get account balance."""
        try:
            return await self._request("GET", "/balance")
        except PolymarketAPIError as e:
            logger.warning("Balance API failed", error=str(e))
            # Return default for paper trading
            return {"available": 10000.0, "total": 10000.0}

    async def get_positions(self) -> list[dict]:
        """Get current positions."""
        try:
            return await self._request("GET", "/positions")
        except PolymarketAPIError:
            return []

    async def get_open_orders(self) -> list[dict]:
        """Get open orders."""
        try:
            return await self._request("GET", "/orders?status=open")
        except PolymarketAPIError:
            return []

    async def place_order(
        self,
        token_id: str,
        side: OrderSide,
        price: float,
        size: float,
        order_type: OrderType = OrderType.GTC,
    ) -> Order:
        """
        Place an order on Polymarket.

        For paper trading: Simulates the order locally.
        For live trading: Uses py-clob-client to place real orders.
        """
        logger.info(
            "Placing order",
            token_id=token_id[:30] + "...",
            side=side.value,
            price=price,
            size=size,
            paper_trading=self.config.is_paper_trading,
        )

        if self.config.is_paper_trading:
            # Simulate order in paper trading mode
            order = Order(
                id=f"paper_{int(time.time() * 1000)}",
                market_id="",
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                status=OrderStatus.OPEN,
            )
            self._open_orders[order.id] = order

            logger.info(
                "[PAPER] Order simulated",
                order_id=order.id,
                side=side.value,
                price=price,
                size=size,
            )

            return order

        # Live trading - use CLOB client
        if not self._clob_client:
            await self._initialize_clob_client()

        if not self._clob_client:
            raise PolymarketAPIError("CLOB client not initialized. Check your credentials.")

        try:
            # Convert side to CLOB format
            clob_side = BUY if side == OrderSide.BUY else SELL

            # Build order arguments
            # Price must be between 0.01 and 0.99 for Polymarket
            price = max(0.01, min(0.99, price))

            # Size is the number of contracts (shares)
            # Each share pays $1 if the outcome is correct
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=clob_side,
            )

            logger.info(
                "Creating CLOB order",
                token_id=token_id[:30] + "...",
                price=price,
                size=size,
                side=clob_side,
            )

            # Create and sign the order
            signed_order = self._clob_client.create_order(order_args)

            logger.info("Order signed, posting to CLOB...")

            # Post the order
            response = self._clob_client.post_order(signed_order, order_type=ClobOrderType.GTC)

            # Log the raw response for debugging
            logger.info(
                "Order response received",
                response_type=type(response).__name__,
                response=str(response)[:500],
            )

            # Parse response - can be dict, string, or other types
            order_id = f"clob_{int(time.time() * 1000)}"
            success = False

            if isinstance(response, dict):
                order_id = str(response.get("orderID", "") or response.get("order_id", "") or response.get("id", "") or order_id)
                success = response.get("success", True)  # Assume success if no explicit failure
                if "error" in response:
                    raise PolymarketAPIError(f"Order rejected: {response.get('error')}")
            elif isinstance(response, str):
                # Response might be a string (order ID or error message)
                if response and not response.startswith("{") and len(response) > 10:
                    order_id = response
                    success = True
                else:
                    logger.warning("Unexpected string response", response=response[:100])
            else:
                # Handle other response types (could be an object with attributes)
                logger.info("Non-dict response, checking attributes", type=type(response).__name__)
                if hasattr(response, 'orderID'):
                    order_id = str(response.orderID)
                    success = True
                elif hasattr(response, 'order_id'):
                    order_id = str(response.order_id)
                    success = True
                elif hasattr(response, '__dict__'):
                    # Try to access as dict
                    resp_dict = response.__dict__
                    order_id = str(resp_dict.get("orderID", "") or resp_dict.get("order_id", "") or order_id)
                    success = True

            logger.info(
                "Order placed successfully",
                order_id=order_id,
                success=success,
            )

            order = Order(
                id=order_id,
                market_id="",
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                status=OrderStatus.LIVE if success else OrderStatus.PENDING,
            )
            self._open_orders[order.id] = order

            return order

        except Exception as e:
            logger.error(
                "Failed to place order via CLOB client",
                error=str(e),
                token_id=token_id[:30] + "...",
            )
            import traceback
            traceback.print_exc()
            raise PolymarketAPIError(f"Order placement failed: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if self.config.is_paper_trading:
            if order_id in self._open_orders:
                self._open_orders[order_id].status = OrderStatus.CANCELLED
                del self._open_orders[order_id]
                logger.info("[PAPER] Order cancelled", order_id=order_id)
                return True
            return False

        # Live trading - use CLOB client
        if self._clob_client:
            try:
                response = self._clob_client.cancel(order_id)
                if order_id in self._open_orders:
                    self._open_orders[order_id].status = OrderStatus.CANCELLED
                    del self._open_orders[order_id]
                logger.info("Order cancelled", order_id=order_id)
                return True
            except Exception as e:
                logger.error("Failed to cancel order", order_id=order_id, error=str(e))
                return False

        # Fallback to REST API
        try:
            await self._request("DELETE", f"/order/{order_id}")
            if order_id in self._open_orders:
                self._open_orders[order_id].status = OrderStatus.CANCELLED
                del self._open_orders[order_id]
            logger.info("Order cancelled", order_id=order_id)
            return True
        except PolymarketAPIError as e:
            logger.error("Failed to cancel order", order_id=order_id, error=str(e))
            return False

    async def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        if self.config.is_paper_trading:
            count = len(self._open_orders)
            self._open_orders.clear()
            logger.info("[PAPER] All orders cancelled", count=count)
            return count

        # Live trading - use CLOB client
        if self._clob_client:
            try:
                response = self._clob_client.cancel_all()
                cancelled = len(response) if isinstance(response, list) else response.get("cancelled", 0)
                self._open_orders.clear()
                logger.info("All orders cancelled", count=cancelled)
                return cancelled
            except Exception as e:
                logger.error("Failed to cancel all orders", error=str(e))

        # Fallback to REST API
        try:
            response = await self._request("DELETE", "/orders")
            cancelled = response.get("cancelled", 0)
            self._open_orders.clear()
            logger.info("All orders cancelled", count=cancelled)
            return cancelled
        except PolymarketAPIError as e:
            logger.error("Failed to cancel all orders", error=str(e))
            return 0

    # ==================== WebSocket Methods ====================

    async def connect_websocket(self) -> None:
        """Connect to Polymarket WebSocket for real-time data."""
        if self._ws_connected:
            logger.warning("WebSocket already connected")
            return

        await self._establish_websocket()

    async def _establish_websocket(self) -> None:
        """Establish WebSocket connection with reconnection support."""
        if not self._session:
            await self.connect()

        try:
            logger.info("Connecting to WebSocket", url=self.conn_config.ws_url, proxy=self._proxy_url[:30] + "..." if self._proxy_url else "none")

            # For SOCKS proxies, the connector handles the proxy - don't pass proxy param
            proxy_param = None if getattr(self, '_is_socks_proxy', False) else self._proxy_url

            self._ws = await self._session.ws_connect(
                self.conn_config.ws_url,
                heartbeat=self.conn_config.ws_ping_interval_seconds,
                proxy=proxy_param,
            )
            self._ws_connected = True

            logger.info("WebSocket connected successfully")

            # Resubscribe to any previous subscriptions (send all at once)
            if self._ws_subscriptions:
                await self._send_subscribe(list(self._ws_subscriptions))

        except Exception as e:
            logger.error("WebSocket connection failed", error=str(e))
            self._ws_connected = False
            raise

    async def _close_websocket(self) -> None:
        """Close WebSocket connection."""
        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None

        if self._ws:
            await self._ws.close()
            self._ws = None
            self._ws_connected = False
            logger.info("WebSocket closed")

    async def _send_subscribe(self, token_ids: list[str] | str) -> None:
        """Send subscription message for token(s)."""
        if not self._ws or not self._ws_connected:
            raise RuntimeError("WebSocket not connected")

        # Ensure we have a list
        if isinstance(token_ids, str):
            token_ids = [token_ids]

        # Polymarket expects this format for market channel
        subscribe_msg = {
            "assets_ids": token_ids,
            "type": "market",
        }

        await self._ws.send_json(subscribe_msg)
        logger.info("Subscribed to orderbook", token_ids=[t[:20] + "..." for t in token_ids])

    async def subscribe_orderbook(self, token_id: str) -> None:
        """Subscribe to order book updates for a token."""
        self._ws_subscriptions.add(token_id)

        if self._ws_connected:
            await self._send_subscribe(token_id)

    async def unsubscribe_orderbook(self, token_id: str) -> None:
        """Unsubscribe from order book updates."""
        self._ws_subscriptions.discard(token_id)
        # Note: Polymarket may not support individual unsubscription
        # The subscription is managed by reconnecting with updated assets_ids
        logger.debug("Removed from subscription list", token_id=token_id[:30] + "...")

    def on_orderbook_update(
        self, callback: Callable[[OrderBook], Coroutine[Any, Any, None]]
    ) -> None:
        """Register callback for order book updates."""
        self._orderbook_callbacks.append(callback)

    async def _process_ws_message(self, msg: aiohttp.WSMessage) -> None:
        """Process incoming WebSocket message."""
        if msg.type == aiohttp.WSMsgType.TEXT:
            # Ignore empty messages (pings/keepalives)
            if not msg.data or not msg.data.strip():
                return

            try:
                data = json.loads(msg.data)
                await self._handle_ws_data(data)
            except json.JSONDecodeError as e:
                # Only log if it's not an empty/whitespace message
                if msg.data.strip():
                    logger.warning("Failed to parse WS message", error=str(e), data=msg.data[:100])

        elif msg.type == aiohttp.WSMsgType.ERROR:
            logger.error("WebSocket error", error=str(self._ws.exception()))

        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
            logger.warning("WebSocket closed")
            self._ws_connected = False

    async def _handle_ws_data(self, data: dict | list) -> None:
        """Handle parsed WebSocket data."""
        # Handle array of events
        if isinstance(data, list):
            for item in data:
                await self._handle_single_event(item)
        else:
            await self._handle_single_event(data)

    async def _handle_single_event(self, data: dict) -> None:
        """Handle a single WebSocket event."""
        # Polymarket uses event_type or type
        event_type = data.get("event_type") or data.get("type", "")

        if event_type == "book":
            # Full order book update
            token_id = data.get("asset_id", "")
            if not token_id:
                return

            # Parse bids and asks - handle both list and dict formats
            raw_bids = data.get("bids", [])
            raw_asks = data.get("asks", [])

            bids = []
            for b in raw_bids:
                if isinstance(b, (list, tuple)) and len(b) >= 2:
                    bids.append(OrderBookLevel(price=float(b[0]), size=float(b[1])))
                elif isinstance(b, dict):
                    bids.append(OrderBookLevel(
                        price=float(b.get("price", 0)),
                        size=float(b.get("size", 0))
                    ))

            asks = []
            for a in raw_asks:
                if isinstance(a, (list, tuple)) and len(a) >= 2:
                    asks.append(OrderBookLevel(price=float(a[0]), size=float(a[1])))
                elif isinstance(a, dict):
                    asks.append(OrderBookLevel(
                        price=float(a.get("price", 0)),
                        size=float(a.get("size", 0))
                    ))

            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)

            orderbook = OrderBook(
                market_id=data.get("market", ""),
                token_id=token_id,
                bids=bids,
                asks=asks,
            )

            self._orderbooks[token_id] = orderbook
            self._last_orderbook_update = datetime.now(timezone.utc)
            self._orderbook_update_count += 1

            # Log periodically
            if self._orderbook_update_count % 100 == 0:
                logger.debug(
                    "Orderbook updates received",
                    count=self._orderbook_update_count,
                    mid_price=orderbook.mid_price,
                )

            # Notify callbacks
            for callback in self._orderbook_callbacks:
                try:
                    await callback(orderbook)
                except Exception as e:
                    logger.error("Orderbook callback error", error=str(e))

        elif event_type == "price_change":
            # Price change event - lighter than full book
            token_id = data.get("asset_id", "")
            if not token_id:
                return

            best_bid = float(data.get("best_bid", 0)) if data.get("best_bid") else None
            best_ask = float(data.get("best_ask", 0)) if data.get("best_ask") else None

            if best_bid is not None and best_ask is not None:
                # Update or create minimal orderbook
                orderbook = OrderBook(
                    market_id=data.get("market", ""),
                    token_id=token_id,
                    bids=[OrderBookLevel(price=best_bid, size=0)] if best_bid else [],
                    asks=[OrderBookLevel(price=best_ask, size=0)] if best_ask else [],
                )

                self._orderbooks[token_id] = orderbook
                self._last_orderbook_update = datetime.now(timezone.utc)
                self._orderbook_update_count += 1

                for callback in self._orderbook_callbacks:
                    try:
                        await callback(orderbook)
                    except Exception as e:
                        logger.error("Orderbook callback error", error=str(e))

        elif event_type == "error":
            logger.error("WebSocket error message", error=data.get("message", "Unknown"))

        elif event_type:
            # Log unknown event types for debugging
            logger.debug("Unknown WS event type", event_type=event_type)

    async def listen_websocket(self) -> None:
        """
        Main WebSocket listening loop.

        Handles messages and automatic reconnection.
        """
        reconnect_delay = self.conn_config.ws_reconnect_delay_seconds

        while True:
            try:
                if not self._ws_connected:
                    await self._establish_websocket()

                async for msg in self._ws:
                    await self._process_ws_message(msg)

                # Connection closed normally
                logger.info("WebSocket connection ended")
                self._ws_connected = False

            except asyncio.CancelledError:
                logger.info("WebSocket listener cancelled")
                break

            except Exception as e:
                logger.error(
                    "WebSocket error, reconnecting",
                    error=str(e),
                    delay=reconnect_delay,
                )
                self._ws_connected = False

            # Wait before reconnecting
            await asyncio.sleep(reconnect_delay)

    def get_cached_orderbook(self, token_id: str) -> OrderBook | None:
        """Get cached order book for a token."""
        return self._orderbooks.get(token_id)

    def get_mid_price(self, token_id: str) -> float | None:
        """Get current mid price for a token."""
        orderbook = self._orderbooks.get(token_id)
        if orderbook:
            return orderbook.mid_price
        return None

    def get_connection_status(self) -> dict:
        """Get current connection status for debugging."""
        return {
            "http_session_active": self._session is not None,
            "ws_connected": self._ws_connected,
            "ws_subscriptions": list(self._ws_subscriptions),
            "orderbook_update_count": self._orderbook_update_count,
            "last_orderbook_update": self._last_orderbook_update.isoformat() if self._last_orderbook_update else None,
            "clob_client_initialized": self._clob_initialized,
            "paper_trading": self.config.is_paper_trading,
            "proxy": self._proxy_url[:30] + "..." if self._proxy_url else None,
        }
