"""
Historical Data Loader - Polymarket Spike Bot

Fetches and manages historical price data from Polymarket APIs.

Supports multiple data sources:
- Polymarket CLOB API (order books, trades)
- Gamma API (market data, historical prices)
- Local cache for faster reloads

Features:
- Automatic data caching to disk
- Rate limiting to respect API limits
- Data validation and cleaning
- Support for multiple timeframes
"""

import asyncio
import json
import os
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import aiohttp
import structlog

from .models import PriceData, OHLCV, MarketData


logger = structlog.get_logger(__name__)


# API Endpoints
CLOB_API_URL = "https://clob.polymarket.com"
GAMMA_API_URL = "https://gamma-api.polymarket.com"
STRAPI_API_URL = "https://strapi-matic.poly.market"


@dataclass
class APIRateLimiter:
    """Simple rate limiter for API calls."""
    requests_per_second: float = 5.0
    last_request: float = 0.0

    async def wait(self) -> None:
        """Wait if needed to respect rate limit."""
        now = asyncio.get_event_loop().time()
        min_interval = 1.0 / self.requests_per_second
        elapsed = now - self.last_request

        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)

        self.last_request = asyncio.get_event_loop().time()


class HistoricalDataLoader:
    """
    Loads historical price data from Polymarket.

    Features:
    - Fetches data from multiple API sources
    - Caches data locally for faster reloads
    - Validates and cleans data
    - Supports various timeframes
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        use_cache: bool = True,
        rate_limit: float = 5.0,
    ):
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.rate_limiter = APIRateLimiter(requests_per_second=rate_limit)

        # Ensure cache directory exists
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # HTTP session
        self._session: aiohttp.ClientSession | None = None

        logger.info(
            "HistoricalDataLoader initialized",
            cache_dir=str(self.cache_dir),
            use_cache=use_cache,
        )

    async def __aenter__(self) -> "HistoricalDataLoader":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> None:
        """Create HTTP session."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=60)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def disconnect(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a given key."""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.json"

    def _load_from_cache(self, key: str) -> dict | None:
        """Load data from cache if available."""
        if not self.use_cache:
            return None

        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r") as f:
                data = json.load(f)

            # Check cache expiry (24 hours for historical data)
            cached_at = datetime.fromisoformat(data.get("cached_at", "1970-01-01"))
            if datetime.now(timezone.utc) - cached_at > timedelta(hours=24):
                logger.debug("Cache expired", key=key)
                return None

            logger.debug("Loaded from cache", key=key)
            return data.get("data")

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to load cache", key=key, error=str(e))
            return None

    def _save_to_cache(self, key: str, data: Any) -> None:
        """Save data to cache."""
        if not self.use_cache:
            return

        cache_path = self._get_cache_path(key)
        try:
            cache_data = {
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "key": key,
                "data": data,
            }
            with open(cache_path, "w") as f:
                json.dump(cache_data, f)
            logger.debug("Saved to cache", key=key)
        except Exception as e:
            logger.warning("Failed to save cache", key=key, error=str(e))

    async def _request(
        self,
        url: str,
        params: dict | None = None,
        retries: int = 3,
    ) -> dict | list | None:
        """Make HTTP request with rate limiting and retries."""
        if not self._session:
            await self.connect()

        for attempt in range(retries):
            try:
                await self.rate_limiter.wait()

                async with self._session.get(url, params=params) as response:
                    if response.status == 429:
                        # Rate limited - wait and retry
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue

                    if response.status >= 400:
                        text = await response.text()
                        logger.warning(
                            "API error",
                            url=url,
                            status=response.status,
                            response=text[:200],
                        )
                        return None

                    return await response.json()

            except asyncio.TimeoutError:
                logger.warning(f"Request timeout, attempt {attempt + 1}/{retries}")
                await asyncio.sleep(2 ** attempt)
            except aiohttp.ClientError as e:
                logger.warning(f"HTTP error: {e}, attempt {attempt + 1}/{retries}")
                await asyncio.sleep(2 ** attempt)

        return None

    # ==================== Market Discovery ====================

    async def get_markets(
        self,
        active_only: bool = True,
        limit: int = 100,
    ) -> list[dict]:
        """Get list of available markets from Gamma API."""
        cache_key = f"markets_active={active_only}_limit={limit}"
        cached = self._load_from_cache(cache_key)
        if cached:
            return cached

        # Try Gamma API first (more complete data)
        url = f"{GAMMA_API_URL}/markets"
        params = {
            "limit": limit,
            "active": str(active_only).lower(),
            "closed": str(not active_only).lower(),
        }

        data = await self._request(url, params)

        if data:
            self._save_to_cache(cache_key, data)
            return data if isinstance(data, list) else []

        # Fallback to CLOB API
        url = f"{CLOB_API_URL}/markets"
        data = await self._request(url)

        if data:
            markets = data if isinstance(data, list) else data.get("data", [])
            self._save_to_cache(cache_key, markets)
            return markets

        return []

    async def get_market_info(self, condition_id: str) -> dict | None:
        """Get detailed market information."""
        cache_key = f"market_info_{condition_id}"
        cached = self._load_from_cache(cache_key)
        if cached:
            return cached

        # Try Gamma API
        url = f"{GAMMA_API_URL}/markets/{condition_id}"
        data = await self._request(url)

        if data:
            self._save_to_cache(cache_key, data)
            return data

        # Fallback to CLOB
        url = f"{CLOB_API_URL}/markets/{condition_id}"
        data = await self._request(url)

        if data:
            self._save_to_cache(cache_key, data)
            return data

        return None

    async def search_markets(self, query: str, limit: int = 50) -> list[dict]:
        """Search markets by name/description."""
        all_markets = []

        # Try Gamma API with different parameters
        url = f"{GAMMA_API_URL}/markets"

        # Fetch active markets
        params = {
            "limit": 200,
            "active": "true",
            "closed": "false",
        }

        data = await self._request(url, params)
        if data and isinstance(data, list):
            all_markets.extend(data)

        # Also try fetching with different sorting
        params2 = {
            "limit": 200,
            "order": "volume",
            "ascending": "false",
        }
        data2 = await self._request(url, params2)
        if data2 and isinstance(data2, list):
            # Add markets not already in list
            existing_ids = {m.get("condition_id") or m.get("conditionId") for m in all_markets}
            for m in data2:
                mid = m.get("condition_id") or m.get("conditionId")
                if mid and mid not in existing_ids:
                    all_markets.append(m)

        if not all_markets:
            return []

        # Filter by query
        query_lower = query.lower()
        matches = [
            m for m in all_markets
            if query_lower in m.get("question", "").lower()
            or query_lower in m.get("description", "").lower()
            or query_lower in m.get("title", "").lower()
        ]

        # Sort by volume (most active first)
        def get_volume(m):
            vol = m.get("volume", 0)
            try:
                return float(vol) if vol else 0
            except (ValueError, TypeError):
                return 0

        matches.sort(key=get_volume, reverse=True)

        return matches[:limit]

    # ==================== Historical Price Data ====================

    async def get_price_history(
        self,
        token_id: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1m",  # 1m, 5m, 1h, 1d
    ) -> list[PriceData]:
        """
        Get historical price data for a token.

        Uses multiple sources to construct price history:
        1. Gamma API timeseries endpoint
        2. CLOB trades endpoint
        3. Reconstructed from order book snapshots
        """
        cache_key = f"prices_{token_id}_{start_date.date()}_{end_date.date()}_{interval}"
        cached = self._load_from_cache(cache_key)
        if cached:
            return self._parse_price_data(cached)

        prices = []

        # Try Gamma API timeseries
        gamma_prices = await self._fetch_gamma_prices(token_id, start_date, end_date, interval)
        if gamma_prices:
            prices = gamma_prices

        # If no data from Gamma, try CLOB trades
        if not prices:
            clob_prices = await self._fetch_clob_trades(token_id, start_date, end_date)
            if clob_prices:
                prices = clob_prices

        # Cache raw data
        if prices:
            raw_data = [
                {
                    "timestamp": p.timestamp.isoformat(),
                    "price": p.price,
                    "volume": p.volume,
                    "bid": p.bid,
                    "ask": p.ask,
                }
                for p in prices
            ]
            self._save_to_cache(cache_key, raw_data)

        logger.info(
            "Fetched price history",
            token_id=token_id,
            data_points=len(prices),
            start=start_date.isoformat(),
            end=end_date.isoformat(),
        )

        return prices

    def _parse_price_data(self, raw_data: list[dict]) -> list[PriceData]:
        """Parse raw price data into PriceData objects."""
        prices = []
        for item in raw_data:
            try:
                timestamp = item.get("timestamp")
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                elif isinstance(timestamp, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)

                prices.append(PriceData(
                    timestamp=timestamp,
                    price=float(item.get("price", 0)),
                    volume=float(item.get("volume", 0)),
                    bid=float(item["bid"]) if item.get("bid") else None,
                    ask=float(item["ask"]) if item.get("ask") else None,
                ))
            except (ValueError, KeyError, TypeError) as e:
                logger.debug(f"Failed to parse price data: {e}")
                continue

        # Sort by timestamp
        prices.sort(key=lambda x: x.timestamp)
        return prices

    async def _fetch_gamma_prices(
        self,
        token_id: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ) -> list[PriceData]:
        """Fetch price history from Gamma API."""
        prices = []

        # Method 1: Try the prices/history endpoint with clob_token_id
        url = f"{GAMMA_API_URL}/prices/history"
        params = {
            "clob_token_id": token_id,
            "interval": "max",  # Get all available data
        }
        data = await self._request(url, params)

        if data and isinstance(data, dict) and "history" in data:
            for point in data["history"]:
                try:
                    ts = point.get("t")
                    if isinstance(ts, (int, float)):
                        timestamp = datetime.fromtimestamp(ts / 1000 if ts > 1e10 else ts, tz=timezone.utc)
                    else:
                        continue

                    price = point.get("p")
                    if price is None:
                        continue

                    # Filter by date range
                    if start_date <= timestamp <= end_date:
                        prices.append(PriceData(
                            timestamp=timestamp,
                            price=float(price),
                            volume=0,
                        ))
                except (ValueError, KeyError, TypeError):
                    continue

        # Method 2: Try market-specific endpoint
        if not prices:
            url = f"{GAMMA_API_URL}/markets/{token_id}/prices"
            data = await self._request(url)

            if data and isinstance(data, list):
                for point in data:
                    try:
                        ts = point.get("timestamp") or point.get("t")
                        if isinstance(ts, str):
                            timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        elif isinstance(ts, (int, float)):
                            timestamp = datetime.fromtimestamp(ts / 1000 if ts > 1e10 else ts, tz=timezone.utc)
                        else:
                            continue

                        price = point.get("price") or point.get("p")
                        if price is None:
                            continue

                        if start_date <= timestamp <= end_date:
                            prices.append(PriceData(
                                timestamp=timestamp,
                                price=float(price),
                                volume=float(point.get("volume", 0)),
                            ))
                    except (ValueError, KeyError, TypeError):
                        continue

        # Method 3: Try CLOB book snapshots to get current price at least
        if not prices:
            url = f"{CLOB_API_URL}/book"
            params = {"token_id": token_id}
            data = await self._request(url, params)

            if data:
                try:
                    bids = data.get("bids", [])
                    asks = data.get("asks", [])

                    best_bid = float(bids[0]["price"]) if bids else None
                    best_ask = float(asks[0]["price"]) if asks else None

                    if best_bid and best_ask:
                        mid_price = (best_bid + best_ask) / 2
                        prices.append(PriceData(
                            timestamp=datetime.now(timezone.utc),
                            price=mid_price,
                            volume=0,
                            bid=best_bid,
                            ask=best_ask,
                        ))
                        logger.info(f"Got current price from order book: {mid_price}")
                except (ValueError, KeyError, TypeError, IndexError):
                    pass

        return prices

    async def _fetch_clob_trades(
        self,
        token_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[PriceData]:
        """Fetch trade history from CLOB API."""
        all_trades = []

        # Try public trades endpoint
        url = f"{CLOB_API_URL}/trades"
        params = {"asset_id": token_id, "limit": 500}

        data = await self._request(url, params)
        if data:
            trades = data if isinstance(data, list) else data.get("data", [])
            all_trades.extend(trades)
            logger.info(f"Fetched {len(trades)} trades from public API")

        # Convert trades to price data
        prices = []
        for trade in all_trades:
            try:
                ts = trade.get("created_at") or trade.get("timestamp") or trade.get("matchTime") or trade.get("match_time")
                if isinstance(ts, str):
                    timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                elif isinstance(ts, (int, float)):
                    timestamp = datetime.fromtimestamp(ts / 1000 if ts > 1e10 else ts, tz=timezone.utc)
                else:
                    continue

                # Filter by date range
                if timestamp < start_date or timestamp > end_date:
                    continue

                prices.append(PriceData(
                    timestamp=timestamp,
                    price=float(trade.get("price", 0)),
                    volume=float(trade.get("size", 0) or trade.get("amount", 0)),
                ))
            except (ValueError, KeyError, TypeError):
                continue

        prices.sort(key=lambda x: x.timestamp)

        if prices:
            logger.info(f"Got {len(prices)} price points from trades")

        return prices

    # ==================== Simulated Data Generation ====================

    def generate_simulated_data(
        self,
        start_date: datetime,
        end_date: datetime,
        initial_price: float = 0.5,
        volatility: float = 0.02,
        spike_probability: float = 0.01,
        spike_magnitude: float = 0.15,
        mean_reversion_speed: float = 0.1,
        interval_minutes: int = 1,
    ) -> list[PriceData]:
        """
        Generate simulated price data for backtesting.

        This creates realistic price movements with:
        - Random walk with mean reversion
        - Occasional price spikes (for spike strategy testing)
        - Configurable volatility and spike characteristics

        Args:
            start_date: Start of simulation period
            end_date: End of simulation period
            initial_price: Starting price (0-1)
            volatility: Standard deviation of price changes
            spike_probability: Probability of spike per interval
            spike_magnitude: Size of spike as fraction of price
            mean_reversion_speed: How fast price reverts to mean
            interval_minutes: Time between data points

        Returns:
            List of PriceData objects with simulated prices
        """
        import random

        prices = []
        current_price = initial_price
        mean_price = initial_price

        current_time = start_date

        while current_time <= end_date:
            # Normal random walk
            noise = random.gauss(0, volatility)

            # Mean reversion
            reversion = mean_reversion_speed * (mean_price - current_price)

            # Check for spike
            spike = 0
            if random.random() < spike_probability:
                # Spike direction (up or down)
                direction = 1 if random.random() > 0.5 else -1
                spike = direction * spike_magnitude * current_price
                logger.debug(f"Simulated spike at {current_time}: {spike:+.4f}")

            # Update price
            current_price = current_price + noise + reversion + spike

            # Clamp to valid range (0.01 to 0.99)
            current_price = max(0.01, min(0.99, current_price))

            # Add some volume noise
            volume = abs(random.gauss(1000, 500))

            prices.append(PriceData(
                timestamp=current_time,
                price=current_price,
                volume=volume,
            ))

            current_time += timedelta(minutes=interval_minutes)

        logger.info(
            "Generated simulated data",
            data_points=len(prices),
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            spikes=int(len(prices) * spike_probability),
        )

        return prices

    async def get_price_history_with_simulation(
        self,
        token_id: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1m",
        use_simulation_if_no_data: bool = True,
        initial_price: float | None = None,
    ) -> tuple[list[PriceData], bool]:
        """
        Get price history, falling back to simulation if no real data available.

        Returns:
            Tuple of (prices, is_simulated)
        """
        # Try to get real data first
        prices = await self.get_price_history(token_id, start_date, end_date, interval)

        if len(prices) >= 100:
            return prices, False

        if not use_simulation_if_no_data:
            return prices, False

        logger.warning(
            f"Insufficient real data ({len(prices)} points), using simulation"
        )

        # If we have current price from order book, use it as initial
        if prices and not initial_price:
            initial_price = prices[-1].price
        elif not initial_price:
            initial_price = 0.5

        # Generate simulated data
        simulated = self.generate_simulated_data(
            start_date=start_date,
            end_date=end_date,
            initial_price=initial_price,
            interval_minutes=1 if interval == "1m" else 5,
        )

        return simulated, True

    async def get_ohlcv(
        self,
        token_id: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1h",
    ) -> list[OHLCV]:
        """Get OHLCV candlestick data for a token."""
        cache_key = f"ohlcv_{token_id}_{start_date.date()}_{end_date.date()}_{interval}"
        cached = self._load_from_cache(cache_key)
        if cached:
            return self._parse_ohlcv(cached)

        # Try to get from Gamma API
        url = f"{GAMMA_API_URL}/prices"

        interval_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "1h": "1hour",
            "4h": "4hour",
            "1d": "1day",
        }
        gamma_interval = interval_map.get(interval, "1hour")

        params = {
            "market": token_id,
            "interval": gamma_interval,
            "startTs": int(start_date.timestamp() * 1000),
            "endTs": int(end_date.timestamp() * 1000),
        }

        data = await self._request(url, params)

        ohlcv_list = []
        if data:
            history = data if isinstance(data, list) else data.get("history", [])

            for candle in history:
                try:
                    ts = candle.get("t") or candle.get("timestamp")
                    if isinstance(ts, (int, float)):
                        timestamp = datetime.fromtimestamp(ts / 1000 if ts > 1e10 else ts, tz=timezone.utc)
                    else:
                        timestamp = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))

                    ohlcv_list.append(OHLCV(
                        timestamp=timestamp,
                        open=float(candle.get("o", 0) or candle.get("open", 0)),
                        high=float(candle.get("h", 0) or candle.get("high", 0)),
                        low=float(candle.get("l", 0) or candle.get("low", 0)),
                        close=float(candle.get("c", 0) or candle.get("close", 0)),
                        volume=float(candle.get("v", 0) or candle.get("volume", 0)),
                    ))
                except (ValueError, KeyError, TypeError):
                    continue

        # If no OHLCV data available, construct from tick data
        if not ohlcv_list:
            prices = await self.get_price_history(token_id, start_date, end_date, "1m")
            ohlcv_list = self._construct_ohlcv(prices, interval)

        # Cache
        if ohlcv_list:
            raw_data = [
                {
                    "timestamp": o.timestamp.isoformat(),
                    "open": o.open,
                    "high": o.high,
                    "low": o.low,
                    "close": o.close,
                    "volume": o.volume,
                }
                for o in ohlcv_list
            ]
            self._save_to_cache(cache_key, raw_data)

        logger.info(
            "Fetched OHLCV data",
            token_id=token_id,
            candles=len(ohlcv_list),
            interval=interval,
        )

        return ohlcv_list

    def _parse_ohlcv(self, raw_data: list[dict]) -> list[OHLCV]:
        """Parse raw OHLCV data."""
        ohlcv_list = []
        for item in raw_data:
            try:
                timestamp = item.get("timestamp")
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                elif isinstance(timestamp, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)

                ohlcv_list.append(OHLCV(
                    timestamp=timestamp,
                    open=float(item.get("open", 0)),
                    high=float(item.get("high", 0)),
                    low=float(item.get("low", 0)),
                    close=float(item.get("close", 0)),
                    volume=float(item.get("volume", 0)),
                ))
            except (ValueError, KeyError, TypeError):
                continue

        ohlcv_list.sort(key=lambda x: x.timestamp)
        return ohlcv_list

    def _construct_ohlcv(
        self,
        prices: list[PriceData],
        interval: str,
    ) -> list[OHLCV]:
        """Construct OHLCV candles from tick data."""
        if not prices:
            return []

        # Parse interval
        interval_seconds = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
        }.get(interval, 3600)

        ohlcv_list = []
        current_candle = None
        current_bucket = None

        for price in prices:
            # Calculate bucket timestamp
            bucket = price.timestamp.replace(
                second=0,
                microsecond=0,
            )
            bucket_ts = int(bucket.timestamp())
            bucket_ts = bucket_ts - (bucket_ts % interval_seconds)
            bucket = datetime.fromtimestamp(bucket_ts, tz=timezone.utc)

            if current_bucket is None or bucket != current_bucket:
                # Save previous candle
                if current_candle:
                    ohlcv_list.append(current_candle)

                # Start new candle
                current_bucket = bucket
                current_candle = OHLCV(
                    timestamp=bucket,
                    open=price.price,
                    high=price.price,
                    low=price.price,
                    close=price.price,
                    volume=price.volume,
                )
            else:
                # Update current candle
                current_candle.high = max(current_candle.high, price.price)
                current_candle.low = min(current_candle.low, price.price)
                current_candle.close = price.price
                current_candle.volume += price.volume

        # Add last candle
        if current_candle:
            ohlcv_list.append(current_candle)

        return ohlcv_list

    # ==================== Complete Market Data ====================

    async def load_market_data(
        self,
        token_id: str,
        start_date: datetime,
        end_date: datetime,
        resolution: str = "1m",
    ) -> MarketData:
        """
        Load complete market data for backtesting.

        Returns a MarketData object with all available data.
        """
        # Get market info
        market_info = await self.get_market_info(token_id)

        market_name = ""
        condition_id = token_id

        if market_info:
            market_name = market_info.get("question", "") or market_info.get("title", "")
            condition_id = market_info.get("condition_id", token_id)

        # Get price history
        prices = await self.get_price_history(token_id, start_date, end_date, resolution)

        # Get OHLCV data for additional analysis
        ohlcv = await self.get_ohlcv(token_id, start_date, end_date, "1h")

        # Calculate statistics
        total_volume = sum(p.volume for p in prices)
        days = max((end_date - start_date).days, 1)
        avg_daily_volume = total_volume / days

        # Calculate average spread if available
        spreads = [p.spread for p in prices if p.spread is not None]
        avg_spread = sum(spreads) / len(spreads) if spreads else 0.0

        # Calculate liquidity score (0-100)
        liquidity_score = min(100, (avg_daily_volume / 1000) * 10)

        market_data = MarketData(
            token_id=token_id,
            market_name=market_name,
            condition_id=condition_id,
            start_date=start_date,
            end_date=end_date,
            prices=prices,
            ohlcv=ohlcv,
            total_volume=total_volume,
            avg_daily_volume=avg_daily_volume,
            avg_spread=avg_spread,
            liquidity_score=liquidity_score,
        )

        logger.info(
            "Loaded market data",
            token_id=token_id,
            market_name=market_name[:50] if market_name else "Unknown",
            price_points=len(prices),
            ohlcv_candles=len(ohlcv),
            total_volume=f"${total_volume:,.2f}",
        )

        return market_data

    async def load_multiple_markets(
        self,
        token_ids: list[str],
        start_date: datetime,
        end_date: datetime,
        resolution: str = "1m",
    ) -> dict[str, MarketData]:
        """Load data for multiple markets concurrently."""
        tasks = [
            self.load_market_data(token_id, start_date, end_date, resolution)
            for token_id in token_ids
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        market_data = {}
        for token_id, result in zip(token_ids, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to load data for {token_id}: {result}")
            else:
                market_data[token_id] = result

        return market_data

    # ==================== Data Validation ====================

    def validate_data(self, prices: list[PriceData]) -> tuple[bool, list[str]]:
        """
        Validate price data for backtesting.

        Returns (is_valid, list of issues).
        """
        issues = []

        if not prices:
            return False, ["No price data available"]

        # Check for minimum data points
        if len(prices) < 100:
            issues.append(f"Insufficient data points: {len(prices)} (minimum 100)")

        # Check for data gaps
        if len(prices) >= 2:
            timestamps = [p.timestamp for p in prices]
            gaps = []
            for i in range(1, len(timestamps)):
                diff = (timestamps[i] - timestamps[i-1]).total_seconds()
                if diff > 3600:  # Gap larger than 1 hour
                    gaps.append((timestamps[i-1], timestamps[i], diff / 3600))

            if gaps:
                issues.append(f"Found {len(gaps)} data gaps > 1 hour")

        # Check for invalid prices
        invalid_prices = [p for p in prices if p.price <= 0 or p.price > 1]
        if invalid_prices:
            issues.append(f"Found {len(invalid_prices)} invalid prices (should be 0-1)")

        # Check for duplicate timestamps
        timestamps = [p.timestamp for p in prices]
        if len(timestamps) != len(set(timestamps)):
            issues.append("Found duplicate timestamps")

        is_valid = len(issues) == 0 or all("Gap" in i for i in issues)  # Allow gaps
        return is_valid, issues

    def clean_data(self, prices: list[PriceData]) -> list[PriceData]:
        """
        Clean price data by removing invalid entries.

        - Removes invalid prices (<=0 or >1)
        - Removes duplicate timestamps
        - Sorts by timestamp
        """
        # Filter invalid prices
        valid = [
            p for p in prices
            if 0 < p.price <= 1
        ]

        # Remove duplicates (keep first occurrence)
        seen = set()
        unique = []
        for p in valid:
            key = p.timestamp.isoformat()
            if key not in seen:
                seen.add(key)
                unique.append(p)

        # Sort by timestamp
        unique.sort(key=lambda x: x.timestamp)

        removed = len(prices) - len(unique)
        if removed > 0:
            logger.info(f"Cleaned data: removed {removed} invalid/duplicate entries")

        return unique

    def interpolate_gaps(
        self,
        prices: list[PriceData],
        max_gap_seconds: int = 300,
    ) -> list[PriceData]:
        """
        Fill small gaps in data with linear interpolation.

        Only fills gaps smaller than max_gap_seconds.
        """
        if len(prices) < 2:
            return prices

        filled = [prices[0]]

        for i in range(1, len(prices)):
            prev = prices[i - 1]
            curr = prices[i]

            gap_seconds = (curr.timestamp - prev.timestamp).total_seconds()

            if gap_seconds > 60 and gap_seconds <= max_gap_seconds:
                # Interpolate
                num_points = int(gap_seconds / 60)  # One point per minute
                price_step = (curr.price - prev.price) / (num_points + 1)

                for j in range(1, num_points + 1):
                    interp_time = prev.timestamp + timedelta(seconds=60 * j)
                    interp_price = prev.price + price_step * j

                    filled.append(PriceData(
                        timestamp=interp_time,
                        price=interp_price,
                        volume=0,  # No volume for interpolated points
                    ))

            filled.append(curr)

        logger.info(f"Interpolated {len(filled) - len(prices)} missing data points")
        return filled

    # ==================== Cache Management ====================

    def clear_cache(self) -> int:
        """Clear all cached data. Returns number of files removed."""
        if not self.cache_dir.exists():
            return 0

        count = 0
        for f in self.cache_dir.glob("*.json"):
            try:
                f.unlink()
                count += 1
            except Exception:
                pass

        logger.info(f"Cleared {count} cached files")
        return count

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        if not self.cache_dir.exists():
            return {"files": 0, "size_mb": 0}

        files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in files)

        return {
            "files": len(files),
            "size_mb": total_size / (1024 * 1024),
            "path": str(self.cache_dir),
        }
