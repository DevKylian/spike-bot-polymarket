"""
Market Analyzer Module - Polymarket Spike Bot

Provides URL parsing and comprehensive pre-trade analysis:
- Extract market/event IDs from Polymarket URLs
- Fetch detailed market information via Gamma API
- Analyze market conditions before trading
- Calculate trading viability scores
"""

import re
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from urllib.parse import urlparse

import aiohttp
import structlog

logger = structlog.get_logger(__name__)

# Gamma API for market data (no auth required)
GAMMA_API_URL = "https://gamma-api.polymarket.com"


class MarketStatus(str, Enum):
    """Market trading status."""
    ACTIVE = "active"
    CLOSED = "closed"
    RESOLVED = "resolved"
    UNKNOWN = "unknown"


class TradingViability(str, Enum):
    """Trading viability assessment."""
    EXCELLENT = "excellent"  # Score >= 80
    GOOD = "good"           # Score >= 60
    FAIR = "fair"           # Score >= 40
    POOR = "poor"           # Score >= 20
    NOT_RECOMMENDED = "not_recommended"  # Score < 20


@dataclass
class MarketInfo:
    """Detailed market information."""
    # Identifiers
    event_slug: str = ""
    market_slug: str = ""
    event_id: str = ""
    condition_id: str = ""

    # Market data
    question: str = ""
    description: str = ""
    category: str = ""

    # Tokens for trading (YES/NO outcomes)
    tokens: list[dict] = field(default_factory=list)

    # Market metrics
    volume: float = 0.0
    liquidity: float = 0.0
    volume_24h: float = 0.0

    # Prices
    yes_price: float = 0.0
    no_price: float = 0.0

    # Status
    status: MarketStatus = MarketStatus.UNKNOWN
    end_date: datetime | None = None

    # Order book metrics
    spread_percent: float = 0.0
    bid_depth: float = 0.0
    ask_depth: float = 0.0

    # Raw data
    raw_event: dict = field(default_factory=dict)
    raw_market: dict = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Pre-trade analysis result."""
    market_info: MarketInfo

    # Overall score (0-100)
    viability_score: float = 0.0
    viability: TradingViability = TradingViability.NOT_RECOMMENDED

    # Component scores
    liquidity_score: float = 0.0
    volume_score: float = 0.0
    spread_score: float = 0.0
    volatility_score: float = 0.0
    time_score: float = 0.0

    # Analysis details
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    # Suggested parameters
    suggested_order_size: float = 0.0
    suggested_spike_threshold: float = 3.0
    max_position: float = 0.0

    # Timestamp
    analyzed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MarketAnalyzer:
    """
    Analyzes Polymarket markets for trading viability.

    Provides:
    - URL parsing to extract market identifiers
    - Market data fetching via Gamma API
    - Pre-trade analysis with viability scoring
    - Trading parameter recommendations
    """

    def __init__(self):
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "MarketAnalyzer":
        await self._ensure_session()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def _ensure_session(self) -> None:
        """Ensure HTTP session exists."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def parse_polymarket_url(self, url: str) -> tuple[str, str]:
        """
        Parse a Polymarket URL to extract event and market slugs.

        URL formats:
        - https://polymarket.com/event/{event_slug}
        - https://polymarket.com/event/{event_slug}/{market_slug}
        - https://polymarket.com/market/{market_slug}

        Returns:
            Tuple of (event_slug, market_slug)
        """
        url = url.strip()

        # Handle URLs with query params
        parsed = urlparse(url)
        path = parsed.path.strip("/")

        # Split path segments
        segments = [s for s in path.split("/") if s]

        event_slug = ""
        market_slug = ""

        if len(segments) >= 2:
            if segments[0] == "event":
                event_slug = segments[1]
                if len(segments) >= 3:
                    market_slug = segments[2]
            elif segments[0] == "market":
                market_slug = segments[1]
        elif len(segments) == 1:
            # Could be direct slug
            market_slug = segments[0]

        logger.debug(
            "Parsed Polymarket URL",
            url=url,
            event_slug=event_slug,
            market_slug=market_slug,
        )

        return event_slug, market_slug

    async def fetch_event_by_slug(self, slug: str) -> dict | None:
        """Fetch event data from Gamma API by slug."""
        await self._ensure_session()

        try:
            url = f"{GAMMA_API_URL}/events/slug/{slug}"
            async with self._session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 404:
                    logger.warning("Event not found", slug=slug)
                    return None
                else:
                    logger.error("Failed to fetch event", slug=slug, status=response.status)
                    return None
        except Exception as e:
            logger.error("Error fetching event", slug=slug, error=str(e))
            return None

    async def fetch_market_by_slug(self, slug: str) -> dict | None:
        """Fetch market data from Gamma API by slug."""
        await self._ensure_session()

        try:
            url = f"{GAMMA_API_URL}/markets/slug/{slug}"
            async with self._session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 404:
                    logger.warning("Market not found", slug=slug)
                    return None
                else:
                    logger.error("Failed to fetch market", slug=slug, status=response.status)
                    return None
        except Exception as e:
            logger.error("Error fetching market", slug=slug, error=str(e))
            return None

    async def fetch_market_by_condition_id(self, condition_id: str) -> dict | None:
        """Fetch market data by condition ID."""
        await self._ensure_session()

        try:
            url = f"{GAMMA_API_URL}/markets/{condition_id}"
            async with self._session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                return None
        except Exception as e:
            logger.error("Error fetching market by condition_id", error=str(e))
            return None

    async def get_market_from_url(self, url: str) -> MarketInfo | None:
        """
        Get complete market information from a Polymarket URL.

        This is the main method to use for URL-based market lookup.
        """
        event_slug, market_slug = self.parse_polymarket_url(url)

        if not event_slug and not market_slug:
            logger.error("Could not parse URL", url=url)
            return None

        market_info = MarketInfo(
            event_slug=event_slug,
            market_slug=market_slug,
        )

        # Fetch event data
        event_data = None
        if event_slug:
            event_data = await self.fetch_event_by_slug(event_slug)
            if event_data:
                market_info.raw_event = event_data
                market_info.event_id = str(event_data.get("id", ""))
                market_info.category = event_data.get("category", "")

        # Fetch market data
        market_data = None
        if market_slug:
            market_data = await self.fetch_market_by_slug(market_slug)
        elif event_data and "markets" in event_data:
            # Get first market from event
            markets = event_data.get("markets", [])
            if markets:
                market_data = markets[0]

        if market_data:
            market_info.raw_market = market_data
            market_info.condition_id = market_data.get("conditionId", "") or market_data.get("condition_id", "")
            market_info.question = market_data.get("question", "")
            market_info.description = market_data.get("description", "")

            # Get tokens
            tokens = market_data.get("tokens", [])
            if tokens:
                market_info.tokens = tokens
                # Extract YES/NO prices
                for token in tokens:
                    outcome = token.get("outcome", "").upper()
                    price = float(token.get("price", 0) or 0)
                    if outcome == "YES":
                        market_info.yes_price = price
                    elif outcome == "NO":
                        market_info.no_price = price

            # Market metrics
            market_info.volume = float(market_data.get("volume", 0) or 0)
            market_info.liquidity = float(market_data.get("liquidity", 0) or 0)
            market_info.volume_24h = float(market_data.get("volume24hr", 0) or market_data.get("volume_24h", 0) or 0)

            # Status
            if market_data.get("closed"):
                market_info.status = MarketStatus.CLOSED
            elif market_data.get("resolved"):
                market_info.status = MarketStatus.RESOLVED
            elif market_data.get("active", True):
                market_info.status = MarketStatus.ACTIVE

            # End date
            end_date_str = market_data.get("endDate") or market_data.get("end_date")
            if end_date_str:
                try:
                    market_info.end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                except Exception:
                    pass

        # If we don't have essential data, return None
        if not market_info.condition_id and not market_info.tokens:
            logger.error("Could not find market tokens", url=url)
            return None

        return market_info

    async def analyze_market(self, market_info: MarketInfo) -> AnalysisResult:
        """
        Perform comprehensive pre-trade analysis on a market.

        Returns detailed analysis with viability score and recommendations.
        """
        result = AnalysisResult(market_info=market_info)

        # 1. Check market status
        if market_info.status != MarketStatus.ACTIVE:
            result.warnings.append(f"Market is {market_info.status.value}, not active")
            result.viability = TradingViability.NOT_RECOMMENDED
            return result

        # 2. Check time to expiry
        if market_info.end_date:
            time_to_end = (market_info.end_date - datetime.now(timezone.utc)).total_seconds()
            hours_to_end = time_to_end / 3600

            if hours_to_end < 1:
                result.warnings.append("Market closes in less than 1 hour")
                result.time_score = 0
            elif hours_to_end < 24:
                result.warnings.append(f"Market closes in {hours_to_end:.1f} hours")
                result.time_score = min(50, hours_to_end * 2)
            elif hours_to_end < 168:  # 1 week
                result.time_score = 70
            else:
                result.time_score = 100
        else:
            result.time_score = 80  # Unknown expiry, assume reasonable

        # 3. Analyze liquidity
        if market_info.liquidity >= 100000:
            result.liquidity_score = 100
            result.recommendations.append("Excellent liquidity - suitable for larger positions")
        elif market_info.liquidity >= 50000:
            result.liquidity_score = 80
        elif market_info.liquidity >= 10000:
            result.liquidity_score = 60
            result.recommendations.append("Moderate liquidity - use smaller position sizes")
        elif market_info.liquidity >= 1000:
            result.liquidity_score = 40
            result.warnings.append("Low liquidity - high slippage risk")
        else:
            result.liquidity_score = 20
            result.warnings.append("Very low liquidity - trading not recommended")

        # 4. Analyze volume
        if market_info.volume_24h >= 10000:
            result.volume_score = 100
            result.recommendations.append("High 24h volume indicates active market")
        elif market_info.volume_24h >= 5000:
            result.volume_score = 80
        elif market_info.volume_24h >= 1000:
            result.volume_score = 60
        elif market_info.volume_24h >= 100:
            result.volume_score = 40
            result.warnings.append("Low 24h volume - fewer trading opportunities")
        else:
            result.volume_score = 20
            result.warnings.append("Very low volume - market may be inactive")

        # 5. Analyze spread (if we have price data)
        if market_info.yes_price > 0:
            # For prediction markets, spread is implicit in YES/NO prices
            implied_spread = abs(1 - market_info.yes_price - market_info.no_price)
            market_info.spread_percent = implied_spread * 100

            if implied_spread < 0.02:
                result.spread_score = 100
            elif implied_spread < 0.05:
                result.spread_score = 80
            elif implied_spread < 0.10:
                result.spread_score = 60
                result.recommendations.append("Moderate spread - factor into profit calculations")
            else:
                result.spread_score = 40
                result.warnings.append(f"Wide spread ({implied_spread*100:.1f}%) - impacts profitability")
        else:
            result.spread_score = 50  # Unknown

        # 6. Estimate volatility from price position
        # Prices near 0.5 tend to be more volatile
        if market_info.yes_price > 0:
            price_from_center = abs(market_info.yes_price - 0.5)
            if price_from_center < 0.1:  # Price between 0.4-0.6
                result.volatility_score = 100
                result.recommendations.append("Price near 50% - high volatility expected (good for spike detection)")
            elif price_from_center < 0.2:  # 0.3-0.7
                result.volatility_score = 80
            elif price_from_center < 0.3:  # 0.2-0.8
                result.volatility_score = 60
                result.recommendations.append("Price trending - moderate volatility expected")
            else:  # Near 0 or 1
                result.volatility_score = 30
                result.warnings.append("Price near extreme - lower spike opportunity")
        else:
            result.volatility_score = 50

        # Calculate overall viability score (weighted average)
        result.viability_score = (
            result.liquidity_score * 0.30 +
            result.volume_score * 0.25 +
            result.spread_score * 0.15 +
            result.volatility_score * 0.20 +
            result.time_score * 0.10
        )

        # Determine viability category
        if result.viability_score >= 80:
            result.viability = TradingViability.EXCELLENT
        elif result.viability_score >= 60:
            result.viability = TradingViability.GOOD
        elif result.viability_score >= 40:
            result.viability = TradingViability.FAIR
        elif result.viability_score >= 20:
            result.viability = TradingViability.POOR
        else:
            result.viability = TradingViability.NOT_RECOMMENDED

        # Calculate suggested parameters
        # Order size based on liquidity (max 1% of liquidity)
        result.suggested_order_size = min(100, market_info.liquidity * 0.01)
        result.max_position = min(500, market_info.liquidity * 0.05)

        # Spike threshold based on volatility
        if result.volatility_score >= 80:
            result.suggested_spike_threshold = 2.5  # More sensitive for volatile markets
        elif result.volatility_score >= 50:
            result.suggested_spike_threshold = 3.0
        else:
            result.suggested_spike_threshold = 4.0  # Less sensitive for stable markets

        return result

    async def analyze_from_url(self, url: str) -> AnalysisResult | None:
        """
        Complete analysis pipeline from URL.

        Fetches market info and performs full analysis.
        """
        market_info = await self.get_market_from_url(url)
        if not market_info:
            return None

        return await self.analyze_market(market_info)


def format_analysis_report(result: AnalysisResult) -> str:
    """Format analysis result as human-readable report."""
    info = result.market_info

    lines = [
        "=" * 60,
        "MARKET ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Question: {info.question[:80]}..." if len(info.question) > 80 else f"Question: {info.question}",
        f"Category: {info.category}",
        f"Status: {info.status.value}",
        "",
        "--- PRICING ---",
        f"YES Price: ${info.yes_price:.3f}" if info.yes_price else "YES Price: N/A",
        f"NO Price: ${info.no_price:.3f}" if info.no_price else "NO Price: N/A",
        f"Spread: {info.spread_percent:.2f}%",
        "",
        "--- MARKET METRICS ---",
        f"Total Volume: ${info.volume:,.0f}",
        f"24h Volume: ${info.volume_24h:,.0f}",
        f"Liquidity: ${info.liquidity:,.0f}",
        "",
        "--- VIABILITY SCORES ---",
        f"Overall Score: {result.viability_score:.0f}/100 ({result.viability.value.upper()})",
        f"  Liquidity: {result.liquidity_score:.0f}",
        f"  Volume: {result.volume_score:.0f}",
        f"  Spread: {result.spread_score:.0f}",
        f"  Volatility: {result.volatility_score:.0f}",
        f"  Time: {result.time_score:.0f}",
        "",
    ]

    if result.warnings:
        lines.append("--- WARNINGS ---")
        for w in result.warnings:
            lines.append(f"⚠ {w}")
        lines.append("")

    if result.recommendations:
        lines.append("--- RECOMMENDATIONS ---")
        for r in result.recommendations:
            lines.append(f"✓ {r}")
        lines.append("")

    lines.extend([
        "--- SUGGESTED PARAMETERS ---",
        f"Order Size: ${result.suggested_order_size:.2f}",
        f"Max Position: ${result.max_position:.2f}",
        f"Spike Threshold: {result.suggested_spike_threshold:.1f}%",
        "",
        "--- TOKEN IDs (for trading) ---",
    ])

    for token in info.tokens:
        lines.append(f"  {token.get('outcome', 'Unknown')}: {token.get('token_id', 'N/A')}")

    lines.append("=" * 60)

    return "\n".join(lines)
