"""
Proxy Manager Module - Polymarket Spike Bot

Handles proxy configuration and free proxy fetching for geo-blocked regions.
"""

import asyncio
import random
from dataclasses import dataclass

import aiohttp
import structlog

logger = structlog.get_logger(__name__)

# Free proxy sources (US proxies)
FREE_PROXY_SOURCES = [
    "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=5000&country=US&ssl=yes&anonymity=elite",
    "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt",
    "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/https.txt",
]

# Backup proxies (may not always work)
BACKUP_PROXIES = [
    "http://38.154.227.167:5868",
    "http://185.199.229.156:7492",
    "http://185.199.228.220:7300",
]


@dataclass
class ProxyInfo:
    """Information about a proxy."""
    url: str
    protocol: str  # http, https, socks5
    host: str
    port: int
    country: str = "US"
    latency_ms: float = 0.0
    working: bool = False


class ProxyManager:
    """
    Manages proxy connections for bypassing geo-restrictions.

    Features:
    - Fetches free US proxies from multiple sources
    - Tests proxy connectivity
    - Rotates between working proxies
    - Falls back to backup proxies
    """

    def __init__(self):
        self._proxies: list[ProxyInfo] = []
        self._current_proxy: str | None = None
        self._last_fetch: float = 0
        self._fetch_interval = 300  # Refresh every 5 minutes

    async def get_working_proxy(self) -> str | None:
        """
        Get a working US proxy.

        Returns proxy URL or None if no proxy available.
        """
        # Try current proxy first
        if self._current_proxy:
            if await self._test_proxy(self._current_proxy):
                return self._current_proxy

        # Fetch new proxies
        await self._fetch_proxies()

        # Find a working proxy
        for proxy in self._proxies:
            if await self._test_proxy(proxy.url):
                self._current_proxy = proxy.url
                logger.info("Found working proxy", proxy=proxy.url[:30] + "...")
                return proxy.url

        # Try backup proxies
        for proxy_url in BACKUP_PROXIES:
            if await self._test_proxy(proxy_url):
                self._current_proxy = proxy_url
                logger.info("Using backup proxy", proxy=proxy_url[:30] + "...")
                return proxy_url

        logger.error("No working proxy found")
        return None

    async def _fetch_proxies(self) -> None:
        """Fetch proxies from free sources."""
        logger.info("Fetching free US proxies...")

        all_proxies = []

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            for source in FREE_PROXY_SOURCES:
                try:
                    async with session.get(source) as response:
                        if response.status == 200:
                            text = await response.text()
                            proxies = self._parse_proxy_list(text)
                            all_proxies.extend(proxies)
                            logger.debug(f"Fetched {len(proxies)} proxies from source")
                except Exception as e:
                    logger.debug(f"Failed to fetch from source: {e}")
                    continue

        # Shuffle and limit
        random.shuffle(all_proxies)
        self._proxies = all_proxies[:50]  # Keep top 50

        logger.info(f"Loaded {len(self._proxies)} proxies")

    def _parse_proxy_list(self, text: str) -> list[ProxyInfo]:
        """Parse proxy list from text."""
        proxies = []

        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                # Format: ip:port or protocol://ip:port
                if "://" in line:
                    protocol = line.split("://")[0]
                    host_port = line.split("://")[1]
                else:
                    protocol = "http"
                    host_port = line

                if ":" in host_port:
                    host, port_str = host_port.split(":")[:2]
                    port = int(port_str)

                    proxy_url = f"{protocol}://{host}:{port}"
                    proxies.append(ProxyInfo(
                        url=proxy_url,
                        protocol=protocol,
                        host=host,
                        port=port,
                    ))
            except (ValueError, IndexError):
                continue

        return proxies

    async def _test_proxy(self, proxy_url: str, timeout: float = 5.0) -> bool:
        """Test if a proxy is working by making a request to Polymarket."""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                # Test with Polymarket API
                async with session.get(
                    "https://clob.polymarket.com/time",
                    proxy=proxy_url,
                ) as response:
                    if response.status == 200:
                        return True
        except Exception:
            pass

        return False

    def get_current_proxy(self) -> str | None:
        """Get the currently active proxy."""
        return self._current_proxy

    def clear_current_proxy(self) -> None:
        """Clear the current proxy (force rotation)."""
        self._current_proxy = None


# Global proxy manager instance
_proxy_manager: ProxyManager | None = None


def get_proxy_manager() -> ProxyManager:
    """Get the global proxy manager instance."""
    global _proxy_manager
    if _proxy_manager is None:
        _proxy_manager = ProxyManager()
    return _proxy_manager


async def get_working_proxy() -> str | None:
    """Convenience function to get a working proxy."""
    manager = get_proxy_manager()
    return await manager.get_working_proxy()
