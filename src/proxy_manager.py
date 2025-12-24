"""
Proxy Manager Module - Polymarket Spike Bot

Handles proxy configuration and free proxy fetching for geo-blocked regions.
Supports HTTP, HTTPS, and SOCKS5 proxies.
"""

import asyncio
import random
import time
from dataclasses import dataclass

import aiohttp
import structlog

try:
    from aiohttp_socks import ProxyConnector
    SOCKS_AVAILABLE = True
except ImportError:
    SOCKS_AVAILABLE = False

logger = structlog.get_logger(__name__)

# Free proxy sources (US and worldwide proxies)
FREE_PROXY_SOURCES = [
    # ProxyScrape - most reliable
    "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=10000&country=US&ssl=all&anonymity=all",
    "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=10000&country=all",
    "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=socks5&timeout=10000&country=all",
    # Free Proxy List
    "https://www.proxy-list.download/api/v1/get?type=http&country=US",
    "https://www.proxy-list.download/api/v1/get?type=https&country=US",
    "https://www.proxy-list.download/api/v1/get?type=http",
    "https://www.proxy-list.download/api/v1/get?type=socks5",
    # Geonode
    "https://proxylist.geonode.com/api/proxy-list?limit=100&page=1&sort_by=lastChecked&sort_type=desc&protocols=http%2Chttps%2Csocks5",
    # GitHub lists - very reliable sources
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
    "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/socks5.txt",
    "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/https.txt",
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/socks5.txt",
    "https://raw.githubusercontent.com/hookzof/socks5_list/master/proxy.txt",
    "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",
    "https://raw.githubusercontent.com/sunny9577/proxy-scraper/master/proxies.txt",
    "https://raw.githubusercontent.com/roosterkid/openproxylist/main/HTTPS_RAW.txt",
    "https://raw.githubusercontent.com/mertguvencli/http-proxy-list/main/proxy-list/data.txt",
    "https://raw.githubusercontent.com/jetkai/proxy-list/main/online-proxies/txt/proxies-http.txt",
    "https://raw.githubusercontent.com/jetkai/proxy-list/main/online-proxies/txt/proxies-https.txt",
    "https://raw.githubusercontent.com/jetkai/proxy-list/main/online-proxies/txt/proxies-socks5.txt",
    "https://raw.githubusercontent.com/prxchk/proxy-list/main/http.txt",
    "https://raw.githubusercontent.com/prxchk/proxy-list/main/socks5.txt",
    # Spys.me
    "https://spys.me/proxy.txt",
    "https://spys.me/socks.txt",
    # Other sources
    "https://openproxy.space/list/http",
    "https://openproxy.space/list/socks5",
]

# Backup proxies (tested working proxies - update regularly)
BACKUP_PROXIES = [
    "http://38.154.227.167:5868",
    "http://185.199.229.156:7492",
    "http://185.199.228.220:7300",
    "http://104.223.135.178:10000",
    "http://204.44.69.89:10001",
    "http://173.211.0.148:6641",
    "http://206.41.172.74:3128",
    "http://47.251.70.179:80",
    "http://20.206.106.192:8123",
    "http://50.217.226.41:80",
    "http://50.200.12.66:80",
    "http://50.218.57.70:80",
    "http://50.174.7.152:80",
    "http://50.168.72.113:80",
    "http://50.207.199.80:80",
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

        # Test proxies in parallel (faster)
        logger.info(f"Testing {len(self._proxies)} proxies in parallel...")

        # Test in batches of 20
        batch_size = 20
        for i in range(0, len(self._proxies), batch_size):
            batch = self._proxies[i:i + batch_size]
            tasks = [self._test_proxy(p.url) for p in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for j, result in enumerate(results):
                if result is True:
                    self._current_proxy = batch[j].url
                    logger.info("Found working proxy", proxy=self._current_proxy[:30] + "...")
                    return self._current_proxy

        # Try backup proxies in parallel
        logger.info("Testing backup proxies...")
        tasks = [self._test_proxy(p) for p in BACKUP_PROXIES]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if result is True:
                self._current_proxy = BACKUP_PROXIES[i]
                logger.info("Using backup proxy", proxy=self._current_proxy[:30] + "...")
                return self._current_proxy

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

    async def _test_proxy(self, proxy_url: str, timeout: float = 8.0) -> bool:
        """Test if a proxy is working by making a request to Polymarket."""
        try:
            # Handle SOCKS5 proxies
            if proxy_url.startswith("socks5://") or proxy_url.startswith("socks4://"):
                if not SOCKS_AVAILABLE:
                    return False
                connector = ProxyConnector.from_url(proxy_url)
                async with aiohttp.ClientSession(
                    connector=connector,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as session:
                    async with session.get("https://clob.polymarket.com/time") as response:
                        if response.status == 200:
                            return True
            else:
                # HTTP/HTTPS proxy
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as session:
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
