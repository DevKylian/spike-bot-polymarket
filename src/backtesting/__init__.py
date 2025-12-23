"""
Backtesting Module - Polymarket Spike Bot

Complete backtesting engine for testing spike hunting strategies
on historical Polymarket data.

Features:
- Historical data downloading from Polymarket APIs
- Realistic trade simulation with slippage and fees
- Comprehensive performance metrics
- Parameter optimization
- Detailed reporting
"""

from .models import (
    BacktestConfig,
    BacktestResult,
    Trade,
    OHLCV,
    PriceData,
    OptimizationResult,
    ParameterSet,
)
from .data_loader import HistoricalDataLoader
from .engine import BacktestEngine
from .optimizer import ParameterOptimizer
from .metrics import PerformanceMetrics

__all__ = [
    # Models
    "BacktestConfig",
    "BacktestResult",
    "Trade",
    "OHLCV",
    "PriceData",
    "OptimizationResult",
    "ParameterSet",
    # Components
    "HistoricalDataLoader",
    "BacktestEngine",
    "ParameterOptimizer",
    "PerformanceMetrics",
]
