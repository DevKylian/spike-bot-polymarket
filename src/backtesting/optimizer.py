"""
Parameter Optimizer - Strategy parameter optimization

Provides multiple optimization methods:
- Grid search (exhaustive)
- Random search
- Walk-forward optimization
- Monte Carlo analysis

Features:
- Multi-objective optimization (return vs risk)
- Parallel execution
- Parameter sensitivity analysis
- Overfitting detection
"""

import asyncio
import itertools
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

import structlog

from .models import (
    BacktestConfig,
    BacktestResult,
    ParameterSet,
    OptimizationResult,
    MarketData,
)
from .engine import BacktestEngine
from .metrics import PerformanceMetrics


logger = structlog.get_logger(__name__)


@dataclass
class ParameterRange:
    """Defines a range of values for a parameter."""
    name: str
    min_value: float
    max_value: float
    step: float
    is_integer: bool = False

    def get_values(self) -> list[float]:
        """Generate all values in the range."""
        values = []
        current = self.min_value
        while current <= self.max_value:
            if self.is_integer:
                values.append(int(current))
            else:
                values.append(round(current, 4))
            current += self.step
        return values

    def get_random(self) -> float:
        """Get a random value in the range."""
        value = random.uniform(self.min_value, self.max_value)
        if self.is_integer:
            return int(value)
        return round(value, 4)


@dataclass
class OptimizationConfig:
    """Configuration for parameter optimization."""
    # Parameter ranges
    spike_threshold_range: ParameterRange = field(
        default_factory=lambda: ParameterRange("spike_threshold", 1.0, 10.0, 0.5)
    )
    spike_window_range: ParameterRange = field(
        default_factory=lambda: ParameterRange("spike_window", 1.0, 10.0, 1.0)
    )
    take_profit_range: ParameterRange = field(
        default_factory=lambda: ParameterRange("take_profit", 0.5, 5.0, 0.5)
    )
    stop_loss_range: ParameterRange = field(
        default_factory=lambda: ParameterRange("stop_loss", 2.0, 10.0, 1.0)
    )
    offset_range: ParameterRange = field(
        default_factory=lambda: ParameterRange("offset", 0.1, 1.0, 0.1)
    )

    # Optimization settings
    optimization_metric: str = "sharpe_ratio"  # Metric to optimize
    min_trades: int = 10  # Minimum trades for valid result
    max_workers: int = 4  # Parallel workers

    # Walk-forward settings
    in_sample_ratio: float = 0.7  # 70% in-sample, 30% out-of-sample
    walk_forward_windows: int = 3  # Number of windows for walk-forward


class ParameterOptimizer:
    """
    Optimizes strategy parameters using various methods.

    Supports grid search, random search, and walk-forward optimization.
    """

    def __init__(
        self,
        market_data: dict[str, MarketData],
        base_config: BacktestConfig,
        opt_config: OptimizationConfig | None = None,
    ):
        self.market_data = market_data
        self.base_config = base_config
        self.opt_config = opt_config or OptimizationConfig()

        # Results storage
        self.all_results: list[tuple[ParameterSet, BacktestResult]] = []

    def grid_search(
        self,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> OptimizationResult:
        """
        Exhaustive grid search over all parameter combinations.

        This tests every combination of parameters and finds the best one.
        Can be slow for large parameter spaces.
        """
        import time
        start_time = time.time()

        # Generate all parameter combinations
        param_values = {
            "spike_threshold": self.opt_config.spike_threshold_range.get_values(),
            "spike_window": self.opt_config.spike_window_range.get_values(),
            "take_profit": self.opt_config.take_profit_range.get_values(),
            "stop_loss": self.opt_config.stop_loss_range.get_values(),
            "offset": self.opt_config.offset_range.get_values(),
        }

        # Create all combinations
        keys = list(param_values.keys())
        combinations = list(itertools.product(*[param_values[k] for k in keys]))

        total = len(combinations)
        logger.info(f"Grid search: {total} parameter combinations")

        self.all_results = []
        best_metric = float('-inf')
        best_params = None
        best_result = None

        for i, combo in enumerate(combinations):
            params = ParameterSet(
                spike_threshold_percent=combo[0],
                spike_window_seconds=combo[1],
                take_profit_percent=combo[2],
                stop_loss_percent=combo[3],
                limit_order_offset_percent=combo[4],
            )

            result = self._run_backtest(params)

            if result and result.total_trades >= self.opt_config.min_trades:
                PerformanceMetrics.calculate_all(result)
                metric_value = self._get_metric(result)

                self.all_results.append((params, result))

                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params
                    best_result = result

            if progress_callback:
                progress_callback(i + 1, total)

            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%)")

        elapsed = time.time() - start_time

        if best_params is None or best_result is None:
            raise ValueError("No valid results found in optimization")

        # Calculate parameter sensitivity
        sensitivity = self._calculate_sensitivity()

        result = OptimizationResult(
            best_params=best_params,
            best_result=best_result,
            all_results=self.all_results,
            total_combinations=total,
            optimization_time_seconds=elapsed,
            parameter_sensitivity=sensitivity,
        )

        logger.info(
            "Grid search completed",
            best_metric=f"{best_metric:.4f}",
            total_combinations=total,
            time=f"{elapsed:.1f}s",
        )

        return result

    def random_search(
        self,
        n_iterations: int = 100,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> OptimizationResult:
        """
        Random search for parameter optimization.

        More efficient than grid search when parameter space is large.
        """
        import time
        start_time = time.time()

        logger.info(f"Random search: {n_iterations} iterations")

        self.all_results = []
        best_metric = float('-inf')
        best_params = None
        best_result = None

        for i in range(n_iterations):
            params = ParameterSet(
                spike_threshold_percent=self.opt_config.spike_threshold_range.get_random(),
                spike_window_seconds=self.opt_config.spike_window_range.get_random(),
                take_profit_percent=self.opt_config.take_profit_range.get_random(),
                stop_loss_percent=self.opt_config.stop_loss_range.get_random(),
                limit_order_offset_percent=self.opt_config.offset_range.get_random(),
            )

            result = self._run_backtest(params)

            if result and result.total_trades >= self.opt_config.min_trades:
                PerformanceMetrics.calculate_all(result)
                metric_value = self._get_metric(result)

                self.all_results.append((params, result))

                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params
                    best_result = result

            if progress_callback:
                progress_callback(i + 1, n_iterations)

        elapsed = time.time() - start_time

        if best_params is None or best_result is None:
            raise ValueError("No valid results found in optimization")

        result = OptimizationResult(
            best_params=best_params,
            best_result=best_result,
            all_results=self.all_results,
            total_combinations=n_iterations,
            optimization_time_seconds=elapsed,
            parameter_sensitivity=self._calculate_sensitivity(),
        )

        logger.info(
            "Random search completed",
            best_metric=f"{best_metric:.4f}",
            iterations=n_iterations,
            time=f"{elapsed:.1f}s",
        )

        return result

    def walk_forward_optimization(
        self,
        n_windows: int | None = None,
    ) -> dict[str, Any]:
        """
        Walk-forward optimization to test for overfitting.

        Splits data into multiple in-sample/out-of-sample windows,
        optimizes on in-sample, and tests on out-of-sample.
        """
        n_windows = n_windows or self.opt_config.walk_forward_windows

        # Get date range from data
        all_timestamps = []
        for data in self.market_data.values():
            all_timestamps.extend([p.timestamp for p in data.prices])

        if not all_timestamps:
            raise ValueError("No data available")

        start_date = min(all_timestamps)
        end_date = max(all_timestamps)
        total_days = (end_date - start_date).days

        window_days = total_days // n_windows
        in_sample_days = int(window_days * self.opt_config.in_sample_ratio)

        results = []
        window_results = []

        logger.info(
            "Walk-forward optimization",
            windows=n_windows,
            window_days=window_days,
            in_sample_days=in_sample_days,
        )

        for i in range(n_windows):
            window_start = start_date + timedelta(days=i * window_days)
            in_sample_end = window_start + timedelta(days=in_sample_days)
            window_end = window_start + timedelta(days=window_days)

            logger.info(
                f"Window {i + 1}/{n_windows}",
                start=window_start.strftime("%Y-%m-%d"),
                in_sample_end=in_sample_end.strftime("%Y-%m-%d"),
                end=window_end.strftime("%Y-%m-%d"),
            )

            # Filter data for in-sample period
            in_sample_data = self._filter_data_by_date(window_start, in_sample_end)

            if not self._has_enough_data(in_sample_data):
                logger.warning(f"Window {i + 1}: Not enough in-sample data")
                continue

            # Optimize on in-sample
            in_sample_optimizer = ParameterOptimizer(
                in_sample_data,
                self.base_config,
                self.opt_config,
            )
            opt_result = in_sample_optimizer.random_search(n_iterations=50)

            # Test on out-of-sample
            out_sample_data = self._filter_data_by_date(in_sample_end, window_end)

            if not self._has_enough_data(out_sample_data):
                logger.warning(f"Window {i + 1}: Not enough out-of-sample data")
                continue

            out_sample_result = self._run_backtest_with_params(
                opt_result.best_params,
                out_sample_data,
            )

            if out_sample_result:
                PerformanceMetrics.calculate_all(out_sample_result)

                window_results.append({
                    "window": i + 1,
                    "in_sample_start": window_start.isoformat(),
                    "in_sample_end": in_sample_end.isoformat(),
                    "out_sample_end": window_end.isoformat(),
                    "best_params": opt_result.best_params.to_dict(),
                    "in_sample_return": opt_result.best_result.total_return_percent,
                    "in_sample_sharpe": opt_result.best_result.metrics.get("sharpe_ratio", 0),
                    "out_sample_return": out_sample_result.total_return_percent,
                    "out_sample_sharpe": out_sample_result.metrics.get("sharpe_ratio", 0),
                    "out_sample_trades": out_sample_result.total_trades,
                })

                results.append((opt_result.best_params, out_sample_result))

        # Calculate aggregate statistics
        if window_results:
            avg_in_sample_return = sum(w["in_sample_return"] for w in window_results) / len(window_results)
            avg_out_sample_return = sum(w["out_sample_return"] for w in window_results) / len(window_results)
            avg_in_sample_sharpe = sum(w["in_sample_sharpe"] for w in window_results) / len(window_results)
            avg_out_sample_sharpe = sum(w["out_sample_sharpe"] for w in window_results) / len(window_results)

            # Overfitting ratio (in-sample / out-sample performance)
            overfitting_ratio = avg_in_sample_return / avg_out_sample_return if avg_out_sample_return != 0 else float('inf')
        else:
            avg_in_sample_return = avg_out_sample_return = 0
            avg_in_sample_sharpe = avg_out_sample_sharpe = 0
            overfitting_ratio = 0

        return {
            "windows": window_results,
            "summary": {
                "avg_in_sample_return": round(avg_in_sample_return, 2),
                "avg_out_sample_return": round(avg_out_sample_return, 2),
                "avg_in_sample_sharpe": round(avg_in_sample_sharpe, 2),
                "avg_out_sample_sharpe": round(avg_out_sample_sharpe, 2),
                "overfitting_ratio": round(overfitting_ratio, 2),
                "is_overfit": overfitting_ratio > 2.0,  # If in-sample is 2x better, likely overfit
            },
            "out_sample_results": results,
        }

    def monte_carlo_analysis(
        self,
        params: ParameterSet,
        n_simulations: int = 1000,
    ) -> dict[str, Any]:
        """
        Monte Carlo analysis of strategy robustness.

        Randomly shuffles trade results to estimate distribution
        of possible outcomes.
        """
        # First run backtest with given params
        result = self._run_backtest(params)
        if not result or not result.trades:
            raise ValueError("No trades to analyze")

        PerformanceMetrics.calculate_all(result)

        # Get trade P&Ls
        trade_pnls = [t.pnl for t in result.trades if t.is_closed]

        if len(trade_pnls) < 10:
            raise ValueError("Need at least 10 trades for Monte Carlo")

        # Run simulations
        simulated_returns = []
        simulated_max_drawdowns = []

        for _ in range(n_simulations):
            # Shuffle trades and calculate equity curve
            shuffled = trade_pnls.copy()
            random.shuffle(shuffled)

            equity = self.base_config.initial_capital
            peak = equity
            max_dd = 0

            for pnl in shuffled:
                equity += pnl
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak * 100
                max_dd = max(max_dd, dd)

            final_return = (equity - self.base_config.initial_capital) / self.base_config.initial_capital * 100
            simulated_returns.append(final_return)
            simulated_max_drawdowns.append(max_dd)

        # Calculate percentiles
        sorted_returns = sorted(simulated_returns)
        sorted_drawdowns = sorted(simulated_max_drawdowns)

        def percentile(data: list, p: int) -> float:
            idx = int(len(data) * p / 100)
            return data[min(idx, len(data) - 1)]

        return {
            "original_return": result.total_return_percent,
            "original_max_drawdown": result.metrics.get("max_drawdown_percent", 0),
            "simulations": n_simulations,
            "return_distribution": {
                "min": min(simulated_returns),
                "p5": percentile(sorted_returns, 5),
                "p25": percentile(sorted_returns, 25),
                "median": percentile(sorted_returns, 50),
                "p75": percentile(sorted_returns, 75),
                "p95": percentile(sorted_returns, 95),
                "max": max(simulated_returns),
                "mean": sum(simulated_returns) / len(simulated_returns),
            },
            "drawdown_distribution": {
                "min": min(simulated_drawdowns),
                "p5": percentile(sorted_drawdowns, 5),
                "median": percentile(sorted_drawdowns, 50),
                "p95": percentile(sorted_drawdowns, 95),
                "max": max(simulated_drawdowns),
            },
            "probability_of_loss": len([r for r in simulated_returns if r < 0]) / n_simulations * 100,
            "probability_of_ruin": len([dd for dd in simulated_drawdowns if dd > 50]) / n_simulations * 100,
        }

    def _run_backtest(self, params: ParameterSet) -> BacktestResult | None:
        """Run a backtest with given parameters."""
        return self._run_backtest_with_params(params, self.market_data)

    def _run_backtest_with_params(
        self,
        params: ParameterSet,
        market_data: dict[str, MarketData],
    ) -> BacktestResult | None:
        """Run a backtest with given parameters and data."""
        try:
            # Get date range
            all_timestamps = []
            for data in market_data.values():
                all_timestamps.extend([p.timestamp for p in data.prices])

            if not all_timestamps:
                return None

            config = BacktestConfig(
                start_date=min(all_timestamps),
                end_date=max(all_timestamps),
                token_ids=list(market_data.keys()),
                spike_threshold_percent=params.spike_threshold_percent,
                spike_window_seconds=params.spike_window_seconds,
                limit_order_offset_percent=params.limit_order_offset_percent,
                take_profit_percent=params.take_profit_percent,
                stop_loss_percent=params.stop_loss_percent,
                initial_capital=self.base_config.initial_capital,
                position_size_usdc=params.position_size_usdc or self.base_config.position_size_usdc,
                max_positions=self.base_config.max_positions,
                slippage_percent=self.base_config.slippage_percent,
                fee_percent=self.base_config.fee_percent,
            )

            engine = BacktestEngine(config)
            return engine.run(market_data)

        except Exception as e:
            logger.debug(f"Backtest failed: {e}")
            return None

    def _get_metric(self, result: BacktestResult) -> float:
        """Get the optimization metric value from a result."""
        metric = self.opt_config.optimization_metric

        if metric == "sharpe_ratio":
            return result.metrics.get("sharpe_ratio", 0)
        elif metric == "total_return":
            return result.total_return_percent
        elif metric == "profit_factor":
            return result.metrics.get("profit_factor", 0)
        elif metric == "sortino_ratio":
            return result.metrics.get("sortino_ratio", 0)
        elif metric == "calmar_ratio":
            return result.metrics.get("calmar_ratio", 0)
        elif metric == "win_rate":
            return result.win_rate
        else:
            return result.total_pnl

    def _filter_data_by_date(
        self,
        start: datetime,
        end: datetime,
    ) -> dict[str, MarketData]:
        """Filter market data to a specific date range."""
        filtered = {}

        for token_id, data in self.market_data.items():
            prices = [p for p in data.prices if start <= p.timestamp < end]

            if prices:
                filtered[token_id] = MarketData(
                    token_id=token_id,
                    market_name=data.market_name,
                    condition_id=data.condition_id,
                    start_date=start,
                    end_date=end,
                    prices=prices,
                    ohlcv=[o for o in data.ohlcv if start <= o.timestamp < end],
                    total_volume=data.total_volume,
                    avg_daily_volume=data.avg_daily_volume,
                    avg_spread=data.avg_spread,
                    liquidity_score=data.liquidity_score,
                )

        return filtered

    def _has_enough_data(self, data: dict[str, MarketData]) -> bool:
        """Check if there's enough data for analysis."""
        total_points = sum(len(d.prices) for d in data.values())
        return total_points >= 100

    def _calculate_sensitivity(self) -> dict[str, list[tuple[float, float]]]:
        """
        Calculate parameter sensitivity.

        Returns mapping of parameter -> [(value, metric_value), ...]
        """
        sensitivity = {}

        if not self.all_results:
            return sensitivity

        # Group results by parameter values
        param_metrics = {
            "spike_threshold": {},
            "spike_window": {},
            "take_profit": {},
            "stop_loss": {},
            "offset": {},
        }

        for params, result in self.all_results:
            metric = self._get_metric(result)

            # Track metrics for each parameter value
            if params.spike_threshold_percent not in param_metrics["spike_threshold"]:
                param_metrics["spike_threshold"][params.spike_threshold_percent] = []
            param_metrics["spike_threshold"][params.spike_threshold_percent].append(metric)

            if params.spike_window_seconds not in param_metrics["spike_window"]:
                param_metrics["spike_window"][params.spike_window_seconds] = []
            param_metrics["spike_window"][params.spike_window_seconds].append(metric)

            if params.take_profit_percent not in param_metrics["take_profit"]:
                param_metrics["take_profit"][params.take_profit_percent] = []
            param_metrics["take_profit"][params.take_profit_percent].append(metric)

            if params.stop_loss_percent not in param_metrics["stop_loss"]:
                param_metrics["stop_loss"][params.stop_loss_percent] = []
            param_metrics["stop_loss"][params.stop_loss_percent].append(metric)

            if params.limit_order_offset_percent not in param_metrics["offset"]:
                param_metrics["offset"][params.limit_order_offset_percent] = []
            param_metrics["offset"][params.limit_order_offset_percent].append(metric)

        # Average metrics for each parameter value
        for param, values in param_metrics.items():
            sensitivity[param] = sorted([
                (val, sum(metrics) / len(metrics))
                for val, metrics in values.items()
            ])

        return sensitivity


def find_optimal_parameters(
    market_data: dict[str, MarketData],
    method: str = "random",
    n_iterations: int = 100,
    optimization_metric: str = "sharpe_ratio",
) -> OptimizationResult:
    """
    Convenience function to find optimal parameters.

    Args:
        market_data: Historical market data
        method: Optimization method ("grid" or "random")
        n_iterations: Number of iterations for random search
        optimization_metric: Metric to optimize

    Returns:
        OptimizationResult with best parameters
    """
    # Get date range
    all_timestamps = []
    for data in market_data.values():
        all_timestamps.extend([p.timestamp for p in data.prices])

    base_config = BacktestConfig(
        start_date=min(all_timestamps),
        end_date=max(all_timestamps),
        token_ids=list(market_data.keys()),
    )

    opt_config = OptimizationConfig(
        optimization_metric=optimization_metric,
    )

    optimizer = ParameterOptimizer(market_data, base_config, opt_config)

    if method == "grid":
        return optimizer.grid_search()
    else:
        return optimizer.random_search(n_iterations=n_iterations)
