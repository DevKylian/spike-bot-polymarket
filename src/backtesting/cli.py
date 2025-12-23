#!/usr/bin/env python3
"""
Backtesting CLI - Command-line interface for running backtests

Usage:
    python -m src.backtesting.cli backtest --token TOKEN_ID --days 30
    python -m src.backtesting.cli optimize --token TOKEN_ID --method grid
    python -m src.backtesting.cli report --file results.json

Commands:
    backtest    Run a single backtest
    optimize    Find optimal parameters
    analyze     Analyze historical data
    report      Generate report from results
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Setup path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.backtesting.data_loader import HistoricalDataLoader
from src.backtesting.engine import BacktestEngine, run_quick_backtest
from src.backtesting.metrics import PerformanceMetrics
from src.backtesting.optimizer import (
    ParameterOptimizer,
    OptimizationConfig,
    ParameterRange,
    find_optimal_parameters,
)
from src.backtesting.models import BacktestConfig, ParameterSet


def setup_logging(verbose: bool = False):
    """Setup logging for CLI."""
    import structlog
    import logging

    level = logging.DEBUG if verbose else logging.INFO

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(level=level, format="%(message)s")


async def cmd_backtest(args):
    """Run a backtest."""
    print(f"\n{'='*60}")
    print("           POLYMARKET SPIKE BOT - BACKTEST")
    print(f"{'='*60}\n")

    # Parse dates
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=args.days)

    print(f"Token ID:      {args.token}")
    print(f"Period:        {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Capital:       ${args.capital:,.2f}")
    print(f"Position Size: ${args.position_size:,.2f}")
    print()

    # Load data
    print("Loading historical data...")
    async with HistoricalDataLoader(use_cache=not args.no_cache) as loader:
        market_data = await loader.load_market_data(
            args.token,
            start_date,
            end_date,
            resolution="1m",
        )

        if not market_data.prices:
            print("ERROR: No price data available for this token.")
            return 1

        print(f"  Data points: {len(market_data.prices)}")
        print(f"  Market:      {market_data.market_name[:50] or 'Unknown'}...")
        print()

        # Run backtest
        print("Running backtest...")
        result = run_quick_backtest(
            {args.token: market_data},
            spike_threshold=args.threshold,
            spike_window=args.window,
            take_profit=args.take_profit,
            stop_loss=args.stop_loss,
            initial_capital=args.capital,
            position_size=args.position_size,
        )

        # Calculate metrics
        PerformanceMetrics.calculate_all(result)

        # Print report
        print()
        print(PerformanceMetrics.generate_report(result))

        # Save results if requested
        if args.output:
            save_results(result, args.output)
            print(f"\nResults saved to: {args.output}")

    return 0


async def cmd_optimize(args):
    """Run parameter optimization."""
    print(f"\n{'='*60}")
    print("        POLYMARKET SPIKE BOT - OPTIMIZATION")
    print(f"{'='*60}\n")

    # Parse dates
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=args.days)

    print(f"Token ID:      {args.token}")
    print(f"Period:        {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Method:        {args.method}")
    print(f"Metric:        {args.metric}")
    print()

    # Load data
    print("Loading historical data...")
    async with HistoricalDataLoader(use_cache=not args.no_cache) as loader:
        market_data = await loader.load_market_data(
            args.token,
            start_date,
            end_date,
            resolution="1m",
        )

        if not market_data.prices:
            print("ERROR: No price data available.")
            return 1

        print(f"  Data points: {len(market_data.prices)}")
        print()

        # Setup optimizer
        base_config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            token_ids=[args.token],
            initial_capital=args.capital,
            position_size_usdc=args.position_size,
        )

        opt_config = OptimizationConfig(
            optimization_metric=args.metric,
            spike_threshold_range=ParameterRange("spike_threshold", 1.0, 8.0, 1.0),
            spike_window_range=ParameterRange("spike_window", 1.0, 5.0, 1.0),
            take_profit_range=ParameterRange("take_profit", 1.0, 5.0, 0.5),
            stop_loss_range=ParameterRange("stop_loss", 2.0, 8.0, 1.0),
        )

        optimizer = ParameterOptimizer(
            {args.token: market_data},
            base_config,
            opt_config,
        )

        # Progress callback
        def progress(current, total):
            pct = current / total * 100
            bar = "=" * int(pct / 5) + ">" + " " * (20 - int(pct / 5))
            print(f"\r  Progress: [{bar}] {pct:.1f}%", end="", flush=True)

        print("Running optimization...")
        if args.method == "grid":
            result = optimizer.grid_search(progress_callback=progress)
        else:
            result = optimizer.random_search(
                n_iterations=args.iterations,
                progress_callback=progress,
            )

        print("\n")

        # Print results
        print("=" * 60)
        print("                  OPTIMIZATION RESULTS")
        print("=" * 60)
        print()
        print("BEST PARAMETERS:")
        print("-" * 40)
        print(f"  Spike Threshold:  {result.best_params.spike_threshold_percent}%")
        print(f"  Spike Window:     {result.best_params.spike_window_seconds}s")
        print(f"  Take Profit:      {result.best_params.take_profit_percent}%")
        print(f"  Stop Loss:        {result.best_params.stop_loss_percent}%")
        print(f"  Order Offset:     {result.best_params.limit_order_offset_percent}%")
        print()
        print("BEST RESULT:")
        print("-" * 40)
        print(f"  Total Return:     {result.best_result.total_return_percent:.2f}%")
        print(f"  Sharpe Ratio:     {result.best_result.metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Win Rate:         {result.best_result.win_rate:.1f}%")
        print(f"  Total Trades:     {result.best_result.total_trades}")
        print(f"  Max Drawdown:     {result.best_result.metrics.get('max_drawdown_percent', 0):.2f}%")
        print()
        print("OPTIMIZATION STATS:")
        print("-" * 40)
        print(f"  Combinations:     {result.total_combinations}")
        print(f"  Time:             {result.optimization_time_seconds:.1f}s")
        print()

        # Show top 5 results
        print("TOP 5 PARAMETER SETS:")
        print("-" * 40)
        top_5 = result.get_top_n(5, args.metric)
        for i, (params, res) in enumerate(top_5, 1):
            print(f"  {i}. Thr={params.spike_threshold_percent}% Win={params.spike_window_seconds}s "
                  f"TP={params.take_profit_percent}% SL={params.stop_loss_percent}% "
                  f"-> Return={res.total_return_percent:.1f}% Sharpe={res.metrics.get('sharpe_ratio', 0):.2f}")
        print()

        # Parameter sensitivity
        if result.parameter_sensitivity:
            print("PARAMETER SENSITIVITY:")
            print("-" * 40)
            for param, values in result.parameter_sensitivity.items():
                if values:
                    best_val = max(values, key=lambda x: x[1])
                    worst_val = min(values, key=lambda x: x[1])
                    print(f"  {param}:")
                    print(f"    Best:  {best_val[0]} -> {best_val[1]:.2f}")
                    print(f"    Worst: {worst_val[0]} -> {worst_val[1]:.2f}")
            print()

        # Save if requested
        if args.output:
            save_optimization_results(result, args.output)
            print(f"Results saved to: {args.output}")

        # Walk-forward if requested
        if args.walk_forward:
            print("\n" + "=" * 60)
            print("           WALK-FORWARD ANALYSIS")
            print("=" * 60 + "\n")

            wf_result = optimizer.walk_forward_optimization(n_windows=3)

            print("WINDOW RESULTS:")
            print("-" * 40)
            for window in wf_result["windows"]:
                print(f"  Window {window['window']}:")
                print(f"    In-Sample:  {window['in_sample_return']:.2f}% return, {window['in_sample_sharpe']:.2f} Sharpe")
                print(f"    Out-Sample: {window['out_sample_return']:.2f}% return, {window['out_sample_sharpe']:.2f} Sharpe")
                print()

            summary = wf_result["summary"]
            print("SUMMARY:")
            print("-" * 40)
            print(f"  Avg In-Sample Return:   {summary['avg_in_sample_return']:.2f}%")
            print(f"  Avg Out-Sample Return:  {summary['avg_out_sample_return']:.2f}%")
            print(f"  Overfitting Ratio:      {summary['overfitting_ratio']:.2f}")
            print(f"  Likely Overfit:         {'YES' if summary['is_overfit'] else 'NO'}")
            print()

    return 0


async def cmd_analyze(args):
    """Analyze historical data."""
    print(f"\n{'='*60}")
    print("         POLYMARKET SPIKE BOT - DATA ANALYSIS")
    print(f"{'='*60}\n")

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=args.days)

    async with HistoricalDataLoader(use_cache=not args.no_cache) as loader:
        if args.search:
            # Search for markets
            print(f"Searching for markets matching: {args.search}\n")
            markets = await loader.search_markets(args.search, limit=20)

            if not markets:
                print("No markets found.")
                return 0

            print(f"Found {len(markets)} markets:\n")
            for m in markets[:10]:
                question = m.get("question", "")[:60]
                cond_id = m.get("condition_id", "") or m.get("conditionId", "") or m.get("id", "")

                # Get token IDs for trading
                tokens = m.get("tokens", [])
                token_ids = [t.get("token_id", "") for t in tokens] if tokens else []

                # Handle volume as string or number
                volume = m.get("volume", 0)
                try:
                    volume = float(volume) if volume else 0
                except (ValueError, TypeError):
                    volume = 0

                print(f"  {question}...")
                print(f"    Condition ID: {cond_id}")
                if token_ids:
                    print(f"    Token IDs:    {', '.join(token_ids[:2])}")
                print(f"    Volume:       ${volume:,.0f}")
                print()

        elif args.token:
            # Analyze specific token
            print(f"Analyzing token: {args.token}\n")

            market_data = await loader.load_market_data(
                args.token,
                start_date,
                end_date,
            )

            if not market_data.prices:
                print("No data available for this token.")
                return 1

            # Calculate statistics
            prices = [p.price for p in market_data.prices]
            volumes = [p.volume for p in market_data.prices]

            print("DATA SUMMARY:")
            print("-" * 40)
            print(f"  Market:         {market_data.market_name[:50] or 'Unknown'}")
            print(f"  Data Points:    {len(prices)}")
            print(f"  Period:         {market_data.start_date.strftime('%Y-%m-%d')} to {market_data.end_date.strftime('%Y-%m-%d')}")
            print()

            print("PRICE STATISTICS:")
            print("-" * 40)
            print(f"  Min:            {min(prices):.4f}")
            print(f"  Max:            {max(prices):.4f}")
            print(f"  Mean:           {sum(prices) / len(prices):.4f}")
            print(f"  Current:        {prices[-1]:.4f}")
            print()

            # Calculate volatility
            returns = [(prices[i] - prices[i-1]) / prices[i-1] * 100
                      for i in range(1, len(prices)) if prices[i-1] > 0]
            if returns:
                avg_return = sum(returns) / len(returns)
                variance = sum((r - avg_return)**2 for r in returns) / len(returns)
                volatility = variance ** 0.5

                print("VOLATILITY:")
                print("-" * 40)
                print(f"  Avg Change:     {avg_return:.4f}%")
                print(f"  Volatility:     {volatility:.4f}%")
                print(f"  Max Up Move:    {max(returns):.2f}%")
                print(f"  Max Down Move:  {min(returns):.2f}%")
                print()

            # Count potential spikes
            spike_counts = {2: 0, 3: 0, 5: 0, 10: 0}
            for r in returns:
                for threshold in spike_counts:
                    if abs(r) >= threshold:
                        spike_counts[threshold] += 1

            print("SPIKE ANALYSIS:")
            print("-" * 40)
            for threshold, count in spike_counts.items():
                print(f"  >{threshold}% moves:     {count} ({count / len(returns) * 100:.2f}%)")
            print()

            print(f"  Total Volume:   ${sum(volumes):,.2f}")
            print(f"  Liquidity:      {market_data.liquidity_score:.1f}/100")

        else:
            # List available markets
            print("Loading available markets...\n")
            markets = await loader.get_markets(active_only=True, limit=20)

            print(f"Found {len(markets)} active markets:\n")
            for m in markets[:10]:
                question = m.get("question", "")[:60]
                print(f"  - {question}...")

    return 0


async def cmd_report(args):
    """Generate report from saved results."""
    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}")
        return 1

    with open(args.file, "r") as f:
        data = json.load(f)

    # Reconstruct result (simplified)
    print(f"\n{'='*60}")
    print("           BACKTEST REPORT")
    print(f"{'='*60}\n")

    if "config" in data:
        print("CONFIGURATION:")
        print("-" * 40)
        for key, value in data["config"].items():
            print(f"  {key}: {value}")
        print()

    if "metrics" in data:
        print("PERFORMANCE METRICS:")
        print("-" * 40)
        for key, value in data["metrics"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        print()

    if "trades" in data:
        print(f"TRADES: {len(data['trades'])} total")
        print("-" * 40)
        for trade in data["trades"][:5]:
            print(f"  {trade.get('timestamp', '')} | {trade.get('side', '')} | "
                  f"Entry: {trade.get('entry_price', 0):.4f} | "
                  f"PnL: ${trade.get('pnl', 0):.2f}")
        if len(data["trades"]) > 5:
            print(f"  ... and {len(data['trades']) - 5} more trades")

    return 0


def save_results(result, filepath: str):
    """Save backtest results to JSON."""
    data = {
        "config": result.config.to_dict(),
        "metrics": result.metrics,
        "summary": result.get_summary(),
        "trades": [
            {
                "id": t.id,
                "timestamp": t.timestamp.isoformat(),
                "token_id": t.token_id,
                "side": t.side.value,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "size": t.size,
                "pnl": t.pnl,
                "pnl_percent": t.pnl_percent,
                "hold_duration_seconds": t.hold_duration_seconds,
                "signal_reason": t.signal_reason.value,
            }
            for t in result.trades if t.is_closed
        ],
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def save_optimization_results(result, filepath: str):
    """Save optimization results to JSON."""
    data = {
        "best_params": result.best_params.to_dict(),
        "best_metrics": result.best_result.metrics,
        "total_combinations": result.total_combinations,
        "optimization_time_seconds": result.optimization_time_seconds,
        "parameter_sensitivity": result.parameter_sensitivity,
        "top_results": [
            {
                "params": params.to_dict(),
                "return": res.total_return_percent,
                "sharpe": res.metrics.get("sharpe_ratio", 0),
                "trades": res.total_trades,
            }
            for params, res in result.get_top_n(20)
        ],
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Spike Bot - Backtesting CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtest on a specific token
  python -m src.backtesting.cli backtest --token TOKEN_ID --days 30

  # Optimize parameters
  python -m src.backtesting.cli optimize --token TOKEN_ID --method random

  # Analyze market data
  python -m src.backtesting.cli analyze --search "election"

  # Generate report
  python -m src.backtesting.cli report --file results.json
        """,
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Backtest command
    bt_parser = subparsers.add_parser("backtest", help="Run a backtest")
    bt_parser.add_argument("--token", "-t", required=True, help="Token ID to backtest")
    bt_parser.add_argument("--days", "-d", type=int, default=30, help="Number of days of history")
    bt_parser.add_argument("--capital", "-c", type=float, default=10000, help="Initial capital")
    bt_parser.add_argument("--position-size", "-p", type=float, default=100, help="Position size (USDC)")
    bt_parser.add_argument("--threshold", type=float, default=3.0, help="Spike threshold (%)")
    bt_parser.add_argument("--window", type=float, default=2.0, help="Spike window (seconds)")
    bt_parser.add_argument("--take-profit", type=float, default=2.0, help="Take profit (%)")
    bt_parser.add_argument("--stop-loss", type=float, default=5.0, help="Stop loss (%)")
    bt_parser.add_argument("--output", "-o", help="Output file for results (JSON)")
    bt_parser.add_argument("--no-cache", action="store_true", help="Don't use cached data")

    # Optimize command
    opt_parser = subparsers.add_parser("optimize", help="Optimize parameters")
    opt_parser.add_argument("--token", "-t", required=True, help="Token ID")
    opt_parser.add_argument("--days", "-d", type=int, default=30, help="Days of history")
    opt_parser.add_argument("--method", "-m", choices=["grid", "random"], default="random")
    opt_parser.add_argument("--iterations", "-i", type=int, default=100, help="Iterations (random search)")
    opt_parser.add_argument("--metric", choices=["sharpe_ratio", "total_return", "profit_factor", "sortino_ratio"],
                          default="sharpe_ratio", help="Optimization metric")
    opt_parser.add_argument("--capital", "-c", type=float, default=10000, help="Initial capital")
    opt_parser.add_argument("--position-size", "-p", type=float, default=100, help="Position size")
    opt_parser.add_argument("--walk-forward", "-w", action="store_true", help="Run walk-forward analysis")
    opt_parser.add_argument("--output", "-o", help="Output file for results")
    opt_parser.add_argument("--no-cache", action="store_true", help="Don't use cached data")

    # Analyze command
    an_parser = subparsers.add_parser("analyze", help="Analyze market data")
    an_parser.add_argument("--token", "-t", help="Token ID to analyze")
    an_parser.add_argument("--search", "-s", help="Search for markets")
    an_parser.add_argument("--days", "-d", type=int, default=7, help="Days of history")
    an_parser.add_argument("--no-cache", action="store_true", help="Don't use cached data")

    # Report command
    rp_parser = subparsers.add_parser("report", help="Generate report from results")
    rp_parser.add_argument("--file", "-f", required=True, help="Results JSON file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    setup_logging(args.verbose)

    # Run command
    if args.command == "backtest":
        return asyncio.run(cmd_backtest(args))
    elif args.command == "optimize":
        return asyncio.run(cmd_optimize(args))
    elif args.command == "analyze":
        return asyncio.run(cmd_analyze(args))
    elif args.command == "report":
        return asyncio.run(cmd_report(args))


if __name__ == "__main__":
    sys.exit(main() or 0)
