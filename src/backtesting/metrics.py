"""
Performance Metrics - Comprehensive trading performance analysis

Calculates all standard trading metrics:
- Return metrics (total, CAGR, etc.)
- Risk metrics (Sharpe, Sortino, max drawdown)
- Trade analysis (win rate, profit factor, etc.)
- Time-based analysis (monthly, daily returns)
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import structlog

from .models import BacktestResult, Trade, EquityPoint


logger = structlog.get_logger(__name__)


@dataclass
class DrawdownInfo:
    """Information about a drawdown period."""
    start_date: datetime
    end_date: datetime | None
    recovery_date: datetime | None
    peak_equity: float
    trough_equity: float
    drawdown_percent: float
    duration_days: float
    recovery_days: float | None


class PerformanceMetrics:
    """
    Calculates comprehensive performance metrics for backtests.

    All metrics are calculated following industry standards and
    are suitable for comparing different strategies.
    """

    # Risk-free rate for Sharpe/Sortino (annualized)
    RISK_FREE_RATE = 0.05  # 5% annual

    @classmethod
    def calculate_all(cls, result: BacktestResult) -> dict[str, Any]:
        """
        Calculate all performance metrics for a backtest result.

        Returns a dictionary with all metrics that can be stored
        in BacktestResult.metrics.
        """
        metrics = {}

        # Return metrics
        metrics.update(cls.calculate_return_metrics(result))

        # Risk metrics
        metrics.update(cls.calculate_risk_metrics(result))

        # Trade analysis
        metrics.update(cls.calculate_trade_metrics(result))

        # Time-based analysis
        metrics.update(cls.calculate_time_metrics(result))

        # Store in result
        result.metrics = metrics

        return metrics

    @classmethod
    def calculate_return_metrics(cls, result: BacktestResult) -> dict[str, float]:
        """Calculate return-related metrics."""
        initial = result.config.initial_capital
        final = result.final_equity

        if initial <= 0:
            return {}

        # Total return
        total_return = (final - initial) / initial * 100
        total_pnl = final - initial

        # Trading days
        if result.equity_curve:
            start_date = result.equity_curve[0].timestamp
            end_date = result.equity_curve[-1].timestamp
            days = max((end_date - start_date).days, 1)
        else:
            days = max((result.config.end_date - result.config.start_date).days, 1)

        years = days / 365.0

        # CAGR (Compound Annual Growth Rate)
        if years > 0 and final > 0:
            cagr = (pow(final / initial, 1 / years) - 1) * 100
        else:
            cagr = 0.0

        # Average daily return
        daily_returns = cls._calculate_daily_returns(result)
        avg_daily_return = sum(daily_returns) / len(daily_returns) if daily_returns else 0.0

        # Annualized return (from daily)
        annualized_return = avg_daily_return * 252  # Trading days

        return {
            "total_return_percent": round(total_return, 2),
            "total_pnl": round(total_pnl, 2),
            "cagr_percent": round(cagr, 2),
            "avg_daily_return_percent": round(avg_daily_return, 4),
            "annualized_return_percent": round(annualized_return, 2),
            "trading_days": days,
        }

    @classmethod
    def calculate_risk_metrics(cls, result: BacktestResult) -> dict[str, float]:
        """Calculate risk-related metrics."""
        metrics = {}

        # Daily returns for volatility calculations
        daily_returns = cls._calculate_daily_returns(result)

        if not daily_returns:
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown_percent": 0.0,
                "volatility_annual_percent": 0.0,
                "var_95_percent": 0.0,
                "calmar_ratio": 0.0,
            }

        # Volatility (annualized)
        if len(daily_returns) > 1:
            variance = sum((r - sum(daily_returns)/len(daily_returns))**2 for r in daily_returns) / (len(daily_returns) - 1)
            daily_volatility = math.sqrt(variance)
            annual_volatility = daily_volatility * math.sqrt(252)
        else:
            daily_volatility = 0.0
            annual_volatility = 0.0

        metrics["volatility_annual_percent"] = round(annual_volatility, 2)

        # Sharpe Ratio
        avg_daily_return = sum(daily_returns) / len(daily_returns)
        daily_risk_free = cls.RISK_FREE_RATE / 252

        if daily_volatility > 0:
            sharpe = (avg_daily_return - daily_risk_free) / daily_volatility * math.sqrt(252)
        else:
            sharpe = 0.0

        metrics["sharpe_ratio"] = round(sharpe, 2)

        # Sortino Ratio (uses downside deviation)
        negative_returns = [r for r in daily_returns if r < 0]
        if negative_returns:
            downside_variance = sum(r**2 for r in negative_returns) / len(negative_returns)
            downside_deviation = math.sqrt(downside_variance) * math.sqrt(252)

            if downside_deviation > 0:
                annualized_return = avg_daily_return * 252
                sortino = (annualized_return - cls.RISK_FREE_RATE) / downside_deviation
            else:
                sortino = 0.0
        else:
            sortino = float('inf') if avg_daily_return > 0 else 0.0

        metrics["sortino_ratio"] = round(sortino, 2) if sortino != float('inf') else 999.99

        # Maximum Drawdown
        drawdowns = cls._calculate_drawdowns(result)
        if drawdowns:
            max_dd = max(d.drawdown_percent for d in drawdowns)
            max_dd_duration = max(d.duration_days for d in drawdowns)
        else:
            max_dd = 0.0
            max_dd_duration = 0.0

        metrics["max_drawdown_percent"] = round(max_dd, 2)
        metrics["max_drawdown_duration_days"] = round(max_dd_duration, 1)

        # Calmar Ratio (CAGR / Max Drawdown)
        cagr = cls.calculate_return_metrics(result).get("cagr_percent", 0)
        if max_dd > 0:
            calmar = abs(cagr / max_dd)
        else:
            calmar = 0.0

        metrics["calmar_ratio"] = round(calmar, 2)

        # Value at Risk (95% confidence)
        sorted_returns = sorted(daily_returns)
        var_index = int(len(sorted_returns) * 0.05)
        var_95 = sorted_returns[var_index] if var_index < len(sorted_returns) else 0.0

        metrics["var_95_percent"] = round(var_95, 2)

        # Expected Shortfall (CVaR) - average of returns below VaR
        if var_index > 0:
            cvar = sum(sorted_returns[:var_index]) / var_index
        else:
            cvar = var_95

        metrics["cvar_95_percent"] = round(cvar, 2)

        return metrics

    @classmethod
    def calculate_trade_metrics(cls, result: BacktestResult) -> dict[str, Any]:
        """Calculate trade-related metrics."""
        closed_trades = [t for t in result.trades if t.is_closed]

        if not closed_trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate_percent": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "avg_trade_pnl": 0.0,
                "avg_hold_time_seconds": 0.0,
                "expectancy": 0.0,
            }

        winning = [t for t in closed_trades if t.pnl > 0]
        losing = [t for t in closed_trades if t.pnl < 0]
        breakeven = [t for t in closed_trades if t.pnl == 0]

        total = len(closed_trades)
        win_count = len(winning)
        loss_count = len(losing)

        # Win rate
        win_rate = (win_count / total) * 100 if total > 0 else 0.0

        # P&L metrics
        total_profit = sum(t.pnl for t in winning)
        total_loss = abs(sum(t.pnl for t in losing))

        avg_win = total_profit / win_count if win_count > 0 else 0.0
        avg_loss = total_loss / loss_count if loss_count > 0 else 0.0

        largest_win = max((t.pnl for t in winning), default=0.0)
        largest_loss = min((t.pnl for t in losing), default=0.0)

        # Profit factor
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        if profit_factor == float('inf'):
            profit_factor = 999.99

        # Average trade
        avg_pnl = sum(t.pnl for t in closed_trades) / total

        # Hold time
        avg_hold_time = sum(t.hold_duration_seconds for t in closed_trades) / total

        # Expectancy (expected value per trade)
        win_rate_decimal = win_rate / 100
        if win_rate_decimal > 0 and win_rate_decimal < 1:
            expectancy = (win_rate_decimal * avg_win) - ((1 - win_rate_decimal) * avg_loss)
        else:
            expectancy = avg_pnl

        # Trade distribution by exit reason
        exit_reasons = defaultdict(int)
        for trade in closed_trades:
            exit_reasons[trade.signal_reason.value] += 1

        return {
            "total_trades": total,
            "winning_trades": win_count,
            "losing_trades": loss_count,
            "breakeven_trades": len(breakeven),
            "win_rate_percent": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "largest_win": round(largest_win, 2),
            "largest_loss": round(largest_loss, 2),
            "avg_trade_pnl": round(avg_pnl, 2),
            "avg_hold_time_seconds": round(avg_hold_time, 1),
            "avg_hold_time_minutes": round(avg_hold_time / 60, 1),
            "expectancy": round(expectancy, 2),
            "exit_reasons": dict(exit_reasons),
        }

    @classmethod
    def calculate_time_metrics(cls, result: BacktestResult) -> dict[str, Any]:
        """Calculate time-based metrics (monthly, weekly returns)."""
        if not result.equity_curve:
            return {}

        # Group equity by month
        monthly_equity = defaultdict(list)
        for point in result.equity_curve:
            key = point.timestamp.strftime("%Y-%m")
            monthly_equity[key].append(point.equity)

        # Calculate monthly returns
        months = sorted(monthly_equity.keys())
        monthly_returns = []
        prev_equity = result.config.initial_capital

        for month in months:
            end_equity = monthly_equity[month][-1]
            monthly_return = (end_equity - prev_equity) / prev_equity * 100
            monthly_returns.append({
                "month": month,
                "return_percent": round(monthly_return, 2),
                "end_equity": round(end_equity, 2),
            })
            prev_equity = end_equity

        # Best and worst months
        if monthly_returns:
            best_month = max(monthly_returns, key=lambda x: x["return_percent"])
            worst_month = min(monthly_returns, key=lambda x: x["return_percent"])
            positive_months = len([m for m in monthly_returns if m["return_percent"] > 0])
        else:
            best_month = worst_month = None
            positive_months = 0

        # Trades per day analysis
        trades_by_day = defaultdict(int)
        for trade in result.trades:
            day_key = trade.timestamp.strftime("%Y-%m-%d")
            trades_by_day[day_key] += 1

        avg_trades_per_day = len(result.trades) / max(len(trades_by_day), 1)

        return {
            "monthly_returns": monthly_returns,
            "best_month": best_month,
            "worst_month": worst_month,
            "positive_months": positive_months,
            "total_months": len(monthly_returns),
            "positive_month_rate": round(positive_months / max(len(monthly_returns), 1) * 100, 1),
            "avg_trades_per_day": round(avg_trades_per_day, 2),
            "active_trading_days": len(trades_by_day),
        }

    @classmethod
    def _calculate_daily_returns(cls, result: BacktestResult) -> list[float]:
        """Calculate daily returns from equity curve."""
        if not result.equity_curve or len(result.equity_curve) < 2:
            return []

        # Group by day
        daily_equity = {}
        for point in result.equity_curve:
            day = point.timestamp.date()
            daily_equity[day] = point.equity

        # Calculate returns
        days = sorted(daily_equity.keys())
        returns = []

        for i in range(1, len(days)):
            prev_equity = daily_equity[days[i - 1]]
            curr_equity = daily_equity[days[i]]

            if prev_equity > 0:
                daily_return = (curr_equity - prev_equity) / prev_equity * 100
                returns.append(daily_return)

        return returns

    @classmethod
    def _calculate_drawdowns(cls, result: BacktestResult) -> list[DrawdownInfo]:
        """Calculate all drawdown periods."""
        if not result.equity_curve:
            return []

        drawdowns = []
        peak = result.config.initial_capital
        current_drawdown = None

        for point in result.equity_curve:
            if point.equity >= peak:
                # New peak - close any existing drawdown
                if current_drawdown:
                    current_drawdown.recovery_date = point.timestamp
                    current_drawdown.recovery_days = (
                        point.timestamp - current_drawdown.start_date
                    ).days
                    drawdowns.append(current_drawdown)
                    current_drawdown = None

                peak = point.equity
            else:
                # In drawdown
                drawdown_pct = (peak - point.equity) / peak * 100

                if current_drawdown is None:
                    # Start new drawdown
                    current_drawdown = DrawdownInfo(
                        start_date=point.timestamp,
                        end_date=None,
                        recovery_date=None,
                        peak_equity=peak,
                        trough_equity=point.equity,
                        drawdown_percent=drawdown_pct,
                        duration_days=0,
                        recovery_days=None,
                    )
                else:
                    # Update existing drawdown
                    if point.equity < current_drawdown.trough_equity:
                        current_drawdown.trough_equity = point.equity
                        current_drawdown.drawdown_percent = drawdown_pct
                        current_drawdown.end_date = point.timestamp

                    current_drawdown.duration_days = (
                        point.timestamp - current_drawdown.start_date
                    ).days

        # Handle ongoing drawdown
        if current_drawdown:
            drawdowns.append(current_drawdown)

        return drawdowns

    @classmethod
    def generate_report(cls, result: BacktestResult) -> str:
        """Generate a text report of the backtest results."""
        metrics = result.metrics if result.metrics else cls.calculate_all(result)

        report = []
        report.append("=" * 60)
        report.append("             BACKTEST PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append("")

        # Period
        report.append("PERIOD")
        report.append("-" * 40)
        report.append(f"  Start:          {result.config.start_date.strftime('%Y-%m-%d')}")
        report.append(f"  End:            {result.config.end_date.strftime('%Y-%m-%d')}")
        report.append(f"  Trading Days:   {metrics.get('trading_days', 0)}")
        report.append("")

        # Capital
        report.append("CAPITAL")
        report.append("-" * 40)
        report.append(f"  Initial:        ${result.config.initial_capital:,.2f}")
        report.append(f"  Final:          ${result.final_equity:,.2f}")
        report.append(f"  P&L:            ${metrics.get('total_pnl', 0):,.2f}")
        report.append("")

        # Returns
        report.append("RETURNS")
        report.append("-" * 40)
        report.append(f"  Total Return:   {metrics.get('total_return_percent', 0):.2f}%")
        report.append(f"  CAGR:           {metrics.get('cagr_percent', 0):.2f}%")
        report.append(f"  Annualized:     {metrics.get('annualized_return_percent', 0):.2f}%")
        report.append("")

        # Risk
        report.append("RISK METRICS")
        report.append("-" * 40)
        report.append(f"  Sharpe Ratio:   {metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"  Sortino Ratio:  {metrics.get('sortino_ratio', 0):.2f}")
        report.append(f"  Max Drawdown:   {metrics.get('max_drawdown_percent', 0):.2f}%")
        report.append(f"  Volatility:     {metrics.get('volatility_annual_percent', 0):.2f}%")
        report.append(f"  Calmar Ratio:   {metrics.get('calmar_ratio', 0):.2f}")
        report.append(f"  VaR (95%):      {metrics.get('var_95_percent', 0):.2f}%")
        report.append("")

        # Trades
        report.append("TRADE STATISTICS")
        report.append("-" * 40)
        report.append(f"  Total Trades:   {metrics.get('total_trades', 0)}")
        report.append(f"  Winning:        {metrics.get('winning_trades', 0)}")
        report.append(f"  Losing:         {metrics.get('losing_trades', 0)}")
        report.append(f"  Win Rate:       {metrics.get('win_rate_percent', 0):.1f}%")
        report.append(f"  Profit Factor:  {metrics.get('profit_factor', 0):.2f}")
        report.append(f"  Expectancy:     ${metrics.get('expectancy', 0):.2f}")
        report.append("")
        report.append(f"  Avg Win:        ${metrics.get('avg_win', 0):.2f}")
        report.append(f"  Avg Loss:       ${metrics.get('avg_loss', 0):.2f}")
        report.append(f"  Largest Win:    ${metrics.get('largest_win', 0):.2f}")
        report.append(f"  Largest Loss:   ${metrics.get('largest_loss', 0):.2f}")
        report.append("")
        report.append(f"  Avg Hold Time:  {metrics.get('avg_hold_time_minutes', 0):.1f} min")
        report.append("")

        # Exit reasons
        exit_reasons = metrics.get('exit_reasons', {})
        if exit_reasons:
            report.append("EXIT REASONS")
            report.append("-" * 40)
            for reason, count in sorted(exit_reasons.items()):
                report.append(f"  {reason:15} {count:5}")
            report.append("")

        # Monthly performance
        monthly = metrics.get('monthly_returns', [])
        if monthly and len(monthly) <= 12:
            report.append("MONTHLY RETURNS")
            report.append("-" * 40)
            for m in monthly:
                sign = "+" if m['return_percent'] >= 0 else ""
                report.append(f"  {m['month']}:      {sign}{m['return_percent']:.2f}%")
            report.append("")

        report.append(f"  Best Month:     {metrics.get('best_month', {}).get('return_percent', 0):.2f}%")
        report.append(f"  Worst Month:    {metrics.get('worst_month', {}).get('return_percent', 0):.2f}%")
        report.append(f"  Positive Rate:  {metrics.get('positive_month_rate', 0):.1f}%")
        report.append("")

        # Strategy parameters
        report.append("STRATEGY PARAMETERS")
        report.append("-" * 40)
        report.append(f"  Spike Threshold: {result.config.spike_threshold_percent}%")
        report.append(f"  Spike Window:    {result.config.spike_window_seconds}s")
        report.append(f"  Take Profit:     {result.config.take_profit_percent}%")
        report.append(f"  Stop Loss:       {result.config.stop_loss_percent}%")
        report.append(f"  Position Size:   ${result.config.position_size_usdc}")
        report.append("")

        report.append("=" * 60)

        return "\n".join(report)
