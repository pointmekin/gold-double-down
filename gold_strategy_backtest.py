"""
Gold Double-Down Strategy Backtester
Visualizes an investment strategy of buying at fixed price intervals and closing on target profit.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy import stats
from typing import Dict, List, Tuple, Optional
from enum import Enum

# Common tickers with their margin requirements (approximate)
TICKER_INFO = {
    "GC=F": {"name": "Gold Futures", "point_value": 100, "margin_multiplier": 0.10, "tick_size": 0.1},
    "SI=F": {"name": "Silver Futures", "point_value": 5000, "margin_multiplier": 0.12, "tick_size": 0.005},
    "CL=F": {"name": "Crude Oil", "point_value": 1000, "margin_multiplier": 0.15, "tick_size": 0.01},
    "ES=F": {"name": "S&P 500 Futures", "point_value": 50, "margin_multiplier": 0.10, "tick_size": 0.25},
    "NQ=F": {"name": "Nasdaq 100 Futures", "point_value": 20, "margin_multiplier": 0.10, "tick_size": 0.25},
    "ZN=F": {"name": "10-Year T-Note", "point_value": 1000, "margin_multiplier": 0.08, "tick_size": 0.015625},
    "ZC=F": {"name": "Corn Futures", "point_value": 50, "margin_multiplier": 0.12, "tick_size": 0.25},
    "GLD": {"name": "Gold ETF", "point_value": 1, "margin_multiplier": 1.0, "tick_size": 0.01},
    "SLV": {"name": "Silver ETF", "point_value": 1, "margin_multiplier": 1.0, "tick_size": 0.01},
    "TLT": {"name": "20+ Year Treasury", "point_value": 1, "margin_multiplier": 1.0, "tick_size": 0.01},
    "SPY": {"name": "S&P 500 ETF", "point_value": 1, "margin_multiplier": 1.0, "tick_size": 0.01},
    "QQQ": {"name": "Nasdaq ETF", "point_value": 1, "margin_multiplier": 1.0, "tick_size": 0.01},
}


class TakeProfitType(Enum):
    """Take profit calculation types."""
    PERCENTAGE = "percentage"
    FIXED_AMOUNT = "fixed_amount"


class PriceGapType(Enum):
    """Price gap calculation types."""
    FIXED = "fixed"
    PERCENTAGE = "percentage"


class StrategyConfig:
    """Configuration for the double-down strategy."""

    def __init__(
        self,
        ticker: str = "GC=F",
        price_gap: float = 10.0,
        price_gap_type: PriceGapType = PriceGapType.FIXED,
        price_gap_pct: float = 0.02,  # 2% price drop
        position_size: float = 1.0,
        initial_capital: float = 50000.0,
        margin_level: float = 1.0,
        target_profit_pct: float = 0.10,
        target_profit_amount: Optional[float] = None,
        take_profit_type: TakeProfitType = TakeProfitType.PERCENTAGE,
        stop_loss_pct: Optional[float] = None,
        stop_loss_enabled: bool = False,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        max_positions: int = 50,
        commission_per_contract: float = 0.0,
        commission_pct: float = 0.0,
        daily_swap_per_contract: float = 0.0,
        # Progressive scaling settings
        enable_progressive_scaling: bool = False,
        scaling_interval: int = 5,  # Every N positions, increase gap and size
        price_gap_multiplier: float = 1.5,  # Multiply gap by this amount each interval
        position_size_multiplier: float = 2.0,  # Multiply size by this amount each interval
    ):
        self.ticker = ticker
        self.price_gap = price_gap
        self.price_gap_type = price_gap_type
        self.price_gap_pct = price_gap_pct
        self.position_size = position_size
        self.initial_capital = initial_capital
        self.margin_level = margin_level  # 1.0 = no leverage, 2.0 = 2x leverage
        self.target_profit_pct = target_profit_pct
        self.target_profit_amount = target_profit_amount
        self.take_profit_type = take_profit_type
        self.stop_loss_pct = stop_loss_pct
        self.stop_loss_enabled = stop_loss_enabled
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.max_positions = max_positions
        self.commission_per_contract = commission_per_contract  # Fixed commission per contract
        self.commission_pct = commission_pct  # Percentage commission on trade value
        self.daily_swap_per_contract = daily_swap_per_contract  # Daily swap/holding cost per contract
        self.enable_progressive_scaling = enable_progressive_scaling
        self.scaling_interval = scaling_interval
        self.price_gap_multiplier = price_gap_multiplier
        self.position_size_multiplier = position_size_multiplier


class BacktestResult:
    """Container for backtest results."""

    def __init__(self):
        self.trades: List[Dict] = []
        self.equity_curve: pd.Series = None
        self.positions_history: List[Dict] = []
        self.drawdowns: pd.Series = None
        self.final_capital: float = 0
        self.peak_capital: float = 0
        self.max_drawdown: float = 0
        self.total_return: float = 0
        self.annualized_return: float = 0
        self.sharpe_ratio: float = 0
        self.sortino_ratio: float = 0
        self.win_rate: float = 0
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.avg_trade_return: float = 0
        self.profit_factor: float = 0
        self.calmar_ratio: float = 0
        self.liquidity_price: float = 0
        self.risk_of_ruin_levels: Dict[float, float] = {}
        self.total_commissions_paid: float = 0
        self.total_swap_paid: float = 0
        self.all_entries: List[Dict] = []  # All individual entry orders
        self.all_exits: List[Dict] = []  # All exit events


class DoubleDownBacktester:
    """Backtester for the double-down strategy."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.data = None
        self.ticker_info = TICKER_INFO.get(config.ticker, {
            "name": config.ticker,
            "point_value": 1,
            "margin_multiplier": 0.10,
            "tick_size": 0.01
        })

    def fetch_data(self) -> pd.DataFrame:
        """Fetch historical data from Yahoo Finance."""
        print(f"Fetching data for {self.config.ticker}...")
        ticker_obj = yf.Ticker(self.config.ticker)
        self.data = ticker_obj.history(
            start=self.config.start_date,
            end=self.config.end_date,
            interval="1d"
        )
        if self.data.empty:
            raise ValueError(f"No data found for ticker {self.config.ticker}")
        print(f"Loaded {len(self.data)} days of data")
        return self.data

    def _calculate_commission(self, price: float, contracts: float) -> float:
        """Calculate total commission for a trade."""
        # Fixed per contract
        fixed_comm = self.config.commission_per_contract * contracts

        # Percentage of trade value
        point_value = self.ticker_info["point_value"]
        trade_value = price * contracts * point_value
        pct_comm = trade_value * self.config.commission_pct

        return fixed_comm + pct_comm

    def run_backtest(self) -> BacktestResult:
        """Run the backtest simulation."""
        if self.data is None:
            self.fetch_data()

        result = BacktestResult()
        capital = self.config.initial_capital
        positions = []  # List of (entry_price, entry_date) tuples
        equity_values = []
        positions_history = []

        # Get point value for position sizing
        point_value = self.ticker_info["point_value"]

        buy_levels = set()  # Track price levels where we've bought
        last_buy_price = None
        position_entry_dates = []  # Track when each position was entered
        total_commissions = 0
        total_swaps = 0

        for i, (date, row) in enumerate(self.data.iterrows()):
            price = row["Close"]

            # Calculate current P&L for open positions
            if positions:
                avg_entry = sum(p[0] for p in positions) / len(positions)
                unrealized_pnl = (price - avg_entry) * len(positions) * point_value * self.config.position_size
            else:
                avg_entry = None
                unrealized_pnl = 0

            # Calculate daily swap costs for open positions
            daily_swap = 0
            if positions:
                daily_swap = len(positions) * self.config.daily_swap_per_contract * self.config.position_size
                total_swaps += daily_swap

            # Calculate total equity
            equity = capital + unrealized_pnl - total_commissions - total_swaps

            # Check for buying opportunity (price dropped by price_gap)
            if len(positions) < self.config.max_positions:
                # Determine current price gap and position size based on progressive scaling
                current_position_count = len(positions)
                current_price_gap = self.config.price_gap
                current_position_size = self.config.position_size
                current_price_gap_pct = self.config.price_gap_pct

                if self.config.enable_progressive_scaling and current_position_count > 0:
                    # Calculate which tier we're in (0-indexed)
                    tier = (current_position_count) // self.config.scaling_interval
                    # Apply multipliers
                    if self.config.price_gap_type == PriceGapType.FIXED:
                        current_price_gap = self.config.price_gap * (self.config.price_gap_multiplier ** tier)
                    else:
                        current_price_gap_pct = self.config.price_gap_pct * (self.config.price_gap_multiplier ** tier)
                    current_position_size = self.config.position_size * (self.config.position_size_multiplier ** tier)

                if positions:
                    # Check if we've dropped below any existing buy level by current_price_gap
                    lowest_buy = min(p[0] for p in positions)

                    # Calculate required drop based on gap type
                    should_buy = False
                    if self.config.price_gap_type == PriceGapType.FIXED:
                        should_buy = price <= lowest_buy - current_price_gap
                    else:
                        # Percentage-based: buy if price dropped by price_gap_pct from lowest_buy
                        should_buy = price <= lowest_buy * (1 - current_price_gap_pct)

                    if should_buy:
                        # Calculate buy price
                        if self.config.price_gap_type == PriceGapType.FIXED:
                            buy_price = (price // current_price_gap) * current_price_gap
                        else:
                            buy_price = price  # Buy at current price when percentage-based

                        if buy_price not in buy_levels:
                            # Calculate commission for opening
                            open_comm = self._calculate_commission(buy_price, current_position_size)
                            total_commissions += open_comm

                            positions.append((buy_price, date, current_position_size))
                            position_entry_dates.append(date)
                            buy_levels.add(buy_price)
                            last_buy_price = buy_price

                            # Track entry for visualization
                            result.all_entries.append({
                                "date": date,
                                "price": buy_price,
                                "position_count": len(positions),
                                "position_size": current_position_size,
                                "price_gap": current_price_gap if self.config.price_gap_type == PriceGapType.FIXED else current_price_gap_pct
                            })
                else:
                    # First position
                    open_comm = self._calculate_commission(price, current_position_size)
                    total_commissions += open_comm

                    positions.append((price, date, current_position_size))
                    position_entry_dates.append(date)
                    buy_levels.add(price)
                    last_buy_price = price

                    # Track entry for visualization
                    result.all_entries.append({
                        "date": date,
                        "price": price,
                        "position_count": 1,
                        "position_size": current_position_size,
                        "price_gap": current_price_gap if self.config.price_gap_type == PriceGapType.FIXED else current_price_gap_pct
                    })

            # Calculate total unrealized P&L percentage
            if positions:
                avg_entry = sum(p[0] for p in positions) / len(positions)
                # Sum up P&L using individual position sizes
                unrealized_pnl = sum((price - p[0]) * p[2] * point_value for p in positions)
                pnl_pct = (price - avg_entry) / avg_entry
            else:
                pnl_pct = 0
                unrealized_pnl = 0

            # Check for target profit (close all positions)
            close_positions = False
            close_reason = ""

            if positions:
                if self.config.take_profit_type == TakeProfitType.PERCENTAGE:
                    if pnl_pct >= self.config.target_profit_pct:
                        close_positions = True
                        close_reason = "profit_target_pct"
                elif self.config.take_profit_type == TakeProfitType.FIXED_AMOUNT:
                    if self.config.target_profit_amount and unrealized_pnl >= self.config.target_profit_amount:
                        close_positions = True
                        close_reason = "profit_target_fixed"

            if close_positions:
                avg_entry = sum(p[0] for p in positions) / len(positions)
                realized_pnl = unrealized_pnl

                # Calculate total position size for commission
                total_position_size = sum(p[2] for p in positions)
                close_comm = self._calculate_commission(price, total_position_size)
                total_commissions += close_comm

                # Subtract closing commission from realized P&L
                realized_pnl_after_comm = realized_pnl - close_comm

                trade = {
                    "entry_date": positions[0][1] if positions else date,
                    "exit_date": date,
                    "entry_prices": [p[0] for p in positions],
                    "exit_price": price,
                    "num_contracts": len(positions),
                    "total_position_size": total_position_size,
                    "pnl": realized_pnl_after_comm,
                    "pnl_before_comm": realized_pnl,
                    "commission": close_comm,
                    "swap_paid": sum(daily_swap for _ in positions),  # Approximate
                    "pnl_pct": pnl_pct,
                    "type": close_reason
                }
                result.trades.append(trade)

                # Track exit for visualization
                result.all_exits.append({
                    "date": date,
                    "price": price,
                    "position_count": len(positions),
                    "total_position_size": total_position_size,
                    "pnl": realized_pnl_after_comm,
                    "type": close_reason
                })

                capital += realized_pnl_after_comm
                positions.clear()
                position_entry_dates.clear()
                buy_levels.clear()

            # Check for stop loss
            if positions and self.config.stop_loss_enabled and self.config.stop_loss_pct:
                avg_entry = sum(p[0] for p in positions) / len(positions)
                if (price - avg_entry) / avg_entry <= -self.config.stop_loss_pct:
                    # Calculate P&L using individual position sizes
                    realized_pnl = sum((price - p[0]) * p[2] * point_value for p in positions)

                    # Calculate total position size for commission
                    total_position_size = sum(p[2] for p in positions)
                    close_comm = self._calculate_commission(price, total_position_size)
                    total_commissions += close_comm

                    realized_pnl_after_comm = realized_pnl - close_comm

                    trade = {
                        "entry_date": positions[0][1],
                        "exit_date": date,
                        "entry_prices": [p[0] for p in positions],
                        "exit_price": price,
                        "num_contracts": len(positions),
                        "total_position_size": total_position_size,
                        "pnl": realized_pnl_after_comm,
                        "pnl_before_comm": realized_pnl,
                        "commission": close_comm,
                        "swap_paid": 0,
                        "pnl_pct": (price - avg_entry) / avg_entry,
                        "type": "stop_loss"
                    }
                    result.trades.append(trade)

                    # Track exit for visualization
                    result.all_exits.append({
                        "date": date,
                        "price": price,
                        "position_count": len(positions),
                        "total_position_size": total_position_size,
                        "pnl": realized_pnl_after_comm,
                        "type": "stop_loss"
                    })

                    capital += realized_pnl_after_comm
                    positions.clear()
                    position_entry_dates.clear()
                    buy_levels.clear()

            # Record equity
            equity_values.append(capital + unrealized_pnl - total_commissions - total_swaps)
            positions_history.append({
                "date": date,
                "price": price,
                "num_positions": len(positions),
                "avg_entry": avg_entry,
                "equity": capital + unrealized_pnl - total_commissions - total_swaps,
                "unrealized_pnl": unrealized_pnl,
                "total_commissions": total_commissions,
                "total_swaps": total_swaps
            })

        # Calculate final statistics
        result.equity_curve = pd.Series(equity_values, index=self.data.index)
        result.positions_history = positions_history
        result.final_capital = equity_values[-1]
        result.peak_capital = max(equity_values)

        # Calculate max drawdown using running peak (peak so far, not final peak)
        running_peak = equity_values[0]
        drawdowns = []
        for e in equity_values:
            if e > running_peak:
                running_peak = e
            drawdown = (e - running_peak) / running_peak if running_peak > 0 else 0
            drawdowns.append(drawdown)
        result.max_drawdown = min(drawdowns)

        result.total_return = (result.final_capital - self.config.initial_capital) / self.config.initial_capital

        # Calculate annualized return
        days = len(self.data)
        years = days / 252
        result.annualized_return = (1 + result.total_return) ** (1 / years) - 1

        # Calculate Sharpe Ratio
        returns = result.equity_curve.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            result.sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            # Sortino (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                result.sortino_ratio = (returns.mean() * 252) / (downside_returns.std() * np.sqrt(252))

        # Trade statistics
        result.total_trades = len(result.trades)
        if result.total_trades > 0:
            result.winning_trades = sum(1 for t in result.trades if t["pnl"] > 0)
            result.win_rate = result.winning_trades / result.total_trades
            result.avg_trade_return = np.mean([t["pnl_pct"] for t in result.trades])

            gross_profit = sum(t["pnl"] for t in result.trades if t["pnl"] > 0)
            gross_loss = abs(sum(t["pnl"] for t in result.trades if t["pnl"] < 0))
            result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Calmar Ratio
        if result.max_drawdown != 0:
            result.calmar_ratio = result.annualized_return / abs(result.max_drawdown)

        # Track costs
        result.total_commissions_paid = total_commissions
        result.total_swap_paid = total_swaps

        # Calculate liquidation price and risk of ruin
        result.liquidity_price, result.risk_of_ruin_levels = self._calculate_risk_metrics()

        return result

    def _calculate_risk_metrics(self) -> Tuple[float, Dict[float, float]]:
        """Calculate liquidation price and risk of ruin scenarios."""
        point_value = self.ticker_info["point_value"]

        # Calculate at what price we'd lose a significant portion of capital
        # Assuming we have max_positions at average entry
        current_price = self.data["Close"].iloc[-1]

        # Simulate different price drop scenarios
        risk_levels = {
            0.20: 0,  # 20% portfolio loss
            0.50: 0,  # 50% portfolio loss
            0.80: 0,  # 80% portfolio loss
            1.00: 0,  # 100% portfolio loss (liquidation)
        }

        # For max positions scenario
        max_contracts = self.config.max_positions
        avg_entry = current_price * 0.95  # Assume we entered 5% above current

        for loss_pct in risk_levels.keys():
            loss_amount = self.config.initial_capital * loss_pct
            # Price move needed to cause this loss
            # loss = (entry - price) * contracts * point_value * size
            price_drop_needed = loss_amount / (max_contracts * point_value * self.config.position_size)
            risk_levels[loss_pct] = current_price - price_drop_needed

        return risk_levels[1.00], risk_levels


def create_strategy_visualization(
    config: StrategyConfig,
    result: BacktestResult
) -> go.Figure:
    """Create comprehensive visualization of the strategy."""

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            "Price & Buy Levels",
            "Equity Curve",
            "Drawdowns",
            "Position Count Over Time",
            "P&L Distribution",
            "Trade Returns",
            "Risk of Ruin Analysis",
            "Strategy Metrics"
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": True}],
            [{"type": "histogram"}, {"type": "scatter"}],
            [{"secondary_y": False}, {"type": "table"}]
        ],
        row_heights=[0.3, 0.25, 0.25, 0.2],
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )

    # Extract data
    dates = result.equity_curve.index
    prices = [p["price"] for p in result.positions_history]
    equity = result.equity_curve.values

    # 1. Price & Buy Levels
    fig.add_trace(
        go.Scatter(x=dates, y=prices, mode="lines", name="Price", line=dict(color="gray", width=1)),
        row=1, col=1
    )

    # Add entry markers (green triangles pointing up)
    if result.all_entries:
        entry_dates = [e["date"] for e in result.all_entries]
        entry_prices = [e["price"] for e in result.all_entries]
        entry_sizes = [e["position_count"] * 10 for e in result.all_entries]  # Size based on position count

        fig.add_trace(
            go.Scatter(
                x=entry_dates,
                y=entry_prices,
                mode="markers",
                name="Entry",
                marker=dict(
                    symbol="triangle-up",
                    size=10,
                    color="green",
                    line=dict(color="darkgreen", width=1)
                ),
                text=[f"Pos #{e['position_count']}" for e in result.all_entries],
                hovertemplate="Entry: %{x}<br>Price: $%{y:.2f}<br>%{text}<extra></extra>"
            ),
            row=1, col=1
        )

    # Add exit markers with different colors based on profit/loss
    if result.all_exits:
        exit_dates = [e["date"] for e in result.all_exits]
        exit_prices = [e["price"] for e in result.all_exits]
        exit_colors = ["green" if e.get("pnl", 0) > 0 else "red" for e in result.all_exits]
        exit_symbols = ["circle" for _ in result.all_exits]

        fig.add_trace(
            go.Scatter(
                x=exit_dates,
                y=exit_prices,
                mode="markers+text",
                name="Exit",
                marker=dict(
                    symbol=exit_symbols,
                    size=12,
                    color=exit_colors,
                    line=dict(width=1, color="white")
                ),
                text=[f"${e.get('pnl', 0):.0f}" for e in result.all_exits],
                textposition="top center",
                textfont=dict(size=8),
                hovertemplate="Exit: %{x}<br>Price: $%{y:.2f}<br>P&L: %{text}<extra></extra>"
            ),
            row=1, col=1
        )

    # Add buy level horizontal lines (less prominent now)
    for trade in result.trades:
        for entry_price in trade["entry_prices"]:
            fig.add_hline(
                y=entry_price,
                line=dict(color="lightgreen", width=0.5, dash="dot"),
                row=1, col=1
            )

    # 2. Equity Curve
    fig.add_trace(
        go.Scatter(
            x=dates, y=equity,
            mode="lines",
            name="Equity",
            line=dict(color="blue", width=2)
        ),
        row=1, col=2
    )
    fig.add_hline(
        y=config.initial_capital,
        line=dict(color="gray", width=1, dash="dash"),
        row=1, col=2
    )

    # 3. Drawdowns
    drawdowns = []
    peak = equity[0]
    for e in equity:
        if e > peak:
            peak = e
        drawdowns.append((e - peak) / peak * 100)

    fig.add_trace(
        go.Scatter(
            x=dates, y=drawdowns,
            mode="lines",
            name="Drawdown %",
            fill="tozeroy",
            line=dict(color="red", width=1)
        ),
        row=2, col=1
    )

    # 4. Position Count
    pos_counts = [p["num_positions"] for p in result.positions_history]
    fig.add_trace(
        go.Scatter(
            x=dates, y=pos_counts,
            mode="lines",
            name="Positions",
            line=dict(color="purple", width=2),
            fill="tozeroy"
        ),
        row=2, col=2
    )

    # 5. P&L Distribution
    if result.trades:
        trade_pnls = [t["pnl"] for t in result.trades]
        colors = ["green" if p > 0 else "red" for p in trade_pnls]

        fig.add_trace(
            go.Histogram(
                x=trade_pnls,
                nbinsx=20,
                marker_color="steelblue",
                name="P&L Distribution"
            ),
            row=3, col=1
        )

        # 6. Trade Returns
        trade_returns = [t["pnl_pct"] * 100 for t in result.trades]
        trade_nums = list(range(1, len(trade_returns) + 1))

        fig.add_trace(
            go.Bar(
                x=trade_nums,
                y=trade_returns,
                marker_color=colors,
                name="Trade Return %"
            ),
            row=3, col=2
        )

    # 7. Risk of Ruin Analysis
    loss_levels = list(result.risk_of_ruin_levels.keys())
    price_levels = list(result.risk_of_ruin_levels.values())

    fig.add_trace(
        go.Bar(
            x=[f"{int(l * 100)}%" for l in loss_levels],
            y=price_levels,
            marker_color="orange",
            name="Price for Loss Level"
        ),
        row=4, col=1
    )

    # Build metrics data with costs
    take_profit_label = f"{config.target_profit_pct * 100:.1f}%" if config.take_profit_type == TakeProfitType.PERCENTAGE else f"${config.target_profit_amount:.0f}"
    price_gap_label = f"${config.price_gap}" if config.price_gap_type == PriceGapType.FIXED else f"{config.price_gap_pct * 100:.1f}%"

    metrics_data = [
        ["Initial Capital", f"${config.initial_capital:,.0f}"],
        ["Final Capital", f"${result.final_capital:,.0f}"],
        ["Total Return", f"{result.total_return * 100:.2f}%"],
        ["Annualized Return", f"{result.annualized_return * 100:.2f}%"],
        ["Max Drawdown", f"{result.max_drawdown * 100:.2f}%"],
        ["Sharpe Ratio", f"{result.sharpe_ratio:.2f}"],
        ["Sortino Ratio", f"{result.sortino_ratio:.2f}"],
        ["Calmar Ratio", f"{result.calmar_ratio:.2f}"],
        ["Total Trades", str(result.total_trades)],
        ["Win Rate", f"{result.win_rate * 100:.1f}%"],
        ["Winning Trades", str(result.winning_trades)],
        ["Avg Trade Return", f"{result.avg_trade_return * 100:.2f}%"],
        ["Profit Factor", f"{result.profit_factor:.2f}"],
        ["Total Commissions", f"${result.total_commissions_paid:,.2f}"],
        ["Total Swap/Funding", f"${result.total_swap_paid:,.2f}"],
        ["Take Profit Target", take_profit_label],
    ]

    fig.add_trace(
        go.Table(
            header=dict(values=["Metric", "Value"], fill_color="lightgray"),
            cells=dict(values=[[m[0] for m in metrics_data], [m[1] for m in metrics_data]])
        ),
        row=4, col=2
    )

    fig.update_layout(
        height=1200,
        title_text=f"Strategy Backtest: {config.ticker} | Price Gap: {price_gap_label} | "
                   f"Position Size: {config.position_size} | Margin: {config.margin_level}x | "
                   f"TP: {take_profit_label} | Entries: {len(result.all_entries)} | Exits: {len(result.all_exits)}",
        title_font_size=14,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white"
    )

    return fig


def run_analysis(ticker: str = "GC=F",
                 price_gap: float = 10.0,
                 price_gap_type: PriceGapType = PriceGapType.FIXED,
                 price_gap_pct: float = 0.02,
                 position_size: float = 1.0,
                 initial_capital: float = 50000,
                 margin_level: float = 1.0,
                 target_profit_pct: float = 0.10,
                 target_profit_amount: Optional[float] = None,
                 take_profit_type: TakeProfitType = TakeProfitType.PERCENTAGE,
                 stop_loss_enabled: bool = False,
                 stop_loss_pct: float = 0.20,
                 start_date: str = "2020-01-01",
                 max_positions: int = 50,
                 commission_per_contract: float = 0.0,
                 commission_pct: float = 0.0,
                 daily_swap_per_contract: float = 0.0,
                 enable_progressive_scaling: bool = False,
                 scaling_interval: int = 5,
                 price_gap_multiplier: float = 1.5,
                 position_size_multiplier: float = 2.0) -> Tuple[go.Figure, BacktestResult]:
    """
    Run strategy analysis with given parameters.

    Returns:
        Tuple of (figure, backtest_result)
    """
    config = StrategyConfig(
        ticker=ticker,
        price_gap=price_gap,
        price_gap_type=price_gap_type,
        price_gap_pct=price_gap_pct,
        position_size=position_size,
        initial_capital=initial_capital,
        margin_level=margin_level,
        target_profit_pct=target_profit_pct,
        target_profit_amount=target_profit_amount,
        take_profit_type=take_profit_type,
        stop_loss_enabled=stop_loss_enabled,
        stop_loss_pct=stop_loss_pct,
        start_date=start_date,
        max_positions=max_positions,
        commission_per_contract=commission_per_contract,
        commission_pct=commission_pct,
        daily_swap_per_contract=daily_swap_per_contract,
        enable_progressive_scaling=enable_progressive_scaling,
        scaling_interval=scaling_interval,
        price_gap_multiplier=price_gap_multiplier,
        position_size_multiplier=position_size_multiplier
    )

    backtester = DoubleDownBacktester(config)
    result = backtester.run_backtest()
    fig = create_strategy_visualization(config, result)

    return fig, result


def print_summary(result: BacktestResult, config: StrategyConfig):
    """Print summary statistics to console."""
    take_profit_label = f"{config.target_profit_pct * 100:.1f}%" if config.take_profit_type == TakeProfitType.PERCENTAGE else f"${config.target_profit_amount:.0f}"

    print("\n" + "=" * 60)
    print(f"STRATEGY BACKTEST SUMMARY - {config.ticker}")
    print("=" * 60)
    print(f"Period: {config.start_date} to {config.end_date}")
    print(f"Initial Capital: ${config.initial_capital:,.0f}")
    print("-" * 60)
    print(f"Final Capital:     ${result.final_capital:,.0f}")
    print(f"Total Return:      {result.total_return * 100:.2f}%")
    print(f"Annualized Return: {result.annualized_return * 100:.2f}%")
    print(f"Max Drawdown:      {result.max_drawdown * 100:.2f}%")
    print("-" * 60)
    print(f"Sharpe Ratio:      {result.sharpe_ratio:.2f}")
    print(f"Sortino Ratio:     {result.sortino_ratio:.2f}")
    print(f"Calmar Ratio:      {result.calmar_ratio:.2f}")
    print("-" * 60)
    print(f"Total Trades:      {result.total_trades}")
    print(f"Win Rate:          {result.win_rate * 100:.1f}%")
    print(f"Winning Trades:    {result.winning_trades}")
    print(f"Avg Trade Return:  {result.avg_trade_return * 100:.2f}%")
    print(f"Profit Factor:     {result.profit_factor:.2f}")
    print("-" * 60)
    print(f"Total Commissions: ${result.total_commissions_paid:,.2f}")
    print(f"Total Swap/Funding: ${result.total_swap_paid:,.2f}")
    print(f"Take Profit:       {take_profit_label}")
    print("-" * 60)
    print("RISK OF RUIN ANALYSIS")
    print("-" * 60)
    for loss_pct, price in result.risk_of_ruin_levels.items():
        print(f"Price for {int(loss_pct * 100)}% portfolio loss: ${price:.2f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Example usage
    fig, result = run_analysis(
        ticker="GC=F",
        price_gap=10.0,
        position_size=0.01,
        initial_capital=50000,
        margin_level=1.0,
        target_profit_pct=0.10,
        take_profit_type=TakeProfitType.PERCENTAGE,
        stop_loss_enabled=False,
        start_date="2020-01-01",
        commission_per_contract=2.5,
        daily_swap_per_contract=0.5
    )

    print_summary(result, StrategyConfig())
    fig.show()
