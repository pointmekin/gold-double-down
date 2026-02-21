# Gold Double-Down Strategy Backtester

An interactive visualization tool for analyzing a "double-down" investment strategy where you:
- Buy at fixed price intervals as price drops
- Close all positions when a target profit percentage is reached

## Features

- **Configurable Parameters**:
  - Price gap between buy orders
  - Position size (number of contracts)
  - Initial capital
  - Margin/leverage level
  - Target profit percentage
  - Optional stop-loss

- **Multiple Assets**: Supports gold futures (GC=F), silver (SI=F), crude oil (CL=F), S&P 500 (ES=F), and ETFs like GLD, SLV, SPY, QQQ

- **Backtest Metrics**:
  - Total and annualized returns
  - Sharpe, Sortino, and Calmar ratios
  - Win rate and profit factor
  - Maximum drawdown

- **Risk Analysis**:
  - Liquidation price scenarios
  - Risk of ruin analysis (price levels for 20%, 50%, 80%, 100% portfolio loss)

## Installation

```bash
# Create virtual environment with uv
uv venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install dependencies
uv pip install -r requirements.txt
```

## Usage

### Interactive Dashboard

```bash
# Activate virtual environment first
source .venv/bin/activate

# Run on port 8501 (default) or specify another port
streamlit run app.py --server.port 8501
```

### Python Script

```python
from gold_strategy_backtest import run_analysis, print_summary

# Run backtest
fig, result = run_analysis(
    ticker="GC=F",           # Gold futures
    price_gap=10.0,          # Buy every $10 drop
    position_size=1.0,       # 1 contract per buy
    initial_capital=50000,   # Starting capital
    margin_level=1.0,        # No leverage
    target_profit_pct=0.10,  # 10% profit target
    stop_loss_enabled=False, # No stop loss
    start_date="2020-01-01", # Backtest start
    max_positions=50         # Max buy orders
)

# Print summary
print_summary(result, StrategyConfig())

# Show visualization
fig.show()
```

## Strategy Description

The strategy works as follows:

1. **Entry**: Place a buy order when price drops by `price_gap` from the last buy
2. **Accumulation**: Continue buying at each `price_gap` interval (up to `max_positions`)
3. **Exit**: Close all positions when unrealized P&L reaches `target_profit_pct`
4. **Stop Loss** (optional): Close all positions if loss reaches `stop_loss_pct`

## Supported Tickers

| Ticker | Name | Point Value |
|--------|------|-------------|
| GC=F   | Gold Futures | $100/point |
| SI=F   | Silver Futures | $5000/point |
| CL=F   | Crude Oil | $1000/point |
| ES=F   | S&P 500 Futures | $50/point |
| NQ=F   | Nasdaq 100 Futures | $20/point |
| GLD    | Gold ETF | $1/point |
| SLV    | Silver ETF | $1/point |
| SPY    | S&P 500 ETF | $1/point |
| QQQ    | Nasdaq ETF | $1/point |

## Disclaimer

This tool is for educational purposes only. Past performance does not guarantee future results. Futures trading involves substantial risk of loss.
