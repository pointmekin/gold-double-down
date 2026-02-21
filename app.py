"""
Interactive Dashboard for Gold Double-Down Strategy
Run with: streamlit run app.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gold_strategy_backtest import (
    BacktestResult, DoubleDownBacktester, PriceGapType, StrategyConfig,
    TakeProfitType, TICKER_INFO, create_strategy_visualization,
)

st.set_page_config(
    page_title="Double-Down Strategy Backtester",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .positive { color: #00cc00; }
    .negative { color: #ff4d4d; }
</style>
""", unsafe_allow_html=True)

# File-based persistence
SAVED_BACKTESTS_FILE = Path(__file__).parent / "saved_backtests.json"


def load_saved_results_from_file():
    """Load saved results from JSON file."""
    if SAVED_BACKTESTS_FILE.exists():
        try:
            with open(SAVED_BACKTESTS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            return {}
    return {}


def save_results_to_file(data: dict):
    """Save results to JSON file."""
    try:
        with open(SAVED_BACKTESTS_FILE, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        st.error(f"Error saving to file: {e}")


def save_result(name: str, config, result):
    """Save a backtest result to session state and file."""
    if "saved_results" not in st.session_state:
        st.session_state.saved_results = {}

    # Serialize the result
    result_data = {
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "ticker": config.ticker,
            "price_gap": config.price_gap,
            "price_gap_type": config.price_gap_type.value,
            "price_gap_pct": config.price_gap_pct,
            "position_size": config.position_size,
            "initial_capital": config.initial_capital,
            "margin_level": config.margin_level,
            "target_profit_pct": config.target_profit_pct,
            "target_profit_amount": config.target_profit_amount,
            "take_profit_type": config.take_profit_type.value,
            "stop_loss_enabled": config.stop_loss_enabled,
            "stop_loss_pct": config.stop_loss_pct,
            "start_date": config.start_date,
            "max_positions": config.max_positions,
            "commission_per_contract": config.commission_per_contract,
            "commission_pct": config.commission_pct,
            "daily_swap_per_contract": config.daily_swap_per_contract,
            "enable_progressive_scaling": config.enable_progressive_scaling,
            "scaling_interval": config.scaling_interval,
            "price_gap_multiplier": config.price_gap_multiplier,
            "position_size_multiplier": config.position_size_multiplier,
        },
        "result": {
            "final_capital": result.final_capital,
            "total_return": result.total_return,
            "annualized_return": result.annualized_return,
            "max_drawdown": result.max_drawdown,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "calmar_ratio": result.calmar_ratio,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "winning_trades": result.winning_trades,
            "avg_trade_return": result.avg_trade_return,
            "profit_factor": result.profit_factor,
            "total_commissions_paid": result.total_commissions_paid,
            "total_swap_paid": result.total_swap_paid,
            "trades": result.trades,
            "equity_curve": {str(k): v for k, v in result.equity_curve.to_dict().items()} if result.equity_curve is not None else None,
            "positions_history": result.positions_history,
            "all_entries": result.all_entries,
            "all_exits": result.all_exits,
            "risk_of_ruin_levels": result.risk_of_ruin_levels,
        }
    }

    st.session_state.saved_results[name] = result_data
    # Persist to file
    save_results_to_file(st.session_state.saved_results)


def load_saved_results():
    """Load saved results from file into session state."""
    if "saved_results" not in st.session_state:
        st.session_state.saved_results = load_saved_results_from_file()
    return st.session_state.saved_results


def delete_saved_result(name: str):
    """Delete a saved result from session state and file."""
    if "saved_results" in st.session_state and name in st.session_state.saved_results:
        del st.session_state.saved_results[name]
        save_results_to_file(st.session_state.saved_results)


def load_result_from_dict(data: dict) -> tuple:
    """Load a result from saved dict and return config and result objects."""
    # Reconstruct config
    config = StrategyConfig(
        ticker=data["config"]["ticker"],
        price_gap=data["config"]["price_gap"],
        price_gap_type=PriceGapType(data["config"]["price_gap_type"]),
        price_gap_pct=data["config"].get("price_gap_pct", 0.02),
        position_size=data["config"]["position_size"],
        initial_capital=data["config"]["initial_capital"],
        margin_level=data["config"]["margin_level"],
        target_profit_pct=data["config"]["target_profit_pct"],
        target_profit_amount=data["config"]["target_profit_amount"],
        take_profit_type=TakeProfitType(data["config"]["take_profit_type"]),
        stop_loss_enabled=data["config"]["stop_loss_enabled"],
        stop_loss_pct=data["config"]["stop_loss_pct"],
        start_date=data["config"]["start_date"],
        max_positions=data["config"]["max_positions"],
        commission_per_contract=data["config"]["commission_per_contract"],
        commission_pct=data["config"]["commission_pct"],
        daily_swap_per_contract=data["config"]["daily_swap_per_contract"],
        enable_progressive_scaling=data["config"].get("enable_progressive_scaling", False),
        scaling_interval=data["config"].get("scaling_interval", 5),
        price_gap_multiplier=data["config"].get("price_gap_multiplier", 1.5),
        position_size_multiplier=data["config"].get("position_size_multiplier", 2.0),
    )

    # Reconstruct result
    result = BacktestResult()
    result.final_capital = data["result"]["final_capital"]
    result.total_return = data["result"]["total_return"]
    result.annualized_return = data["result"]["annualized_return"]
    result.max_drawdown = data["result"]["max_drawdown"]
    result.sharpe_ratio = data["result"]["sharpe_ratio"]
    result.sortino_ratio = data["result"]["sortino_ratio"]
    result.calmar_ratio = data["result"]["calmar_ratio"]
    result.total_trades = data["result"]["total_trades"]
    result.win_rate = data["result"]["win_rate"]
    result.winning_trades = data["result"]["winning_trades"]
    result.avg_trade_return = data["result"]["avg_trade_return"]
    result.profit_factor = data["result"]["profit_factor"]
    result.total_commissions_paid = data["result"]["total_commissions_paid"]
    result.total_swap_paid = data["result"]["total_swap_paid"]
    result.trades = data["result"]["trades"]
    result.positions_history = data["result"]["positions_history"]
    result.all_entries = data["result"]["all_entries"]
    result.all_exits = data["result"]["all_exits"]
    result.risk_of_ruin_levels = data["result"]["risk_of_ruin_levels"]

    if data["result"]["equity_curve"]:
        result.equity_curve = pd.Series(data["result"]["equity_curve"])

    return config, result


def persist_to_local_storage():
    """JavaScript code to persist results to localStorage."""
    results = st.session_state.get("saved_results", {})
    json_str = json.dumps(results, indent=2, default=str)
    return f"""
    <script>
    localStorage.setItem('gold_backtest_results', {json.dumps(json_str)});
    </script>
    """


def load_from_local_storage():
    """JavaScript code to load from localStorage into Streamlit."""
    return """
    <script>
    function loadResultsToStreamlit() {{
        const data = localStorage.getItem('gold_backtest_results');
        if (data) {{
            Streamlit.setComponentValue('load_from_storage_trigger', data);
        }}
    }}
    // Auto-load on page load
    setTimeout(loadResultsToStreamlit, 100);
    </script>
    """


# Load results trigger (hidden)
if "load_from_storage_trigger" not in st.session_state:
    st.session_state.load_from_storage_trigger = None

# Trigger to load from localStorage
load_trigger = st.text_input("Load from localStorage", key="load_input", label_visibility="collapsed")

if load_trigger and load_trigger != st.session_state.load_from_storage_trigger:
    try:
        loaded_data = json.loads(load_trigger)
        if loaded_data:
            st.session_state.saved_results = loaded_data
            st.session_state.load_from_storage_trigger = load_trigger
            st.rerun()
    except (json.JSONDecodeError, TypeError):
        pass

st.markdown('<div class="main-header">Double-Down Investment Strategy</div>', unsafe_allow_html=True)
st.markdown("""
This strategy buys more contracts at fixed price intervals (dollar-cost averaging on dips)
and closes all positions when a target profit is reached.
""")

# Sidebar Configuration
st.sidebar.header("Strategy Parameters")

# Ticker Selection
ticker_options = list(TICKER_INFO.keys())
selected_ticker = st.sidebar.selectbox(
    "Select Ticker",
    ticker_options,
    index=0,
    format_func=lambda x: f"{x} - {TICKER_INFO[x]['name']}"
)

# Get ticker info
ticker_info = TICKER_INFO[selected_ticker]
st.sidebar.info(f"**{ticker_info['name']}**\nPoint Value: ${ticker_info['point_value']}\nMargin Req: {ticker_info['margin_multiplier']*100}%")

# Price Gap Type
price_gap_type = st.sidebar.radio(
    "Price Gap Type",
    ["Fixed ($)", "Percentage (%)"],
    index=0,
    help="Choose how to calculate the price drop between buy orders"
)

if price_gap_type == "Fixed ($)":
    price_gap_type_enum = PriceGapType.FIXED
    price_gap = st.sidebar.number_input(
        "Price Drop Between Buys ($)",
        min_value=0.1,
        max_value=500.0,
        value=10.0,
        step=0.5,
        help="How much the price must drop before adding another position"
    )
    price_gap_pct = 0.02  # default fallback
else:
    price_gap_type_enum = PriceGapType.PERCENTAGE
    price_gap_pct = st.sidebar.slider(
        "Price Drop Between Buys (%)",
        min_value=0.1,
        max_value=20.0,
        value=2.0,
        step=0.1
    ) / 100
    price_gap = 10.0  # default fallback

# Position Size (min 0.01)
position_size = st.sidebar.number_input(
    "Position Size (lots/contracts)",
    min_value=0.01,
    max_value=100.0,
    value=0.01,
    step=0.01,
    help="Number of lots/contracts per buy order (min 0.01)"
)

# Initial Capital
initial_capital = st.sidebar.number_input(
    "Initial Capital ($)",
    min_value=1000,
    max_value=10000000,
    value=50000,
    step=1000
)

# Margin Level
margin_level = st.sidebar.slider(
    "Margin Level (Leverage)",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.1,
    help="1.0 = no leverage (full cash), 2.0 = 2x leverage"
)

# Take Profit Section
st.sidebar.subheader("Take Profit Settings")
take_profit_type = st.sidebar.radio(
    "Take Profit Type",
    ["Percentage", "Fixed Amount"],
    index=0,
    help="Choose how to calculate when to close positions"
)

if take_profit_type == "Percentage":
    target_profit_pct = st.sidebar.slider(
        "Target Profit (%)",
        min_value=0.5,
        max_value=100.0,
        value=10.0,
        step=0.5
    ) / 100
    target_profit_amount = None
    tp_type = TakeProfitType.PERCENTAGE
else:
    target_profit_amount = st.sidebar.number_input(
        "Target Profit Amount ($)",
        min_value=0.01,
        max_value=1000000.0,
        value=5000.0,
        step=100.0
    )
    target_profit_pct = 0.10  # default fallback
    tp_type = TakeProfitType.FIXED_AMOUNT

# Stop Loss
st.sidebar.subheader("Stop Loss Settings")
stop_loss_enabled = st.sidebar.checkbox("Enable Stop Loss", value=False)
stop_loss_pct = 0.20  # default

if stop_loss_enabled:
    stop_loss_pct = st.sidebar.slider(
        "Stop Loss (%)",
        min_value=1.0,
        max_value=100.0,
        value=20.0,
        step=1.0
    ) / 100

# Max Positions
max_positions = st.sidebar.number_input(
    "Max Positions",
    min_value=1,
    max_value=200,
    value=50,
    step=1,
    help="Maximum number of buy orders to place"
)

# Trading Costs
st.sidebar.subheader("Trading Costs")
commission_per_contract = st.sidebar.number_input(
    "Commission per Contract ($)",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=0.5,
    help="Fixed commission per contract per trade"
)

commission_pct = st.sidebar.number_input(
    "Commission Percentage (%)",
    min_value=0.0,
    max_value=5.0,
    value=0.0,
    step=0.01,
    help="Percentage commission on trade value"
) / 100

daily_swap_per_contract = st.sidebar.number_input(
    "Daily Swap/Funding per Contract ($)",
    min_value=-100.0,
    max_value=100.0,
    value=0.0,
    step=0.1,
    help="Daily holding cost (negative = you pay, positive = you receive)"
)

# Progressive Scaling Settings
st.sidebar.subheader("Progressive Scaling")
enable_progressive_scaling = st.sidebar.checkbox(
    "Enable Progressive Scaling",
    value=False,
    help="Increase position size and price gap as more positions are added"
)

if enable_progressive_scaling:
    scaling_interval = st.sidebar.number_input(
        "Scaling Interval (positions)",
        min_value=1,
        max_value=50,
        value=5,
        step=1,
        help="Every N positions, increase gap and size"
    )

    price_gap_multiplier = st.sidebar.number_input(
        "Price Gap Multiplier",
        min_value=1.0,
        max_value=10.0,
        value=1.5,
        step=0.1,
        help="Multiply price gap by this amount each interval"
    )

    position_size_multiplier = st.sidebar.number_input(
        "Position Size Multiplier",
        min_value=1.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="Multiply position size by this amount each interval"
    )
else:
    scaling_interval = 5
    price_gap_multiplier = 1.5
    position_size_multiplier = 2.0

# Date Range
st.sidebar.subheader("Backtest Period")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_year = st.number_input("Start Year", min_value=2000, max_value=2025, value=2020)
with col2:
    start_month = st.selectbox("Start Month", range(1, 13), index=0)

start_date = f"{start_year}-{start_month:02d}-01"

# Run Button
run_button = st.sidebar.button("Run Backtest", type="primary", use_container_width=True)

# Store results in session state
if "results" not in st.session_state:
    st.session_state.results = {}
    st.session_state.last_config = None

# Load saved results from file on startup
load_saved_results()

# Create a unique key for the current configuration
config_key = f"{selected_ticker}_{price_gap_type_enum.value}_{price_gap}_{price_gap_pct}_{position_size}_{target_profit_pct}_{target_profit_amount}_{tp_type.value}_{commission_per_contract}_{daily_swap_per_contract}_{enable_progressive_scaling}_{scaling_interval}_{price_gap_multiplier}_{position_size_multiplier}"

if run_button or st.session_state.last_config != config_key:
    with st.spinner("Running backtest..."):
        try:
            config = StrategyConfig(
                ticker=selected_ticker,
                price_gap=price_gap,
                price_gap_type=price_gap_type_enum,
                price_gap_pct=price_gap_pct,
                position_size=position_size,
                initial_capital=initial_capital,
                margin_level=margin_level,
                target_profit_pct=target_profit_pct,
                target_profit_amount=target_profit_amount,
                take_profit_type=tp_type,
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

            st.session_state.results[config_key] = {
                "config": config,
                "result": result
            }
            st.session_state.last_config = config_key
            st.session_state.current_key = config_key

        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")

# Display Results
if st.session_state.get("current_key") and st.session_state.current_key in st.session_state.results:
    data = st.session_state.results[st.session_state.current_key]
    config = data["config"]
    result = data["result"]

    # Key Metrics Row
    st.subheader("Performance Summary")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric(
            "Final Capital",
            f"${result.final_capital:,.0f}",
            delta=f"{result.total_return * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Total Return",
            f"{result.total_return * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Annualized Return",
            f"{result.annualized_return * 100:.2f}%"
        )

    with col4:
        st.metric(
            "Max Drawdown",
            f"{result.max_drawdown * 100:.2f}%"
        )

    with col5:
        st.metric(
            "Sharpe Ratio",
            f"{result.sharpe_ratio:.2f}"
        )

    with col6:
        st.metric(
            "Win Rate",
            f"{result.win_rate * 100:.1f}%"
        )

    # Cost Metrics Row
    st.subheader("Trading Costs")

    cost_col1, cost_col2, cost_col3 = st.columns(3)

    with cost_col1:
        st.metric("Total Commissions Paid", f"${result.total_commissions_paid:,.2f}")
    with cost_col2:
        st.metric("Total Swap/Funding Paid", f"${result.total_swap_paid:,.2f}")
    with cost_col3:
        total_costs = result.total_commissions_paid + result.total_swap_paid
        st.metric("Total Trading Costs", f"${total_costs:,.2f}")

    # Visualization
    fig = create_strategy_visualization(config, result)
    st.plotly_chart(fig, use_container_width=True)

    # Risk Analysis Section
    st.subheader("Risk Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Risk of Ruin Scenarios")
        st.markdown("*Price levels that would cause portfolio loss*")

        risk_data = []
        for loss_pct, price in result.risk_of_ruin_levels.items():
            current_price = result.positions_history[-1]['price']
            risk_data.append({
                "Portfolio Loss": f"{int(loss_pct * 100)}%",
                "Price Level": f"${price:.2f}",
                "From Current": f"{((price / current_price - 1) * 100):.1f}%"
            })

        st.dataframe(
            pd.DataFrame(risk_data),
            use_container_width=True,
            hide_index=True
        )

    with col2:
        st.markdown("### Trade Statistics")

        trade_stats = [
            ["Total Trades", result.total_trades],
            ["Winning Trades", result.winning_trades],
            ["Losing Trades", result.total_trades - result.winning_trades],
            ["Win Rate", f"{result.win_rate * 100:.1f}%"],
            ["Avg Trade Return", f"{result.avg_trade_return * 100:.2f}%"],
            ["Profit Factor", f"{result.profit_factor:.2f}"],
            ["Sortino Ratio", f"{result.sortino_ratio:.2f}"],
            ["Calmar Ratio", f"{result.calmar_ratio:.2f}"],
        ]

        for label, value in trade_stats:
            st.markdown(f"**{label}**: {value}")

    # Trade History
    if result.trades:
        st.subheader("Trade History")

        trades_df = pd.DataFrame(result.trades)
        trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"])

        display_df = trades_df[[
            "exit_date", "num_contracts", "exit_price", "pnl",
            "pnl_before_comm", "commission", "swap_paid", "pnl_pct", "type"
        ]].copy()
        display_df.columns = [
            "Exit Date", "Contracts", "Exit Price", "Net P&L ($)",
            "Gross P&L ($)", "Commission ($)", "Swap ($)", "P&L (%)", "Exit Type"
        ]
        display_df["Exit Date"] = display_df["Exit Date"].dt.strftime("%Y-%m-%d")
        display_df["Exit Price"] = display_df["Exit Price"].apply(lambda x: f"${x:.2f}")
        display_df["Net P&L ($)"] = display_df["Net P&L ($)"].apply(lambda x: f"${x:,.2f}")
        display_df["Gross P&L ($)"] = display_df["Gross P&L ($)"].apply(lambda x: f"${x:,.2f}")
        display_df["Commission ($)"] = display_df["Commission ($)"].apply(lambda x: f"${x:.2f}")
        display_df["Swap ($)"] = display_df["Swap ($)"].apply(lambda x: f"${x:.2f}")
        display_df["P&L (%)"] = display_df["P&L (%)"].apply(lambda x: f"{x * 100:.2f}%")

        # Color code exit type
        def color_exit_type(x):
            if "profit" in str(x).lower():
                return "🟢 " + x
            elif "stop" in str(x).lower():
                return "🔴 " + x
            return x

        display_df["Exit Type"] = display_df["Exit Type"].apply(color_exit_type)

        st.dataframe(display_df, use_container_width=True)

    # Save Result Section
    st.subheader("Save This Result")
    save_col1, save_col2, save_col3 = st.columns(3)

    with save_col1:
        save_name = st.text_input(
            "Result Name",
            value=f"{selected_ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            key="save_name_input"
        )

    with save_col2:
        save_clicked = st.button("💾 Save Result", type="secondary", use_container_width=True)
        if save_clicked and save_name:
            save_result(save_name, config, result)
            st.success(f"Saved as '{save_name}'")

    with save_col3:
        if st.button("🗑️ Discard", type="secondary", use_container_width=True):
            if st.session_state.current_key in st.session_state.results:
                del st.session_state.results[st.session_state.current_key]
                st.session_state.current_key = None
                st.success("Result discarded")

# Saved Results Section
st.subheader("Saved Backtests")

st.markdown("""
<div style="font-size: 0.8rem; color: gray; margin-bottom: 10px;">
    Results are saved to <code>saved_backtests.json</code> in the project directory.
    They persist across sessions. Use Export/Import JSON to backup or transfer results.
</div>
""", unsafe_allow_html=True)

# Export all button
export_col, import_col = st.columns(2)

with export_col:
    if st.session_state.get("saved_results") and st.session_state.saved_results:
        all_data = json.dumps(st.session_state.saved_results, indent=2, default=str)
        st.download_button(
            label="📦 Export All (JSON)",
            data=all_data,
            file_name=f"backtests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

with import_col:
    uploaded_file = st.file_uploader("📥 Import JSON", type=["json"], help="Import backtests from JSON file")

    if uploaded_file is not None:
        try:
            imported_data = json.load(uploaded_file)
            st.session_state.saved_results.update(imported_data)
            st.success(f"Imported {len(imported_data)} backtest(s)")
            st.rerun()
        except Exception as e:
            st.error(f"Error importing: {e}")

# Clear all button
if st.session_state.get("saved_results") and st.session_state.saved_results:
    if st.button("🗑️ Clear All Saved", type="secondary"):
        st.session_state.saved_results = {}
        st.rerun()

# List saved results
if st.session_state.get("saved_results") and st.session_state.saved_results:
    for name, data in list(st.session_state.saved_results.items()):
        col_a, col_b, col_c, col_d = st.columns([3, 1, 1, 1])

        with col_a:
            cfg = data["config"]
            res = data["result"]
            gap_str = f"${cfg['price_gap']}" if cfg.get("price_gap_type") == "fixed" else f"{cfg.get('price_gap_pct', 0)*100:.1f}%"
            summary = f"**{name}** | {cfg['ticker']} | Gap: {gap_str} | Return: {res['total_return']*100:.1f}%"
            st.markdown(summary)

        with col_b:
            if st.button("📊 Load", key=f"load_{name}", use_container_width=True):
                loaded_config, loaded_result = load_result_from_dict(data)
                new_key = f"loaded_{name}_{datetime.now().timestamp()}"
                st.session_state.results[new_key] = {
                    "config": loaded_config,
                    "result": loaded_result
                }
                st.session_state.current_key = new_key
                st.session_state.last_config = None
                st.success(f"Loaded '{name}'")
                st.rerun()

        with col_c:
            # Export single result
            single_data = json.dumps(data, indent=2, default=str)
            st.download_button(
                label="⬇️",
                data=single_data,
                file_name=f"{name}.json",
                mime="application/json",
                key=f"download_{name}",
                help="Export this result"
            )

        with col_d:
            if st.button("🗑️", key=f"delete_{name}", use_container_width=True):
                delete_saved_result(name)
                st.success(f"Deleted '{name}'")
                st.rerun()

        st.markdown("---")
else:
    st.info("No saved backtests. Run a backtest and save it to see it here.")


# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8rem;'>
    Past performance does not guarantee future results. This is for educational purposes only.
</div>
""", unsafe_allow_html=True)
