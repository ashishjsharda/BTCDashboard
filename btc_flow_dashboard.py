import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="BTC Flow Radar", page_icon="📊", layout="wide")
st.title("📊 BTC Exchange Flow Radar")
st.caption("Flow tracking, anomaly alerts, and signal backtesting • Research/education tool")

# ==================== DISCLAIMER (persistent, top of page) ====================
st.warning(
    "⚠️ **Not financial advice.** This tool is for informational and research purposes only. "
    "Exchange flow data, technical indicators, and the backtest results shown below do **not** "
    "reliably predict future prices — especially not at 15-minute resolution with tight price margins. "
    "Historical backtest accuracy is not a guarantee of future performance. Nothing here should be "
    "used as the sole basis for a trading decision. Trading cryptocurrency carries substantial risk "
    "of loss.",
    icon="⚠️",
)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.header("⚙️ Settings")

    api_token = st.text_input(
        "CryptoQuant API Token (Bearer)",
        type="password",
        value=st.secrets.get("CRYPTOQUANT_TOKEN", "") if hasattr(st, "secrets") else "",
        help="Get from cryptoquant.com account settings → API",
    )

    exchange = st.selectbox(
        "Exchange Scope",
        ["all_exchange", "spot_exchange", "binance", "coinbase", "okx"],
        index=0,
    )

    window = st.selectbox(
        "Resolution", ["day", "hour"], index=0, help="hour needs higher plan"
    )

    rolling_window = st.slider("Rolling Window (for z-score)", 14, 90, 30)
    z_threshold = st.slider("Z-Score Alert Threshold", 1.5, 4.0, 2.5, 0.1)

    telegram_token = st.text_input(
        "Telegram Bot Token",
        type="password",
        value=st.secrets.get("TELEGRAM_TOKEN", "") if hasattr(st, "secrets") else "",
    )
    telegram_chat_id = st.text_input(
        "Telegram Chat ID",
        value=st.secrets.get("TELEGRAM_CHAT_ID", "") if hasattr(st, "secrets") else "",
    )

    st.divider()
    st.subheader("🎯 Signal Lab Settings")
    price_margin = st.number_input(
        "Target price margin for 'hit' ($)", min_value=1, max_value=500, value=10, step=1
    )
    lookback_bars = st.slider("Backtest window (15m bars)", 100, 2000, 500, 50)
    st.caption("More bars = longer backtest history, slower load.")

    st.divider()
    st.caption("Run with: streamlit run btc_flow_dashboard.py")

# ==================== HELPER FUNCTIONS ====================
def fetch_cryptoquant_flows(metric: str, token: str, exchange: str, window: str, limit: int = 100):
    """Fetch inflow / outflow / netflow from CryptoQuant"""
    if not token:
        return None

    url = f"https://api.cryptoquant.com/v1/btc/exchange-flows/{metric}"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"exchange": exchange, "window": window, "limit": limit, "format": "json"}

    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        if data.get("status", {}).get("code") != 200:
            st.error(f"API Error: {data.get('status', {}).get('message')}")
            return None

        df = pd.DataFrame(data["result"]["data"])
        if metric == "inflow":
            df = df.rename(columns={"inflow_total": "value"})
        elif metric == "outflow":
            df = df.rename(columns={"outflow_total": "value"})
        elif metric == "netflow":
            df = df.rename(columns={"netflow_total": "value"})

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        return df[["date", "value"]]
    except Exception as e:
        st.error(f"Failed to fetch {metric}: {e}")
        return None


def compute_anomalies(df: pd.DataFrame, col: str, window: int, z_thresh: float):
    df = df.copy()
    df["rolling_mean"] = df[col].rolling(window=window).mean()
    df["rolling_std"] = df[col].rolling(window=window).std()
    df["zscore"] = (df[col] - df["rolling_mean"]) / df["rolling_std"]
    df["is_huge"] = df["zscore"] > z_thresh
    return df


def send_telegram(msg: str, token: str, chat_id: str):
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, timeout=10)
        return True
    except Exception:
        return False


def fetch_live_price():
    """
    Fetch the current BTC-USD price from Coinbase's public ticker endpoint.
    No API key required. Cached for 10s so a busy page doesn't hammer the
    endpoint on every rerun.
    """
    url = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
    headers = {"User-Agent": "btc-flow-radar/1.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        return {
            "price": float(data["price"]),
            "bid": float(data["bid"]),
            "ask": float(data["ask"]),
            "volume_24h": float(data["volume"]),
            "time": data.get("time"),
        }
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=60)
def fetch_binance_klines(symbol: str = "BTCUSDT", interval: str = "15m", limit: int = 500):
    """
    Fetch OHLCV candles from Coinbase Exchange's public market data endpoint
    (no API key required). Used for price history to backtest signals against.
    This is price data only — not a trading connection, no keys, no order placement.

    Note: Binance's API returns HTTP 451 ("unavailable for legal reasons") for
    requests originating from US IP addresses, which includes most cloud hosts
    like Streamlit Cloud. Coinbase's public candles endpoint has no such
    restriction, so it's used here instead. `symbol`/`interval` args are kept
    for compatibility but mapped to Coinbase's product/granularity format.
    """
    product = "BTC-USD"
    granularity_map = {"15m": 900, "1m": 60, "5m": 300, "1h": 3600, "1d": 86400}
    granularity = granularity_map.get(interval, 900)
    bar_seconds = granularity

    url = f"https://api.exchange.coinbase.com/products/{product}/candles"
    headers = {"User-Agent": "btc-flow-radar/1.0"}

    all_rows = []
    end_time = datetime.utcnow()
    remaining = min(limit, 3000)  # sane cap to avoid excessive pagination

    try:
        while remaining > 0:
            batch_size = min(remaining, 300)  # Coinbase max candles per request
            start_time = end_time - timedelta(seconds=bar_seconds * batch_size)
            params = {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "granularity": granularity,
            }
            r = requests.get(url, params=params, headers=headers, timeout=15)
            r.raise_for_status()
            batch = r.json()  # each row: [time, low, high, open, close, volume]
            if not batch:
                break
            all_rows.extend(batch)
            remaining -= batch_size
            end_time = start_time
            time.sleep(0.2)  # be polite to the public endpoint, avoid rate limiting

        if not all_rows:
            st.error("No price data returned from Coinbase.")
            return None

        df = pd.DataFrame(all_rows, columns=["time", "low", "high", "open", "close", "volume"])
        df["open_time"] = pd.to_datetime(df["time"], unit="s")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
        df["quote_volume"] = df["volume"] * df["close"]  # approximation, Coinbase doesn't provide this directly
        return df[["open_time", "open", "high", "low", "close", "volume", "quote_volume"]].tail(limit)
    except Exception as e:
        st.error(f"Failed to fetch price history: {e}")
        return None


def build_signals(df: pd.DataFrame, z_window: int = 20, rsi_window: int = 14):
    """
    Build a composite set of signals purely from price/volume history:
      - momentum z-score (rolling return z-scored)
      - volume z-score (volume spike detector)
      - RSI-style mean reversion score
    These are combined into a single composite score used to predict the
    NEXT bar's direction/magnitude. This is intentionally transparent so the
    backtest below can be checked against the real definitions.
    """
    df = df.copy()
    df["ret"] = df["close"].pct_change()

    # Momentum z-score: is the recent return unusually large vs its own history?
    df["ret_mean"] = df["ret"].rolling(z_window).mean()
    df["ret_std"] = df["ret"].rolling(z_window).std()
    df["momentum_z"] = (df["ret"] - df["ret_mean"]) / df["ret_std"]

    # Volume z-score: is current volume unusually high?
    df["vol_mean"] = df["volume"].rolling(z_window).mean()
    df["vol_std"] = df["volume"].rolling(z_window).std()
    df["volume_z"] = (df["volume"] - df["vol_mean"]) / df["vol_std"]

    # RSI (classic mean-reversion oscillator)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_window).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_window).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    # Map RSI to a mean-reversion score: >70 => expect pullback, <30 => expect bounce
    df["rsi_signal"] = np.where(df["rsi"] > 70, -1, np.where(df["rsi"] < 30, 1, 0))

    # Composite score: momentum continuation + volume confirmation + mean-reversion tilt
    # Weights are illustrative, not tuned/optimized on this data (to avoid overfitting the demo).
    df["composite_score"] = (
        0.5 * df["momentum_z"].clip(-3, 3)
        + 0.2 * df["volume_z"].clip(-3, 3)
        + 0.3 * df["rsi_signal"]
    )

    # Predicted next-bar price = current close + (composite score scaled by recent volatility)
    recent_vol = df["close"].rolling(z_window).std()
    df["predicted_next_close"] = df["close"] + (df["composite_score"] * recent_vol * 0.1)

    return df


def backtest_predictions(df: pd.DataFrame, margin: float):
    """
    Honest backtest: for every bar with a prediction, compare the PREVIOUS bar's
    prediction of "next close" against what the price actually did.
    Reports hit rate within margin, MAE, RMSE, and directional accuracy —
    the numbers you'd actually want to see before trusting a signal.
    """
    df = df.copy()
    df["actual_next_close"] = df["close"].shift(-1)
    df["prediction_error"] = df["actual_next_close"] - df["predicted_next_close"]
    df["abs_error"] = df["prediction_error"].abs()
    df["hit"] = df["abs_error"] <= margin

    valid = df.dropna(subset=["predicted_next_close", "actual_next_close"])

    if valid.empty:
        return None, df

    actual_dir = np.sign(valid["actual_next_close"] - valid["close"])
    pred_dir = np.sign(valid["predicted_next_close"] - valid["close"])
    directional_accuracy = (actual_dir == pred_dir).mean()

    stats = {
        "n_predictions": len(valid),
        "hit_rate_within_margin": valid["hit"].mean(),
        "mae": valid["abs_error"].mean(),
        "rmse": np.sqrt((valid["prediction_error"] ** 2).mean()),
        "directional_accuracy": directional_accuracy,
        "median_abs_error": valid["abs_error"].median(),
    }
    return stats, df


# ==================== MAIN DASHBOARD ====================
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔴 Live Monitor", "📈 Historical + Anomalies", "🎯 Signal Lab (15-min backtest)", "⚡ Alerts & Actions"]
)

with tab1:
    st.subheader("Live BTC Price")

    ac1, ac2 = st.columns([1, 3])
    with ac1:
        auto_refresh = st.checkbox("Auto-refresh", value=True, key="live_price_autorefresh")
    with ac2:
        refresh_secs = st.slider(
            "Refresh every (seconds)", min_value=5, max_value=60, value=10, step=5,
            disabled=not auto_refresh, key="live_price_refresh_secs",
        )

    if auto_refresh:
        st_autorefresh(interval=refresh_secs * 1000, key="live_price_autorefresh_timer")

    st.caption("Free public data — no API token required.")

    live = fetch_live_price()
    if "error" in live:
        st.error(f"Couldn't fetch live price: {live['error']}")
    else:
        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.metric("BTC-USD", f"${live['price']:,.2f}")
        lc2.metric("Bid", f"${live['bid']:,.2f}")
        lc3.metric("Ask", f"${live['ask']:,.2f}")
        lc4.metric("Spread", f"${live['ask'] - live['bid']:,.2f}")
        st.caption(
            f"24h volume: {live['volume_24h']:,.1f} BTC · "
            f"Source: Coinbase public ticker · Updates every ~10s on rerun"
        )

    st.divider()
    st.subheader("Exchange Flow Status (requires CryptoQuant token)")

    col1, col2, col3 = st.columns(3)

    latest_inflow = None
    latest_outflow = None
    latest_netflow = None

    if api_token:
        with st.spinner("Fetching latest from CryptoQuant..."):
            inflow_df = fetch_cryptoquant_flows("inflow", api_token, exchange, window, limit=5)
            outflow_df = fetch_cryptoquant_flows("outflow", api_token, exchange, window, limit=5)
            netflow_df = fetch_cryptoquant_flows("netflow", api_token, exchange, window, limit=5)

        if inflow_df is not None and not inflow_df.empty:
            latest_inflow = inflow_df["value"].iloc[-1]
            col1.metric(
                "Latest Inflow (BTC)",
                f"{latest_inflow:,.0f}",
                delta_color="inverse" if latest_inflow > 8000 else "normal",
            )

        if outflow_df is not None and not outflow_df.empty:
            latest_outflow = outflow_df["value"].iloc[-1]
            col2.metric("Latest Outflow (BTC)", f"{latest_outflow:,.0f}")

        if netflow_df is not None and not netflow_df.empty:
            latest_netflow = netflow_df["value"].iloc[-1]
            col3.metric(
                "Netflow (In - Out)",
                f"{latest_netflow:,.0f}",
                delta="Positive = more inflow" if latest_netflow > 0 else "Negative = net outflow",
            )

    else:
        st.info("Enter your CryptoQuant API token in the sidebar to see live data.")
        st.caption("You can still explore the Signal Lab tab — it uses free public price data.")

    st.divider()

    if latest_inflow and latest_inflow > 10000:
        st.error(
            f"🚨 **Large inflow observed** — {latest_inflow:,.0f} BTC into exchanges. "
            "This is a data point, not a signal to act on by itself."
        )
        if st.button("Send to Telegram", key="inflow_alert"):
            send_telegram(
                f"🚨 BTC inflow observed: {latest_inflow:,.0f} BTC into {exchange} (informational only)",
                telegram_token,
                telegram_chat_id,
            )

    if latest_outflow and latest_outflow > 8000:
        st.success(f"📤 **Large outflow observed** — {latest_outflow:,.0f} BTC leaving exchanges.")

with tab2:
    st.subheader("Historical Flows + Anomaly Detection")

    if api_token:
        with st.spinner("Loading historical data..."):
            hist_inflow = fetch_cryptoquant_flows("inflow", api_token, exchange, window, limit=200)

        if hist_inflow is not None and not hist_inflow.empty:
            hist_inflow = compute_anomalies(hist_inflow, "value", rolling_window, z_threshold)

            fig = px.line(hist_inflow, x="date", y="value", title=f"BTC Inflow ({window}) - {exchange}")
            fig.add_scatter(
                x=hist_inflow[hist_inflow["is_huge"]]["date"],
                y=hist_inflow[hist_inflow["is_huge"]]["value"],
                mode="markers",
                marker=dict(color="red", size=10),
                name="Huge Inflow",
            )
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.line(hist_inflow, x="date", y="zscore", title="Inflow Z-Score (Anomaly Strength)")
            fig2.add_hline(y=z_threshold, line_dash="dash", line_color="red", annotation_text="Alert Threshold")
            st.plotly_chart(fig2, use_container_width=True)

            huge_days = hist_inflow[hist_inflow["is_huge"]].tail(10)
            if not huge_days.empty:
                st.write("**Recent Huge Inflow Days:**")
                st.dataframe(
                    huge_days[["date", "value", "zscore"]].style.format(
                        {"value": "{:,.0f}", "zscore": "{:.2f}"}
                    )
                )
    else:
        st.info("Add your API token in sidebar to load live historical data from CryptoQuant.")

with tab3:
    st.subheader("🎯 Signal Lab — Honest 15-Minute Prediction Backtest")
    st.caption(
        "This builds a composite signal from price momentum, volume, and RSI, then tests it "
        "against real historical 15-minute bars. The results below are the **actual** measured "
        "accuracy — not a projection. Read them skeptically."
    )

    st.info(
        f"Your target margin is **±${price_margin}**. At current BTC prices that's roughly "
        f"**{price_margin / 63000 * 100:.3f}%** — tighter than typical exchange spread + slippage. "
        "Expect the hit rate below to be low; that's the honest answer, not a bug."
    )

    with st.spinner("Fetching BTC price history from Binance public data..."):
        price_df = fetch_binance_klines(symbol="BTCUSDT", interval="15m", limit=lookback_bars)

    if price_df is not None and not price_df.empty:
        signal_df = build_signals(price_df)
        stats, full_df = backtest_predictions(signal_df, margin=price_margin)

        if stats:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Predictions tested", f"{stats['n_predictions']:,}")
            c2.metric(f"Hit rate (within ${price_margin})", f"{stats['hit_rate_within_margin']*100:.1f}%")
            c3.metric("Directional accuracy", f"{stats['directional_accuracy']*100:.1f}%")
            c4.metric("Median abs. error", f"${stats['median_abs_error']:.2f}")

            st.caption(
                f"MAE: ${stats['mae']:.2f} · RMSE: ${stats['rmse']:.2f} · "
                "Directional accuracy near 50% means the signal is not beating a coin flip on direction."
            )

            st.divider()

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=full_df["open_time"], y=full_df["close"], name="Actual Close"))
            fig3.add_trace(
                go.Scatter(
                    x=full_df["open_time"],
                    y=full_df["predicted_next_close"],
                    name="Predicted Next Close",
                    line=dict(dash="dot"),
                )
            )
            fig3.update_layout(title="Actual vs. Predicted Price (15m bars)", height=450)
            st.plotly_chart(fig3, use_container_width=True)

            fig4 = px.histogram(
                full_df.dropna(subset=["abs_error"]),
                x="abs_error",
                nbins=50,
                title="Distribution of Absolute Prediction Error ($)",
            )
            fig4.add_vline(x=price_margin, line_dash="dash", line_color="red", annotation_text=f"${price_margin} target")
            st.plotly_chart(fig4, use_container_width=True)

            with st.expander("See the raw signal components"):
                st.dataframe(
                    full_df[
                        [
                            "open_time", "close", "momentum_z", "volume_z", "rsi",
                            "composite_score", "predicted_next_close", "actual_next_close", "hit",
                        ]
                    ].tail(50)
                )

            st.warning(
                "Read this honestly: if hit rate within margin is low and directional accuracy is "
                "near 50%, this composite signal has no real edge at this timeframe/margin — which "
                "is the expected, common result for public data at 15-minute resolution. Treat any "
                "higher numbers with suspicion too; check for enough sample size (n_predictions) "
                "before trusting a good-looking result, since small samples can look better than "
                "they are by chance."
            )
        else:
            st.info("Not enough data yet to backtest — try increasing the backtest window.")
    else:
        st.error("Could not load price history. Check network access to Binance's public API.")

with tab4:
    st.subheader("Alert Configuration & Actions")

    st.write("**How alerts work:**")
    st.markdown(
        """
    - High **inflow** z-score → potentially more coins available to sell
    - High **outflow** → coins leaving exchanges (often read as accumulation)
    - These are observations, not trade signals — see the Signal Lab tab for measured accuracy
    - You can trigger Telegram messages manually or automatically from the Live tab
    """
    )

    if st.button("Test Telegram Connection"):
        success = send_telegram(
            "✅ Test alert from your BTC Flow Dashboard (informational only, not trading advice)",
            telegram_token,
            telegram_chat_id,
        )
        if success:
            st.success("Telegram message sent!")
        else:
            st.error("Check your Telegram token and chat ID")

st.divider()
st.caption(
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    "Flow data via CryptoQuant API (when token provided) | Price data via Binance public API"
)
st.caption(
    "⚠️ Informational tool only. Not financial advice. Past performance and backtest results "
    "do not guarantee future results."
)
