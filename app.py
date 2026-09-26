
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(page_title="FRVP Auto Backtester", page_icon="📊", layout="wide")

# -----------------------------
# FRVP calculation
# -----------------------------
def calculate_frvp(df, rows=60, va_pct=70):
    lo = float(df["Low"].min())
    hi = float(df["High"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None

    edges = np.linspace(lo, hi, rows + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    profile = np.zeros(rows, dtype=float)

    for r in df.itertuples():
        low, high, vol, close = float(r.Low), float(r.High), float(r.Volume), float(r.Close)
        if not np.isfinite(vol) or vol <= 0:
            continue

        if high <= low:
            idx = int(np.clip(np.searchsorted(edges, close, side="right") - 1, 0, rows - 1))
            profile[idx] += vol
            continue

        first = int(np.clip(np.searchsorted(edges, low, side="right") - 1, 0, rows - 1))
        last = int(np.clip(np.searchsorted(edges, high, side="left"), 0, rows - 1))
        ids = np.arange(min(first, last), max(first, last) + 1)

        overlap = np.maximum(
            0,
            np.minimum(high, edges[ids + 1]) - np.maximum(low, edges[ids])
        )
        s = overlap.sum()
        if s > 0:
            profile[ids] += vol * overlap / s

    if profile.sum() <= 0:
        return None

    poc_i = int(np.argmax(profile))
    target = profile.sum() * va_pct / 100.0
    included = {poc_i}
    total = profile[poc_i]
    left, right = poc_i - 1, poc_i + 1

    while total < target and (left >= 0 or right < rows):
        lv = profile[left] if left >= 0 else -1
        rv = profile[right] if right < rows else -1

        if rv >= lv:
            if right < rows:
                included.add(right); total += profile[right]; right += 1
            elif left >= 0:
                included.add(left); total += profile[left]; left -= 1
        else:
            if left >= 0:
                included.add(left); total += profile[left]; left -= 1
            elif right < rows:
                included.add(right); total += profile[right]; right += 1

    low_i, high_i = min(included), max(included)
    return {
        "POC": float(centers[poc_i]),
        "VAL": float(edges[low_i]),
        "VAH": float(edges[high_i + 1]),
        "edges": edges,
        "centers": centers,
        "profile": profile,
    }

# -----------------------------
# Price-action signals
# -----------------------------
def bullish_rejection(c, level):
    rng = max(c.High - c.Low, 1e-12)
    body = abs(c.Close - c.Open)
    lower_wick = min(c.Open, c.Close) - c.Low
    return c.Low <= level and c.Close > level and c.Close > c.Open and (
        body / rng >= 0.30 or lower_wick / rng >= 0.35
    )

def bearish_rejection(c, level):
    rng = max(c.High - c.Low, 1e-12)
    body = abs(c.Close - c.Open)
    upper_wick = c.High - max(c.Open, c.Close)
    return c.High >= level and c.Close < level and c.Close < c.Open and (
        body / rng >= 0.30 or upper_wick / rng >= 0.35
    )

def detect_signal(history, current, setup, rows, va_pct, retest_pct):
    if len(history) < 2:
        return None

    p = history.iloc[-1]
    c = current
    profile = calculate_frvp(history, rows, va_pct)
    if profile is None:
        return None

    poc, val, vah = profile["POC"], profile["VAL"], profile["VAH"]
    tolerance = max(vah - val, 1e-12) * retest_pct / 100

    if setup in ("All", "POC Bounce"):
        if bullish_rejection(c, poc):
            return {"side":"BUY", "setup":"POC Bounce", "profile":profile}
        if bearish_rejection(c, poc):
            return {"side":"SELL", "setup":"POC Bounce", "profile":profile}

    if setup in ("All", "POC Reversal"):
        if p.Close < poc and c.Close > poc and c.Close > c.Open:
            return {"side":"BUY", "setup":"POC Reversal", "profile":profile}
        if p.Close > poc and c.Close < poc and c.Close < c.Open:
            return {"side":"SELL", "setup":"POC Reversal", "profile":profile}

    if setup in ("All", "VAH/VAL Breakout"):
        if p.Close > vah and c.Low <= vah + tolerance and c.Close > vah and c.Close > c.Open:
            return {"side":"BUY", "setup":"VAH Breakout", "profile":profile}
        if p.Close < val and c.High >= val - tolerance and c.Close < val and c.Close < c.Open:
            return {"side":"SELL", "setup":"VAL Breakout", "profile":profile}

    return None

def make_trade(signal, bar, sl_buffer, target_mode, rr_target):
    side = signal["side"]
    entry = float(bar.Close)

    if side == "BUY":
        sl = float(bar.Low - sl_buffer)
        frvp_target = float(signal["profile"]["VAH"])
        target = entry + abs(entry - sl) * rr_target if target_mode == "RR" else frvp_target
        if target <= entry:
            return None
    else:
        sl = float(bar.High + sl_buffer)
        frvp_target = float(signal["profile"]["VAL"])
        target = entry - abs(entry - sl) * rr_target if target_mode == "RR" else frvp_target
        if target >= entry:
            return None

    return {
        "side":side, "setup":signal["setup"], "entry":entry, "sl":sl, "tp":target,
        "entry_time":bar.Date, "status":"OPEN", "exit":np.nan, "exit_time":pd.NaT,
        "result":None, "r_multiple":np.nan, "bars_held":0
    }

def check_trade(trade, bar):
    trade["bars_held"] += 1
    if trade["side"] == "BUY":
        if bar.Low <= trade["sl"]:
            trade.update(status="CLOSED", exit=trade["sl"], exit_time=bar.Date,
                         result="LOSS", r_multiple=-1.0)
            return True
        if bar.High >= trade["tp"]:
            r = (trade["tp"]-trade["entry"]) / (trade["entry"]-trade["sl"])
            trade.update(status="CLOSED", exit=trade["tp"], exit_time=bar.Date,
                         result="WIN", r_multiple=r)
            return True
    else:
        if bar.High >= trade["sl"]:
            trade.update(status="CLOSED", exit=trade["sl"], exit_time=bar.Date,
                         result="LOSS", r_multiple=-1.0)
            return True
        if bar.Low <= trade["tp"]:
            r = (trade["entry"]-trade["tp"]) / (trade["sl"]-trade["entry"])
            trade.update(status="CLOSED", exit=trade["tp"], exit_time=bar.Date,
                         result="WIN", r_multiple=r)
            return True
    return False

def backtest(df, lookback, rows, va_pct, setup, sl_buffer, target_mode, rr_target, max_bars, retest_pct):
    trades = []
    open_trade = None

    for i in range(max(lookback, 2), len(df)):
        bar = df.iloc[i]

        if open_trade is not None:
            if check_trade(open_trade, bar):
                trades.append(open_trade.copy())
                open_trade = None
                continue

            if open_trade["bars_held"] >= max_bars:
                if open_trade["side"] == "BUY":
                    r = (bar.Close-open_trade["entry"]) / (open_trade["entry"]-open_trade["sl"])
                else:
                    r = (open_trade["entry"]-bar.Close) / (open_trade["sl"]-open_trade["entry"])
                open_trade.update(
                    status="CLOSED", exit=float(bar.Close), exit_time=bar.Date,
                    result="WIN" if r > 0 else "LOSS", r_multiple=float(r)
                )
                trades.append(open_trade.copy())
                open_trade = None
                continue

        if open_trade is None:
            history = df.iloc[i-lookback:i]
            sig = detect_signal(history, bar, setup, rows, va_pct, retest_pct)
            if sig:
                open_trade = make_trade(sig, bar, sl_buffer, target_mode, rr_target)

    if open_trade is not None:
        bar = df.iloc[-1]
        if open_trade["side"] == "BUY":
            r = (bar.Close-open_trade["entry"]) / (open_trade["entry"]-open_trade["sl"])
        else:
            r = (open_trade["entry"]-bar.Close) / (open_trade["sl"]-open_trade["entry"])
        open_trade.update(
            status="CLOSED", exit=float(bar.Close), exit_time=bar.Date,
            result="WIN" if r > 0 else "LOSS", r_multiple=float(r)
        )
        trades.append(open_trade.copy())

    return pd.DataFrame(trades)

# -----------------------------
# Automatic market-data fetch
# -----------------------------
@st.cache_data(ttl=900, show_spinner=False)
def download_data(interval, period):
    data = yf.download(
        "XAUUSD=X",
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.rename(columns={
        "Open":"Open", "High":"High", "Low":"Low",
        "Close":"Close", "Volume":"Volume"
    })

    needed = ["Open","High","Low","Close","Volume"]
    if not all(c in data.columns for c in needed):
        return pd.DataFrame()

    data = data[needed].copy()
    data.index = pd.to_datetime(data.index)
    if getattr(data.index, "tz", None) is not None:
        data.index = data.index.tz_convert(None)

    data["Date"] = data.index
    data = data.reset_index(drop=True)
    return data.dropna().reset_index(drop=True)

# -----------------------------
# UI
# -----------------------------
st.title("FRVP Price Action Backtester")
st.caption("Automatic XAUUSD historical data • Fixed Range Volume Profile + price action only")

with st.sidebar:
    st.header("FRVP")
    rows = st.number_input("Row Size", 10, 200, 60, 5)
    va_pct = st.number_input("Value Area %", 50, 90, 70, 1)
    lookback = st.number_input("Fixed Range Bars", 20, 1000, 150, 10)

    st.header("Market Data")
    interval = st.selectbox("Timeframe", ["5m", "15m", "30m", "1h", "4h", "1d"], index=1)
    period = st.selectbox(
        "Historical data",
        ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )

    st.header("Strategy")
    setup = st.selectbox("Setup", ["All", "POC Bounce", "POC Reversal", "VAH/VAL Breakout"])

    st.header("Trade Rules")
    sl_buffer = st.number_input("SL buffer", 0.0, 100.0, 0.20, 0.05)
    target_mode = st.selectbox("Target", ["FRVP Value Boundary", "RR"])
    rr_target = st.number_input("RR Target", 0.5, 10.0, 1.5, 0.25)
    max_bars = st.number_input("Maximum bars in trade", 1, 500, 30, 1)
    retest_pct = st.number_input("Breakout retest tolerance (% value width)", 0.0, 50.0, 8.0, 1.0)

    run = st.button("🔄 Fetch Data & Run Backtest", type="primary", use_container_width=True)

if "run_once" not in st.session_state:
    st.session_state.run_once = False
if run:
    st.session_state.run_once = True

if st.session_state.run_once:
    with st.spinner("Fetching XAUUSD historical data and running backtest..."):
        df = download_data(interval, period)

    if df.empty:
        st.error("Automatic XAUUSD data could not be loaded. Try another timeframe/period.")
        st.stop()

    st.success(f"Loaded {len(df):,} XAUUSD candles automatically.")

    if len(df) < lookback + 20:
        st.warning(f"Not enough candles. Reduce Fixed Range Bars below {len(df)-20}.")
        st.stop()

    results = backtest(
        df, int(lookback), int(rows), int(va_pct), setup,
        float(sl_buffer), target_mode, float(rr_target),
        int(max_bars), float(retest_pct)
    )

    if results.empty:
        st.warning("No trades matched these FRVP price-action rules.")
        st.stop()

    wins = int((results.result == "WIN").sum())
    losses = int((results.result == "LOSS").sum())
    total = len(results)
    winrate = wins / total * 100
    net_r = results.r_multiple.sum()
    avg_r = results.r_multiple.mean()
    gp = results.loc[results.r_multiple > 0, "r_multiple"].sum()
    gl = -results.loc[results.r_multiple < 0, "r_multiple"].sum()
    pf = gp / gl if gl > 0 else np.inf

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Win Rate", f"{winrate:.2f}%")
    c2.metric("Trades", total)
    c3.metric("Net R", f"{net_r:.2f}R")
    c4.metric("Avg R", f"{avg_r:.2f}R")
    c5.metric("Profit Factor", "∞" if np.isinf(pf) else f"{pf:.2f}")

    st.subheader("Performance by Setup")
    by_setup = results.groupby("setup").agg(
        Trades=("result","size"),
        Wins=("result", lambda x: (x=="WIN").sum()),
        Losses=("result", lambda x: (x=="LOSS").sum()),
        Net_R=("r_multiple","sum"),
        Avg_R=("r_multiple","mean")
    ).reset_index()
    by_setup["WinRate_%"] = by_setup["Wins"] / by_setup["Trades"] * 100
    st.dataframe(by_setup, hide_index=True, use_container_width=True)

    st.subheader("Equity Curve")
    eq = results["r_multiple"].cumsum()
    fig_eq = go.Figure(go.Scatter(
        x=results["exit_time"], y=eq, mode="lines", name="Cumulative R"
    ))
    fig_eq.update_layout(height=350, xaxis_title="Exit", yaxis_title="R")
    st.plotly_chart(fig_eq, use_container_width=True)

    st.subheader("Trade Log")
    cols = ["entry_time","exit_time","setup","side","entry","sl","tp","exit","result","r_multiple","bars_held"]
    st.dataframe(results[cols], hide_index=True, use_container_width=True)

    st.download_button(
        "Download Trade Log CSV",
        results.to_csv(index=False).encode("utf-8"),
        "frvp_backtest_trades.csv",
        "text/csv"
    )

    st.subheader("Latest XAUUSD / FRVP")
    last = df.tail(int(lookback))
    prof = calculate_frvp(last, int(rows), int(va_pct))
    chart = df.tail(min(300, len(df)))

    fig = go.Figure(go.Candlestick(
        x=chart.Date, open=chart.Open, high=chart.High,
        low=chart.Low, close=chart.Close, name="XAUUSD"
    ))
    for y, label, dash in [
        (prof["POC"], "POC", "solid"),
        (prof["VAH"], "VAH", "dash"),
        (prof["VAL"], "VAL", "dash"),
    ]:
        fig.add_hline(y=y, line_dash=dash, annotation_text=label)
    fig.update_layout(height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Backtest integrity: each test candle calculates FRVP from the preceding fixed "
        "range only. It does not use future candles to create the signal. If SL and TP "
        "are both touched within one OHLC candle, SL is counted first because intrabar "
        "order cannot be known from OHLC data alone."
    )
else:
    st.info("Choose your timeframe/period and press **Fetch Data & Run Backtest**. No CSV upload is required.")
