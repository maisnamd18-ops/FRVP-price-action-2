import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="FRVP Price Action Backtester", layout="wide")

st.title("FRVP Price Action Backtester")
st.caption("Fixed Range Volume Profile + price action only • XAUUSD")

# -----------------------------
# FRVP
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
                included.add(right)
                total += profile[right]
                right += 1
            elif left >= 0:
                included.add(left)
                total += profile[left]
                left -= 1
        else:
            if left >= 0:
                included.add(left)
                total += profile[left]
                left -= 1
            elif right < rows:
                included.add(right)
                total += profile[right]
                right += 1

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
# Price action
# -----------------------------
def bullish_rejection(c, level):
    rng = max(c.High - c.Low, 1e-12)
    body = abs(c.Close - c.Open)
    lower_wick = min(c.Open, c.Close) - c.Low
    return (
        c.Low <= level
        and c.Close > level
        and c.Close > c.Open
        and (body / rng >= 0.30 or lower_wick / rng >= 0.35)
    )

def bearish_rejection(c, level):
    rng = max(c.High - c.Low, 1e-12)
    body = abs(c.Close - c.Open)
    upper_wick = c.High - max(c.Open, c.Close)
    return (
        c.High >= level
        and c.Close < level
        and c.Close < c.Open
        and (body / rng >= 0.30 or upper_wick / rng >= 0.35)
    )

def detect_signal(history, current, setup):
    # history contains bars before current; FRVP must be based only on those bars.
    if len(history) < 2:
        return None

    p = history.iloc[-1]
    c = current

    profile = calculate_frvp(history, rows=st.session_state.rows, va_pct=st.session_state.va_pct)
    if profile is None:
        return None

    poc, val, vah = profile["POC"], profile["VAL"], profile["VAH"]
    value_width = max(vah - val, 1e-12)
    tolerance = value_width * st.session_state.retest_pct / 100.0

    # POC Bounce
    if setup in ("All", "POC Bounce"):
        if bullish_rejection(c, poc):
            return {"side": "BUY", "setup": "POC Bounce", "level": poc, "profile": profile}
        if bearish_rejection(c, poc):
            return {"side": "SELL", "setup": "POC Bounce", "level": poc, "profile": profile}

    # POC Reversal: prior close on one side, current candle closes across POC.
    if setup in ("All", "POC Reversal"):
        if p.Close < poc and c.Close > poc and c.Close > c.Open:
            return {"side": "BUY", "setup": "POC Reversal", "level": poc, "profile": profile}
        if p.Close > poc and c.Close < poc and c.Close < c.Open:
            return {"side": "SELL", "setup": "POC Reversal", "level": poc, "profile": profile}

    # VAH/VAL breakout + retest
    if setup in ("All", "VAH/VAL Breakout"):
        if p.Close > vah and c.Low <= vah + tolerance and c.Close > vah and c.Close > c.Open:
            return {"side": "BUY", "setup": "VAH Breakout", "level": vah, "profile": profile}
        if p.Close < val and c.High >= val - tolerance and c.Close < val and c.Close < c.Open:
            return {"side": "SELL", "setup": "VAL Breakout", "level": val, "profile": profile}

    return None

# -----------------------------
# Trade simulator
# -----------------------------
def make_trade(signal, bar, sl_buffer, target_mode, rr_target):
    side = signal["side"]
    entry = float(bar.Close)

    if side == "BUY":
        sl = float(bar.Low - sl_buffer)
        base_target = signal["profile"]["VAH"]
        if target_mode == "RR":
            target = entry + abs(entry - sl) * rr_target
        else:
            target = base_target
        if target <= entry:
            return None
    else:
        sl = float(bar.High + sl_buffer)
        base_target = signal["profile"]["VAL"]
        if target_mode == "RR":
            target = entry - abs(entry - sl) * rr_target
        else:
            target = base_target
        if target >= entry:
            return None

    return {
        "side": side,
        "setup": signal["setup"],
        "entry": entry,
        "sl": sl,
        "tp": float(target),
        "entry_time": bar.Date,
        "status": "OPEN",
        "exit": np.nan,
        "exit_time": pd.NaT,
        "result": None,
        "r_multiple": np.nan,
        "bars_held": 0,
    }

def check_trade(trade, bar):
    trade["bars_held"] += 1

    if trade["side"] == "BUY":
        hit_sl = bar.Low <= trade["sl"]
        hit_tp = bar.High >= trade["tp"]

        # Conservative assumption if both occur in the same OHLC bar:
        # SL is counted first because intrabar sequence is unknown.
        if hit_sl:
            trade.update(status="CLOSED", exit=trade["sl"], exit_time=bar.Date,
                         result="LOSS", r_multiple=-1.0)
            return True
        if hit_tp:
            trade.update(status="CLOSED", exit=trade["tp"], exit_time=bar.Date,
                         result="WIN", r_multiple=(trade["tp"]-trade["entry"]) / (trade["entry"]-trade["sl"]))
            return True
    else:
        hit_sl = bar.High >= trade["sl"]
        hit_tp = bar.Low <= trade["tp"]

        if hit_sl:
            trade.update(status="CLOSED", exit=trade["sl"], exit_time=bar.Date,
                         result="LOSS", r_multiple=-1.0)
            return True
        if hit_tp:
            trade.update(status="CLOSED", exit=trade["tp"], exit_time=bar.Date,
                         result="WIN", r_multiple=(trade["entry"]-trade["tp"]) / (trade["sl"]-trade["entry"]))
            return True

    return False

def backtest(df, lookback, rows, va_pct, setup, sl_buffer, target_mode, rr_target, max_bars):
    st.session_state.rows = rows
    st.session_state.va_pct = va_pct
    st.session_state.retest_pct = retest_pct

    trades = []
    open_trade = None

    start = max(lookback, 2)

    for i in range(start, len(df)):
        bar = df.iloc[i]

        if open_trade is not None:
            closed = check_trade(open_trade, bar)
            if closed:
                trades.append(open_trade.copy())
                open_trade = None
                continue

            if open_trade["bars_held"] >= max_bars:
                if open_trade["side"] == "BUY":
                    r = (bar.Close - open_trade["entry"]) / (open_trade["entry"] - open_trade["sl"])
                else:
                    r = (open_trade["entry"] - bar.Close) / (open_trade["sl"] - open_trade["entry"])
                open_trade.update(
                    status="CLOSED", exit=float(bar.Close), exit_time=bar.Date,
                    result="WIN" if r > 0 else "LOSS", r_multiple=float(r)
                )
                trades.append(open_trade.copy())
                open_trade = None
                continue

        # One position at a time.
        if open_trade is None:
            history = df.iloc[i-lookback:i]
            sig = detect_signal(history, bar, setup)
            if sig:
                open_trade = make_trade(sig, bar, sl_buffer, target_mode, rr_target)

    # Close an unfinished trade at last available close.
    if open_trade is not None:
        bar = df.iloc[-1]
        if open_trade["side"] == "BUY":
            r = (bar.Close - open_trade["entry"]) / (open_trade["entry"] - open_trade["sl"])
        else:
            r = (open_trade["entry"] - bar.Close) / (open_trade["sl"] - open_trade["entry"])
        open_trade.update(
            status="CLOSED", exit=float(bar.Close), exit_time=bar.Date,
            result="WIN" if r > 0 else "LOSS", r_multiple=float(r)
        )
        trades.append(open_trade.copy())

    return pd.DataFrame(trades)

# -----------------------------
# UI
# -----------------------------
with st.sidebar:
    st.header("FRVP Settings")
    rows = st.number_input("Row Size", 10, 200, 60, 5)
    va_pct = st.number_input("Value Area %", 50, 90, 70, 1)
    lookback = st.number_input("Fixed Range Bars", 20, 1000, 150, 10)
    setup = st.selectbox("Strategy", ["All", "POC Bounce", "POC Reversal", "VAH/VAL Breakout"])

    st.header("Backtest")
    sl_buffer = st.number_input("SL buffer (price)", 0.0, 100.0, 0.20, 0.05)
    target_mode = st.selectbox("Target", ["FRVP Value Boundary", "RR"])
    rr_target = st.number_input("RR target", 0.5, 10.0, 1.5, 0.25)
    max_bars = st.number_input("Maximum bars in trade", 1, 500, 30, 1)
    retest_pct = st.number_input("Breakout retest tolerance (% of value width)", 0.0, 50.0, 8.0, 1.0)

    st.session_state.rows = int(rows)
    st.session_state.va_pct = int(va_pct)
    st.session_state.retest_pct = float(retest_pct)

uploaded = st.file_uploader("Upload XAUUSD historical OHLCV CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    # Normalize common column names.
    rename = {}
    for c in df.columns:
        k = c.strip().lower().replace(" ", "").replace("_", "")
        if k in ("date", "datetime", "time", "timestamp"):
            rename[c] = "Date"
        elif k == "open":
            rename[c] = "Open"
        elif k == "high":
            rename[c] = "High"
        elif k == "low":
            rename[c] = "Low"
        elif k == "close":
            rename[c] = "Close"
        elif k in ("volume", "tickvolume"):
            rename[c] = "Volume"

    df = df.rename(columns=rename)
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)

    if missing:
        st.error("Missing columns: " + ", ".join(sorted(missing)))
        st.stop()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date")
    else:
        df["Date"] = pd.RangeIndex(len(df))

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Open","High","Low","Close","Volume","Date"]).reset_index(drop=True)

    if len(df) < lookback + 10:
        st.warning(f"Need at least {lookback + 10} bars. This file has {len(df)}.")
        st.stop()

    if st.button("Run Backtest", type="primary"):
        with st.spinner("Backtesting FRVP price action..."):
            results = backtest(
                df, int(lookback), int(rows), int(va_pct), setup,
                float(sl_buffer), target_mode, float(rr_target), int(max_bars)
            )

        if results.empty:
            st.warning("No trades matched the selected FRVP price-action rules.")
            st.stop()

        wins = int((results.result == "WIN").sum())
        losses = int((results.result == "LOSS").sum())
        total = len(results)
        winrate = wins / total * 100
        avg_r = results.r_multiple.mean()
        net_r = results.r_multiple.sum()
        gross_profit = results.loc[results.r_multiple > 0, "r_multiple"].sum()
        gross_loss = -results.loc[results.r_multiple < 0, "r_multiple"].sum()
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        max_loss_streak = 0
        cur = 0
        for x in results.result:
            if x == "LOSS":
                cur += 1
                max_loss_streak = max(max_loss_streak, cur)
            else:
                cur = 0

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Win Rate", f"{winrate:.2f}%")
        c2.metric("Trades", total)
        c3.metric("Net R", f"{net_r:.2f}R")
        c4.metric("Avg R / Trade", f"{avg_r:.2f}R")
        c5.metric("Profit Factor", "∞" if np.isinf(profit_factor) else f"{profit_factor:.2f}")

        st.subheader("Equity Curve (R)")
        eq = results["r_multiple"].cumsum()
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=results["exit_time"], y=eq, mode="lines", name="Cumulative R"))
        fig_eq.update_layout(height=350, xaxis_title="Exit time", yaxis_title="R")
        st.plotly_chart(fig_eq, use_container_width=True)

        st.subheader("Results by Setup")
        by_setup = results.groupby("setup").agg(
            Trades=("result","size"),
            Wins=("result", lambda x: (x=="WIN").sum()),
            Losses=("result", lambda x: (x=="LOSS").sum()),
            Net_R=("r_multiple","sum"),
            Avg_R=("r_multiple","mean")
        ).reset_index()
        by_setup["WinRate_%"] = by_setup["Wins"] / by_setup["Trades"] * 100
        st.dataframe(by_setup, hide_index=True, use_container_width=True)

        st.subheader("Trade Log")
        display_cols = ["entry_time","exit_time","setup","side","entry","sl","tp","exit","result","r_multiple","bars_held"]
        st.dataframe(results[display_cols], hide_index=True, use_container_width=True)

        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button("Download Trade Log CSV", csv, "frvp_backtest_trades.csv", "text/csv")

        st.subheader("Last FRVP")
        last_profile = calculate_frvp(df.tail(int(lookback)), int(rows), int(va_pct))
        chart_df = df.tail(min(int(lookback), 300))

        fig = go.Figure(data=[go.Candlestick(
            x=chart_df.Date, open=chart_df.Open, high=chart_df.High,
            low=chart_df.Low, close=chart_df.Close, name="XAUUSD"
        )])
        for y, label, dash in [
            (last_profile["POC"], "POC", "solid"),
            (last_profile["VAH"], "VAH", "dash"),
            (last_profile["VAL"], "VAL", "dash"),
        ]:
            fig.add_hline(y=y, line_dash=dash, annotation_text=label)
        fig.update_layout(height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info(
        "Upload historical XAUUSD OHLCV data, choose the FRVP settings, then press "
        "**Run Backtest** to calculate win rate, Net R, Profit Factor, setup statistics and the trade log."
    )

st.caption(
    "Backtest note: the FRVP for each test bar uses only the preceding fixed range, "
    "which avoids using future candles to calculate POC/VAH/VAL. If both SL and TP are "
    "inside the same OHLC bar, the test counts SL first because intrabar order is unknown."
)
