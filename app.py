import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, time
import pytz
from streamlit_autorefresh import st_autorefresh

# ---------- Page + Auto-refresh ----------
st.set_page_config(page_title="TJR NY Session Agent", layout="wide")
# Auto-refresh every 30s to re-evaluate signals and exits
st_autorefresh(interval=30_000, key="tjr_autorefresh_30s")

# ---------- Timezone + market schedule helpers ----------
NY_TZ = pytz.timezone("America/New_York")

def now_ny():
    return datetime.now(NY_TZ)

def futures_closed_now(dt=None):
    """CME ES/NQ hours: Sun 18:00 → Fri 17:00 ET, daily break 16:00–17:00 ET."""
    if dt is None:
        dt = now_ny()
    wd = dt.weekday()  # 0=Mon ... 4=Fri, 5=Sat, 6=Sun
    t = dt.time()

    # Daily maintenance break (no new bars)
    if time(16, 0) <= t < time(17, 0):
        return True
    # Weekend close: Fri 17:00 → Sun 18:00
    if wd == 4 and t >= time(17, 0):  # Fri after 5pm ET
        return True
    if wd == 5:                        # Saturday
        return True
    if wd == 6 and t < time(18, 0):    # Sunday before 6pm ET
        return True
    return False

# ---------- Data fetch ----------
def fetch_intraday(symbol, days=3, interval="5m"):
    """Fetch intraday/daily data, localize to NY time, return (df, age_min)."""
    prepost = not (time(9, 30) <= now_ny().time() <= time(16, 0))
    df = yf.download(symbol, period=f"{days}d", interval=interval,
                     prepost=prepost, progress=False)
    if df is None or df.empty:
        return pd.DataFrame(), 9999

    # Normalize tz + columns
    if not df.index.tz:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(NY_TZ)
    df.rename(columns=str.lower, inplace=True)

    age_min = (now_ny() - df.index[-1]).total_seconds() / 60
    return df, age_min

def fetch_best(ticker: str):
    """Try 5m → 15m → 1h → 1d. Return (df, age_min, used_interval)."""
    for interval, days in [("5m", 10), ("15m", 30), ("1h", 180), ("1d", 365)]:
        df, age = fetch_intraday(ticker, days=days, interval=interval)
        if df is None or df.empty:
            continue
        if "close" not in df.columns:
            continue
        close = df["close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = pd.to_numeric(close, errors="coerce")
        if close.isna().all():
            continue
        df["close"] = close
        return df, age, interval
    return pd.DataFrame(), 9999, None

# ---------- Sessions + signals ----------
def mark_sessions(df):
    asia = df.between_time("19:00", "23:59")
    london = df.between_time("03:00", "07:00")
    return asia, london

def detect_sweep_choch(df):
    """Detect a simple SMC-style sweep → CHOCH confirmation."""
    if len(df) < 50:
        return "WAIT", "Not enough candles yet."
    recent = df.tail(30)
    high = float(recent["high"].max())
    low  = float(recent["low"].min())
    last_close = float(df["close"].iloc[-1])
    last_high  = float(df["high"].iloc[-1])
    last_low   = float(df["low"].iloc[-1])

    # Sweep above highs + reject
    if (last_close < high * 0.999) and (last_high >= high):
        return "SELL", "Sweep above highs → CHOCH down."
    # Sweep below lows + reject
    if (last_close > low * 1.001) and (last_low <= low):
        return "BUY", "Sweep below lows → CHOCH up."
    return "WAIT", "No liquidity sweep detected."

# ---------- Trade plan + exit tracking ----------
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR."""
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def compute_trade_plan(df: pd.DataFrame, action: str):
    """
    Entry = last close; SL = 0.8*ATR; TP1 = 1.5R; TP2 = 3R; BE trigger = +1R.
    Returns dict or None if WAIT/invalid.
    """
    if action not in ("BUY", "SELL") or len(df) < 20:
        return None
    recent = df.tail(200)
    a = atr(recent, period=14).iloc[-1]
    if not np.isfinite(a) or a <= 0:
        return None

    entry = float(recent["close"].iloc[-1])
    R = float(a * 0.8)

    if action == "BUY":
        sl, tp1, tp2, be, side = entry - R, entry + 1.5*R, entry + 3.0*R, entry + 1.0*R, "long"
    else:
        sl, tp1, tp2, be, side = entry + R, entry - 1.5*R, entry - 3.0*R, entry - 1.0*R, "short"

    return {
        "side": side,
        "entry": round(entry, 2),
        "stop_loss": round(sl, 2),
        "tp1 (1.5R)": round(tp1, 2),
        "tp2 (3R)": round(tp2, 2),
        "breakeven trigger (+1R)": round(be, 2),
        "R (risk unit)": round(R, 2),
    }

def check_trade_progress(df: pd.DataFrame, plan: dict) -> str:
    """Return a message describing current exit state vs TP1/TP2/SL."""
    last_price = float(df["close"].iloc[-1])
    side = plan["side"]
    tp1 = plan["tp1 (1.5R)"]
    tp2 = plan["tp2 (3R)"]
    sl  = plan["stop_loss"]

    if side == "long":
        if last_price >= tp2: return "TP2 hit — close remaining position ✅"
        if last_price >= tp1: return "TP1 hit — partials + move stop to breakeven 🟢"
        if last_price <= sl:  return "Stop Loss hit — exit trade 🔴"
    else:
        if last_price <= tp2: return "TP2 hit — close remaining position ✅"
        if last_price <= tp1: return "TP1 hit — partials + move stop to breakeven 🟢"
        if last_price >= sl:  return "Stop Loss hit — exit trade 🔴"
    return "Holding — no exit yet ⏳"

# ---------- Chart ----------
def plot_chart(df, asia, london, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"
    ))
    for sess, color in [(asia, "rgba(0,255,255,0.08)"), (london, "rgba(255,255,0,0.08)")]:
        if not sess.empty:
            fig.add_vrect(x0=sess.index[0], x1=sess.index[-1],
                          fillcolor=color, layer="below", line_width=0)
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

# ---------- UI ----------
st.title("📊 TJR NY Session Agent (5-Min Charts)")
st.caption("Analyzes S&P 500 and Nasdaq 100 during NY session using TJR Smart Money Concepts.")

symbols = {"S&P 500": "ES=F", "Nasdaq 100": "NQ=F"}

for label, ticker in symbols.items():
    st.subheader(label)

    # Market-hours banner
    if futures_closed_now():
        st.info("Markets are closed (CME equity futures paused). New 5-min candles resume Sunday 6:00 pm ET.")

    # Fetch with fallbacks (5m → 15m → 1h → 1d)
    df, age, used_interval = fetch_best(ticker)
    if df.empty:
        st.warning(f"No data available for {label}. Try again during market hours.")
        st.divider()
        continue

    # Freshness
    if used_interval in ("5m", "15m", "1h"):
        freshness = f"Data age: {age:.1f} min"
        if age > 10:
            st.error(f"{freshness} (stale)")
        elif age > 3:
            st.warning(f"{freshness} (slightly delayed)")
        else:
            st.success(f"{freshness} (live)")
    else:
        st.info("Showing daily candles (intraday unavailable).")

    # Sessions
    if used_interval in ("5m", "15m", "1h"):
        asia, london = mark_sessions(df)
    else:
        asia, london = pd.DataFrame(), pd.DataFrame()

    # Signal
    action, reason = detect_sweep_choch(df)

    # Trade plan + live exit status
    plan = compute_trade_plan(df, action)
    if plan:
        st.markdown("#### Trade Plan (ATR-based)")
        st.dataframe(pd.DataFrame([plan]))

        status_msg = check_trade_progress(df, plan)
        key = f"status_{label}"
        prev = st.session_state.get(key)
        if status_msg != prev:
            # toast if available; fallback to banner styles
            try:
                st.toast(status_msg)
            except Exception:
                if "Stop Loss" in status_msg:
                    st.error(status_msg)
                elif "TP" in status_msg:
                    st.success(status_msg)
                else:
                    st.info(status_msg)
            st.session_state[key] = status_msg
        st.caption(f"Exit status: {status_msg}")

    # Chart + details
    plot_chart(df.tail(400), asia, london, f"{label} — {used_interval.upper()} View")
    st.markdown(f"### Signal: {action}")
    st.caption(reason)
    st.caption(f"Last update: {now_ny().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    st.divider()

st.info("✅ 5m/15m/1h intraday via Yahoo Finance. Use during NY session for best results.")

