import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, time
import pytz
from streamlit_autorefresh import st_autorefresh
import requests

# --------------------- Streamlit Setup ---------------------
st.set_page_config(page_title="TJR NY Session Agent", layout="wide")
st_autorefresh(interval=30_000, key="tjr_autorefresh_30s")

# --------------------- Timezone ---------------------
NY_TZ = pytz.timezone("America/New_York")
def now_ny(): return datetime.now(NY_TZ)

# --------------------- Market Hours ---------------------
def futures_closed_now(dt=None):
    if dt is None: dt = now_ny()
    wd, t = dt.weekday(), dt.time()
    if time(16, 0) <= t < time(17, 0): return True
    if wd == 4 and t >= time(17, 0): return True
    if wd == 5: return True
    if wd == 6 and t < time(18, 0): return True
    return False

# --------------------- Fetch OHLC ---------------------
def _sanitize_ohlc(raw):
    if raw is None or raw.empty: return pd.DataFrame()
    df = raw.copy()
    if not df.index.tz: df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(NY_TZ)
    df.columns = [str(c).lower() for c in df.columns]
    try:
        out = pd.DataFrame({
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce")
        }, index=df.index)
        return out.dropna()
    except KeyError:
        return pd.DataFrame()

def fetch_intraday(symbol, days=3, interval="5m"):
    prepost = not (time(9, 30) <= now_ny().time() <= time(16, 0))
    try:
        raw = yf.download(symbol, period=f"{days}d", interval=interval, prepost=prepost, progress=False)
    except Exception:
        return pd.DataFrame(), 9999
    df = _sanitize_ohlc(raw)
    age = (now_ny() - df.index[-1]).total_seconds() / 60.0 if not df.empty else 9999
    return df, age

def fetch_best(ticker):
    for interval, days in [("5m", 5), ("15m", 20), ("1h", 60), ("1d", 365)]:
        df, age = fetch_intraday(ticker, days, interval)
        if not df.empty: return df, age, interval
    return pd.DataFrame(), 9999, None

# --------------------- Pattern Detection ---------------------
def detect_candle_patterns(df):
    if len(df) < 2:
        return []
    last = df.iloc[-2]
    curr = df.iloc[-1]
    patterns = []

    if last["close"] < last["open"] and curr["close"] > curr["open"] and curr["close"] > last["open"] and curr["open"] < last["close"]:
        patterns.append("Bullish Engulfing")
    if last["close"] > last["open"] and curr["close"] < curr["open"] and curr["open"] > last["close"] and curr["close"] < last["open"]:
        patterns.append("Bearish Engulfing")
    if curr["close"] > curr["open"] and (curr["close"] - curr["open"]) < (curr["high"] - curr["low"]) * 0.3:
        patterns.append("Doji")

    return patterns

# --------------------- Trade Plan Logic ---------------------
def atr(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def build_trade_plan(df, signal):
    if not signal or len(df) < 30: return None
    recent = df.tail(200)
    a = atr(recent, 14).iloc[-1]
    entry = float(recent["close"].iloc[-1])
    R = float(a * 0.8)
    if "Bullish" in signal:
        return {"side": "long", "entry": round(entry, 2), "stop_loss": round(entry - R, 2),
                "tp1 (1.5R)": round(entry + 1.5 * R, 2), "tp2 (3R)": round(entry + 3.0 * R, 2),
                "breakeven": round(entry + R, 2), "R": round(R, 2)}
    elif "Bearish" in signal:
        return {"side": "short", "entry": round(entry, 2), "stop_loss": round(entry + R, 2),
                "tp1 (1.5R)": round(entry - 1.5 * R, 2), "tp2 (3R)": round(entry - 3.0 * R, 2),
                "breakeven": round(entry - R, 2), "R": round(R, 2)}
    return None

def check_trade_status(df, plan):
    price = float(df["close"].iloc[-1])
    if plan["side"] == "long":
        if price >= plan["tp2 (3R)"]: return "TP2 hit ✅"
        if price >= plan["tp1 (1.5R)"]: return "TP1 hit 🟢"
        if price <= plan["stop_loss"]: return "Stop hit 🔴"
    else:
        if price <= plan["tp2 (3R)"]: return "TP2 hit ✅"
        if price <= plan["tp1 (1.5R)"]: return "TP1 hit 🟢"
        if price >= plan["stop_loss"]: return "Stop hit 🔴"
    return "Active"

# --------------------- Discord ---------------------
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1431788585249669211/c4IBMH2qChcZbek-TJSw00k7XjBIbeUAvUE30TUEfilr76ENCh5TUh5nZ6gM_HlETvCC"
def send_discord(content):
    if DISCORD_WEBHOOK_URL:
        try: requests.post(DISCORD_WEBHOOK_URL, json={"content": content}, timeout=8)
        except Exception as e: st.warning(f"Discord failed: {e}")

# --------------------- Streamlit UI ---------------------
st.title("TJR NY Session Master Agent — Smart AI Pattern Recognition")
st.caption("Scanning SPY & QQQ all session long.")

assets = {"S&P 500": "SPY", "Nasdaq 100": "QQQ"}
if "last_alerts" not in st.session_state:
    st.session_state.last_alerts = {}

for name, symbol in assets.items():
    st.subheader(name)
    df, age, interval = fetch_best(symbol)
    st.caption(f"DEBUG: {symbol} | Interval: {interval} | Age: {age:.1f} min | Rows: {len(df)}")

    if df.empty:
        st.warning("⚠️ No chart data.")
        continue

    patterns = detect_candle_patterns(df)
    if patterns:
        plan = build_trade_plan(df, ", ".join(patterns))
        if plan:
            msg = (
                f"📈 {name} Signal\n"
                f"Pattern: {', '.join(patterns)}\n"
                f"Entry: {plan['entry']} | SL: {plan['stop_loss']} | TP1: {plan['tp1 (1.5R)']} | TP2: {plan['tp2 (3R)']}"
            )
            if st.session_state.last_alerts.get(name) != msg:
                send_discord(msg)
                st.session_state.last_alerts[name] = msg
            st.write(msg)
            st.dataframe(pd.DataFrame([plan]))
            st.success(check_trade_status(df, plan))

    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"]
    )])
    fig.update_layout(title=f"{name} — {interval} Chart", height=400)
    st.plotly_chart(fig, use_container_width=True)

if futures_closed_now():
    st.warning("🔴 Market Closed — signals paused.")
else:
    st.success("🟢 Market Open — AI scanning for patterns and trades.")
