import textwrap

# ==============================
#  Bootstrap TJR NY Session AI Agent (v2 - Clean)
# ==============================

# Create .replit or Codespace run configuration
replit_cfg = textwrap.dedent("""
modules = ["python-3.12"]

[nix]
channel = "stable-25_05"

[deployment]
deploymentTarget = "autoscale"
run = ["python", "-m", "streamlit", "run", "app.py", "--server.port", "3000", "--server.address", "0.0.0.0"]
""")
open(".replit", "w").write(replit_cfg)

# Create requirements.txt
requirements = textwrap.dedent("""
streamlit
yfinance
pandas
numpy
plotly
pytz
python-dateutil
requests
""")
open("requirements.txt", "w").write(requirements)

# Create app.py
app_code = textwrap.dedent("""
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta, time
import pytz

st.set_page_config(page_title="TJR NY Session Agent", layout="wide")

NY_TZ = pytz.timezone("America/New_York")

def now_ny():
    return datetime.now(NY_TZ)

def fetch_intraday(symbol, days=3, interval="5m"):
    prepost = not (time(9,30) <= now_ny().time() <= time(16,0))
    df = yf.download(symbol, period=f"{days}d", interval=interval, prepost=prepost, progress=False)
    if df is None or df.empty:
        return pd.DataFrame(), 9999
    if not df.index.tz:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(NY_TZ)
    df.rename(columns=str.lower, inplace=True)
    age_min = (now_ny() - df.index[-1]).total_seconds() / 60
    return df, age_min

def mark_sessions(df):
    asia = df.between_time("19:00", "23:59")
    london = df.between_time("03:00", "07:00")
    return asia, london

def detect_sweep_choch(df):
    if len(df) < 50:
        return "WAIT", "Not enough candles yet."
    recent = df.tail(30)
    high = float(recent['high'].max())
    low = float(recent['low'].min())
    last_close = float(df['close'].iloc[-1])
    last_high = float(df['high'].iloc[-1])
    last_low = float(df['low'].iloc[-1])
    if (last_close < high * 0.999) and (last_high >= high):
        return "SELL", "Sweep above highs → CHOCH down."
    elif (last_close > low * 1.001) and (last_low <= low):
        return "BUY", "Sweep below lows → CHOCH up."
    else:
        return "WAIT", "No liquidity sweep detected."

def plot_chart(df, asia, london, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                                 low=df['low'], close=df['close'], name="Price"))
    for sess, color in [(asia, "rgba(0,255,255,0.08)"), (london, "rgba(255,255,0,0.08)")]:
        if not sess.empty:
            fig.add_vrect(x0=sess.index[0], x1=sess.index[-1],
                          fillcolor=color, layer="below", line_width=0)
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

# ---- UI ----
st.title("📊 TJR NY Session Agent (5-Min Charts)")
st.caption("Analyzes S&P 500 and Nasdaq 100 during NY session using TJR Smart Money Concepts.")

symbols = {"S&P 500": "ES=F", "Nasdaq 100": "NQ=F"}

for label, ticker in symbols.items():
    st.subheader(label)
    df, age = fetch_intraday(ticker, days=3, interval="5m")
    if df.empty:
        st.warning(f"No data for {label}.")
        continue
    freshness = f"Data age: {age:.1f} min"
    if age > 10:
        st.error(f"{freshness} (stale)")
    elif age > 3:
        st.warning(f"{freshness} (slightly delayed)")
    else:
        st.success(f"{freshness} (live)")
    asia, london = mark_sessions(df)
    action, reason = detect_sweep_choch(df)
    plot_chart(df.tail(400), asia, london, f"{label} — 5-Min View")
    st.markdown(f"### Signal: {action}")
    st.caption(reason)
    st.caption(f"Last update: {now_ny().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    st.divider()

st.info("✅ Live 5-min data via Yahoo Finance. Use during NY session for best results.")
""")
open("app.py", "w").write(app_code)

print("✅ Files created successfully!")
print("Next steps:")
print("1️⃣ pip install -r requirements.txt")
print("2️⃣ python -m streamlit run app.py --server.port=3000 --server.address=0.0.0.0")
