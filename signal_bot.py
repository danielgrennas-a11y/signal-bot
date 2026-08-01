#!/usr/bin/env python3
"""
signal_bot.py — Technical analysis signal scanner for OMXS30 & Nasdaq,
with optional Telegram push notifications.

WHAT THIS IS
    A read-only analysis tool. It downloads price history, computes a
    standard set of technical indicators (moving average crossovers, RSI,
    MACD) plus rule-based candlestick pattern detection, and prints/plots
    buy and sell SIGNALS based on those rules. In --watch mode it polls
    periodically and pings your phone via Telegram when the signal changes.

WHAT THIS IS NOT
    - It does not place trades or connect to any broker.
    - It is not financial advice. Technical signals are pattern-matching
      on historical price action; they are not a guarantee of future
      performance. Use this as one input among many, size positions
      responsibly, and consider talking to a licensed financial advisor
      before making investment decisions.

----------------------------------------------------------------------
SETUP — run this in a Terminal, NOT in Thonny
----------------------------------------------------------------------
Thonny bundles its own Python interpreter, separate from your Mac's
normal Python. Packages installed via `pip` in Terminal aren't visible
to Thonny, which is why it couldn't find yfinance/pandas. Use Terminal
(Applications > Utilities > Terminal) instead:

    cd ~/Desktop                      # or wherever you saved the file
    python3 -m venv venv              # create an isolated environment (once)
    source venv/bin/activate          # activate it (every new terminal session)
    pip install yfinance pandas numpy matplotlib requests

    python3 signal_bot.py --market omxs30 --interval 1d

----------------------------------------------------------------------
TELEGRAM NOTIFICATIONS — one-time setup (~2 minutes)
----------------------------------------------------------------------
1. In Telegram, message @BotFather -> /newbot -> follow the prompts.
   BotFather gives you a token that looks like:
       123456789:AAExampleTokenNotReal
2. Start a chat with your new bot (search its username, hit Start / send it any message).
3. Open this URL in a browser, replacing <TOKEN>:
       https://api.telegram.org/bot<TOKEN>/getUpdates
   Find "chat":{"id": 123456789, ...} in the response — that number is your chat ID.
4. Save both values as environment variables (Terminal):
       export TG_BOT_TOKEN="123456789:AAExampleTokenNotReal"
       export TG_CHAT_ID="123456789"
   (Add those two lines to ~/.zshrc so they persist across terminal sessions.)

----------------------------------------------------------------------
USAGE
----------------------------------------------------------------------
One-off check, prints signal, sends a Telegram message if BUY/SELL:
    python3 signal_bot.py --market omxs30 --interval 1d --notify

Keep running and check every 15 minutes, only notifying on a NEW signal
(i.e. it won't spam you every 15 min while still in the same BUY zone):
    python3 signal_bot.py --market nasdaq --interval 1h --watch --poll-minutes 15

Run in the background so it survives closing the terminal (see the
launchd section at the bottom of this file for a "start on login" setup).
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MARKET_TICKERS = {
    "omxs30": "^OMX",
    "nasdaq": "^IXIC",
    "nasdaq100": "^NDX",
}

DEFAULT_PERIOD = {
    "1d": "2y",
    "1h": "60d",
}

STATE_FILE = Path.home() / ".signal_bot_state.json"


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        sys.exit(
            "Missing dependency 'yfinance'. In Terminal (not Thonny):\n"
            "    pip install yfinance pandas numpy matplotlib requests"
        )

    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        sys.exit(f"No data returned for ticker '{ticker}' (interval={interval}, period={period}). "
                  f"Check the symbol on finance.yahoo.com.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.title)
    df.index.name = "Date"
    return df


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------

def add_moving_averages(df, fast=20, slow=50):
    df[f"SMA{fast}"] = df["Close"].rolling(fast).mean()
    df[f"SMA{slow}"] = df["Close"].rolling(slow).mean()
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    return df


def add_rsi(df, length=14):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)
    return df


def add_macd(df):
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    return df


def add_atr(df, length=14):
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    return df


def add_pivots(df, lookback=5):
    """
    Mark pivot highs/lows: a bar whose High (Low) is the max (min) within a
    window of `lookback` bars on both sides — a classic swing high/low used
    to read off support and resistance levels.
    """
    df["pivot_high"] = df["High"] == df["High"].rolling(2 * lookback + 1, center=True).max()
    df["pivot_low"] = df["Low"] == df["Low"].rolling(2 * lookback + 1, center=True).min()
    return df


def nearest_levels(df, price, lookback_bars=120):
    """
    Look back over the recent history for the nearest pivot-high (resistance,
    above price) and pivot-low (support, below price). Returns (support, resistance),
    either of which may be None if nothing qualifying was found in range.
    """
    recent = df.tail(lookback_bars)
    highs_above = recent.loc[recent["pivot_high"] & (recent["High"] > price), "High"]
    lows_below = recent.loc[recent["pivot_low"] & (recent["Low"] < price), "Low"]
    resistance = highs_above.min() if not highs_above.empty else None
    support = lows_below.max() if not lows_below.empty else None
    return support, resistance


def build_trade_setup(df, row):
    """
    Turn the current price + ATR + nearby pivots into a plain-language
    entry zone / stop-loss / target, the way a discretionary trader would
    read off a chart. This is a heuristic, not a guarantee — see the
    disclaimer wherever this is surfaced.
    """
    price = row["Close"]
    atr = row.get("ATR", np.nan)
    support, resistance = nearest_levels(df, price)

    if pd.isna(atr):
        return None

    if row["signal"] == "BUY":
        entry_low = price - 0.3 * atr
        entry_high = price + 0.1 * atr
        stop = (support - 0.25 * atr) if support is not None else (price - 1.5 * atr)
        target = resistance if resistance is not None else (price + 2.0 * atr)
    elif row["signal"] == "SELL":
        entry_low = price - 0.1 * atr
        entry_high = price + 0.3 * atr
        stop = (resistance + 0.25 * atr) if resistance is not None else (price + 1.5 * atr)
        target = support if support is not None else (price - 2.0 * atr)
    else:
        return None

    return {
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop": stop,
        "target": target,
        "support": support,
        "resistance": resistance,
    }


def add_bollinger(df, length=20, num_std=2.0):
    mid = df["Close"].rolling(length).mean()
    std = df["Close"].rolling(length).std()
    df["BB_mid"] = mid
    df["BB_upper"] = mid + num_std * std
    df["BB_lower"] = mid - num_std * std
    return df


# --------------------------------------------------------------------------
# Candlestick pattern detection (rule-based, no external TA-lib needed)
# --------------------------------------------------------------------------

def body(df):
    return (df["Close"] - df["Open"]).abs()


def candle_range(df):
    return (df["High"] - df["Low"]).replace(0, np.nan)


def add_candlestick_patterns(df):
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    b = body(df)
    rng = candle_range(df)
    upper_wick = h - df[["Open", "Close"]].max(axis=1)
    lower_wick = df[["Open", "Close"]].min(axis=1) - l

    df["doji"] = (b / rng < 0.1)

    df["hammer"] = (
        (lower_wick > 2 * b)
        & (upper_wick < b)
        & (b / rng < 0.35)
    )

    df["shooting_star"] = (
        (upper_wick > 2 * b)
        & (lower_wick < b)
        & (b / rng < 0.35)
    )

    prev_o, prev_c = o.shift(1), c.shift(1)

    df["bullish_engulfing"] = (
        (prev_c < prev_o) & (c > o) & (c >= prev_o) & (o <= prev_c)
    )

    df["bearish_engulfing"] = (
        (prev_c > prev_o) & (c < o) & (o >= prev_c) & (c <= prev_o)
    )

    return df


# --------------------------------------------------------------------------
# Signal logic
# --------------------------------------------------------------------------

def add_signals(df, fast=20, slow=50):
    score = pd.Series(0, index=df.index)

    ma_bull = df[f"SMA{fast}"] > df[f"SMA{slow}"]
    ma_bull_cross = ma_bull & (~ma_bull.shift(1).fillna(False))
    ma_bear_cross = (~ma_bull) & (ma_bull.shift(1).fillna(False))
    score += ma_bull_cross.astype(int) * 1
    score -= ma_bear_cross.astype(int) * 1

    score += (df["RSI"] < 30).astype(int) * 1
    score -= (df["RSI"] > 70).astype(int) * 1

    macd_bull = df["MACD"] > df["MACD_signal"]
    macd_bull_cross = macd_bull & (~macd_bull.shift(1).fillna(False))
    macd_bear_cross = (~macd_bull) & (macd_bull.shift(1).fillna(False))
    score += macd_bull_cross.astype(int) * 1
    score -= macd_bear_cross.astype(int) * 1

    score += (df["hammer"] | df["bullish_engulfing"]).astype(int) * 1
    score -= (df["shooting_star"] | df["bearish_engulfing"]).astype(int) * 1

    df["score"] = score
    df["signal"] = np.select(
        [df["score"] >= 2, df["score"] <= -2],
        ["BUY", "SELL"],
        default="HOLD",
    )
    return df


def analyze(ticker, interval, period, fast=20, slow=50):
    df = load_data(ticker, interval, period)
    df = add_moving_averages(df, fast, slow)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_atr(df)
    df = add_pivots(df)
    df = add_bollinger(df)
    df = add_candlestick_patterns(df)
    df = add_signals(df, fast, slow)
    return df


# --------------------------------------------------------------------------
# Telegram
# --------------------------------------------------------------------------

def send_telegram(token: str, chat_id: str, message: str) -> bool:
    import requests
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=10)
        if resp.status_code != 200:
            print(f"[telegram] failed ({resp.status_code}): {resp.text}")
            return False
        return True
    except Exception as e:
        print(f"[telegram] error: {e}")
        return False


def format_alert(ticker: str, df: pd.DataFrame, row) -> str:
    patterns = [name for name in
                ["doji", "hammer", "shooting_star", "bullish_engulfing", "bearish_engulfing"]
                if row.get(name, False)]
    lines = [
        f"{ticker}: {row['signal']} signal",
        f"Close: {row['Close']:.2f}",
        f"RSI(14): {row['RSI']:.1f}",
        f"SMA20/50: {row['SMA20']:.2f} / {row['SMA50']:.2f}",
    ]
    if patterns:
        lines.append(f"Candle pattern: {', '.join(patterns)}")

    setup = build_trade_setup(df, row)
    if setup:
        lines.append("")
        lines.append(f"Entry zone: {setup['entry_low']:.2f} - {setup['entry_high']:.2f}")
        lines.append(f"Stop-loss: {setup['stop']:.2f}")
        label = "Resistance/target" if row["signal"] == "BUY" else "Support/target"
        lines.append(f"{label}: {setup['target']:.2f}")

    lines.append("(Technical signal only — not financial advice.)")
    return "\n".join(lines)


def format_daily_summary(ticker: str, df: pd.DataFrame, row) -> str:
    patterns = [name for name in
                ["doji", "hammer", "shooting_star", "bullish_engulfing", "bearish_engulfing"]
                if row.get(name, False)]
    lines = [
        f"Daily update — {ticker}",
        f"Signal: {row['signal']}  (score {int(row['score'])})",
        f"Close: {row['Close']:.2f}",
        f"RSI(14): {row['RSI']:.1f}",
        f"SMA20/50: {row['SMA20']:.2f} / {row['SMA50']:.2f}",
        f"MACD/signal: {row['MACD']:.2f} / {row['MACD_signal']:.2f}",
    ]
    if patterns:
        lines.append(f"Candle pattern: {', '.join(patterns)}")

    support, resistance = nearest_levels(df, row["Close"])
    if support is not None or resistance is not None:
        lines.append("")
        if support is not None:
            lines.append(f"Nearest support: {support:.2f}")
        if resistance is not None:
            lines.append(f"Nearest resistance: {resistance:.2f}")

    setup = build_trade_setup(df, row)
    if setup:
        lines.append(f"Entry zone: {setup['entry_low']:.2f} - {setup['entry_high']:.2f}")
        lines.append(f"Stop-loss: {setup['stop']:.2f}")

    lines.append("(Technical signal only — not financial advice.)")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# State (so --watch only notifies on a NEW signal, not every poll)
# --------------------------------------------------------------------------

def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state))


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_summary(df, ticker, n_recent=10):
    last = df.iloc[-1]
    print("=" * 60)
    print(f"  {ticker} — latest close: {last['Close']:.2f}  ({df.index[-1].strftime('%Y-%m-%d %H:%M')})")
    print("=" * 60)
    print(f"  Current signal : {last['signal']}   (score {int(last['score'])})")
    print(f"  RSI(14)        : {last['RSI']:.1f}")
    print(f"  SMA20 / SMA50  : {last['SMA20']:.2f} / {last['SMA50']:.2f}")
    print(f"  MACD / signal  : {last['MACD']:.2f} / {last['MACD_signal']:.2f}")
    patterns = [name for name in
                ["doji", "hammer", "shooting_star", "bullish_engulfing", "bearish_engulfing"]
                if last.get(name, False)]
    print(f"  Candle pattern : {', '.join(patterns) if patterns else 'none'}")
    print()

    recent = df[df["signal"] != "HOLD"].tail(n_recent)
    if recent.empty:
        print("No BUY/SELL signals in the loaded history.")
    else:
        print(f"Last {len(recent)} non-HOLD signals:")
        for idx, row in recent.iterrows():
            print(f"  {idx.strftime('%Y-%m-%d %H:%M')}  {row['signal']:<4}  "
                  f"close={row['Close']:.2f}  score={int(row['score'])}")
    print()
    print("Reminder: these are rule-based technical signals, not financial advice.")


def plot_chart(df, ticker):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(13, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )

    width = (df.index[1] - df.index[0]) * 0.6 if len(df) > 1 else pd.Timedelta(hours=12)
    for idx, row in df.iterrows():
        color = "#1f9e5a" if row["Close"] >= row["Open"] else "#d64545"
        ax1.plot([idx, idx], [row["Low"], row["High"]], color=color, linewidth=0.8)
        ax1.add_patch(plt.Rectangle(
            (mdates.date2num(idx) - width.total_seconds() / (2 * 86400), min(row["Open"], row["Close"])),
            width.total_seconds() / 86400,
            max(abs(row["Close"] - row["Open"]), 1e-6),
            color=color,
        ))

    ax1.plot(df.index, df["SMA20"], label="SMA20", color="#3b82f6", linewidth=1)
    ax1.plot(df.index, df["SMA50"], label="SMA50", color="#f59e0b", linewidth=1)
    ax1.plot(df.index, df["BB_upper"], color="#9ca3af", linewidth=0.7, linestyle="--")
    ax1.plot(df.index, df["BB_lower"], color="#9ca3af", linewidth=0.7, linestyle="--")

    buys = df[df["signal"] == "BUY"]
    sells = df[df["signal"] == "SELL"]
    ax1.scatter(buys.index, buys["Low"] * 0.995, marker="^", color="#1f9e5a", s=80, label="BUY", zorder=5)
    ax1.scatter(sells.index, sells["High"] * 1.005, marker="v", color="#d64545", s=80, label="SELL", zorder=5)

    ax1.set_title(f"{ticker} — price, moving averages & signals")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.2)

    ax2.plot(df.index, df["RSI"], color="#7c3aed", linewidth=1)
    ax2.axhline(70, color="#d64545", linestyle="--", linewidth=0.7)
    ax2.axhline(30, color="#1f9e5a", linestyle="--", linewidth=0.7)
    ax2.set_ylabel("RSI")
    ax2.grid(alpha=0.2)

    ax3.plot(df.index, df["MACD"], color="#3b82f6", linewidth=1, label="MACD")
    ax3.plot(df.index, df["MACD_signal"], color="#f59e0b", linewidth=1, label="Signal")
    ax3.bar(df.index, df["MACD_hist"], color="#9ca3af", width=width, alpha=0.5)
    ax3.set_ylabel("MACD")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(alpha=0.2)

    fig.autofmt_xdate()
    plt.tight_layout()
    out_path = "signal_chart.png"
    plt.savefig(out_path, dpi=140)
    print(f"Chart saved to {out_path}")
    plt.show()


# --------------------------------------------------------------------------
# Run modes
# --------------------------------------------------------------------------

def run_once(args, token, chat_id):
    df = analyze(args.ticker, args.interval, args.period, args.fast, args.slow)
    print_summary(df, args.ticker)

    last = df.iloc[-1]
    last_ts = df.index[-1].isoformat()
    key = f"{args.ticker}_{args.interval}"

    if args.notify:
        if not (token and chat_id):
            print("--notify was set but TG_BOT_TOKEN / TG_CHAT_ID are not configured. "
                  "See the setup instructions at the top of this file.")
        elif args.daily:
            # Always send a status update, regardless of signal — this run
            # itself is only scheduled once a day, so no dedup needed.
            ok = send_telegram(token, chat_id, format_daily_summary(args.ticker, df, last))
            print("Daily summary sent." if ok else "Daily summary failed — check token/chat id.")
        elif last["signal"] != "HOLD":
            # Only alert if this is a genuinely NEW signal since last run —
            # otherwise a repeating cron job would re-send the same alert
            # every single time it runs while the signal stays active.
            state = load_state()
            prev = state.get(key, {})
            is_new_signal = (prev.get("timestamp") != last_ts or prev.get("signal") != last["signal"])
            if is_new_signal:
                ok = send_telegram(token, chat_id, format_alert(args.ticker, df, last))
                print("Telegram alert sent." if ok else "Telegram alert failed — check token/chat id.")
                state[key] = {"timestamp": last_ts, "signal": last["signal"]}
                save_state(state)
            else:
                print(f"Signal is still {last['signal']} from a previous run — not re-sending.")

    if not args.no_plot:
        plot_chart(df, args.ticker)


def run_watch(args, token, chat_id):
    if not (token and chat_id):
        sys.exit("--watch requires Telegram credentials. Set TG_BOT_TOKEN and TG_CHAT_ID "
                  "(see the setup instructions at the top of this file).")

    state = load_state()
    key = f"{args.ticker}_{args.interval}"
    mode_desc = "daily summary + signal alerts" if args.daily else "signal alerts only"
    print(f"Watching {args.ticker} ({args.interval}) every {args.poll_minutes} min — {mode_desc}. Ctrl+C to stop.")

    while True:
        try:
            df = analyze(args.ticker, args.interval, args.period, args.fast, args.slow)
            last = df.iloc[-1]
            last_ts = df.index[-1].isoformat()
            today = pd.Timestamp.now().strftime("%Y-%m-%d")
            print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] "
                  f"{args.ticker} close={last['Close']:.2f} signal={last['signal']}")

            prev = state.get(key, {})

            # Daily digest: once per calendar day, regardless of signal
            if args.daily and prev.get("last_daily_sent") != today:
                ok = send_telegram(token, chat_id, format_daily_summary(args.ticker, df, last))
                print("  -> Daily summary sent." if ok else "  -> Daily summary failed.")
                prev["last_daily_sent"] = today
                state[key] = prev
                save_state(state)

            # Immediate alert: only on a genuinely new BUY/SELL signal
            is_new_signal = (
                last["signal"] != "HOLD"
                and (prev.get("timestamp") != last_ts or prev.get("signal") != last["signal"])
            )
            if is_new_signal:
                ok = send_telegram(token, chat_id, format_alert(args.ticker, df, last))
                print("  -> Signal alert sent." if ok else "  -> Signal alert failed.")
                prev["timestamp"] = last_ts
                prev["signal"] = last["signal"]
                state[key] = prev
                save_state(state)

        except SystemExit as e:
            print(f"  (data fetch problem, will retry next cycle: {e})")
        except Exception as e:
            print(f"  (unexpected error, will retry next cycle: {e})")

        time.sleep(args.poll_minutes * 60)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Technical analysis signal scanner with Telegram alerts")
    parser.add_argument("--market", choices=MARKET_TICKERS.keys(), default="omxs30")
    parser.add_argument("--ticker", default=None, help="Override with any Yahoo Finance ticker")
    parser.add_argument("--interval", choices=["1d", "1h"], default="1d")
    parser.add_argument("--period", default=None, help="e.g. 6mo, 1y, 2y, 60d")
    parser.add_argument("--fast", type=int, default=20)
    parser.add_argument("--slow", type=int, default=50)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--notify", action="store_true", help="Send a Telegram message if the latest signal is BUY/SELL")
    parser.add_argument("--watch", action="store_true", help="Keep running, poll periodically, notify only on new signals")
    parser.add_argument("--daily", action="store_true", help="With --watch: also send one status summary per calendar day, regardless of signal")
    parser.add_argument("--poll-minutes", type=int, default=15)
    parser.add_argument("--telegram-token", default=None, help="Overrides TG_BOT_TOKEN env var")
    parser.add_argument("--telegram-chat-id", default=None, help="Overrides TG_CHAT_ID env var")
    parser.add_argument("--state-file", default=None,
                         help="Path to the dedup state file. Default: ~/.signal_bot_state.json. "
                              "For scheduled runs (e.g. GitHub Actions) point this at a file inside "
                              "the repo so it can be committed back between runs.")
    args = parser.parse_args()

    args.ticker = args.ticker or MARKET_TICKERS[args.market]
    args.period = args.period or DEFAULT_PERIOD[args.interval]

    if args.state_file:
        global STATE_FILE
        STATE_FILE = Path(args.state_file)

    token = args.telegram_token or os.environ.get("TG_BOT_TOKEN")
    chat_id = args.telegram_chat_id or os.environ.get("TG_CHAT_ID")

    if args.watch:
        run_watch(args, token, chat_id)
    else:
        run_once(args, token, chat_id)


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# OPTIONAL: run automatically in the background on your Mac (launchd)
# --------------------------------------------------------------------------
# Create ~/Library/LaunchAgents/com.signalbot.omxs30.plist with:
#
# <?xml version="1.0" encoding="UTF-8"?>
# <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
#   "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
# <plist version="1.0">
# <dict>
#   <key>Label</key><string>com.signalbot.omxs30</string>
#   <key>ProgramArguments</key>
#   <array>
#     <string>/full/path/to/venv/bin/python3</string>
#     <string>/full/path/to/signal_bot.py</string>
#     <string>--market</string><string>omxs30</string>
#     <string>--interval</string><string>1d</string>
#     <string>--watch</string>
#     <string>--poll-minutes</string><string>15</string>
#     <string>--no-plot</string>
#   </array>
#   <key>EnvironmentVariables</key>
#   <dict>
#     <key>TG_BOT_TOKEN</key><string>YOUR_TOKEN</string>
#     <key>TG_CHAT_ID</key><string>YOUR_CHAT_ID</string>
#   </dict>
#   <key>RunAtLoad</key><true/>
#   <key>KeepAlive</key><true/>
#   <key>StandardOutPath</key><string>/tmp/signalbot.log</string>
#   <key>StandardErrorPath</key><string>/tmp/signalbot.err</string>
# </dict>
# </plist>
#
# Then load it:
#   launchctl load ~/Library/LaunchAgents/com.signalbot.omxs30.plist
# It will now start automatically on login and keep running in the
# background — no terminal window needs to stay open. To stop it:
#   launchctl unload ~/Library/LaunchAgents/com.signalbot.omxs30.plist
