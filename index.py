#!/usr/bin/env python3
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import yfinance as yf
import requests_cache

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.align import Align
from rich.live import Live
from rich.spinner import Spinner
from rich import box

console = Console()

# ---------------------------
# Cache HTTP requests (helps a lot)
# ---------------------------
# Caches Yahoo responses locally for 1 day
requests_cache.install_cache("alphasheet_cache", expire_after=86400)


# ---------------------------
# Helpers
# ---------------------------
def safe_get(d: Dict[str, Any], key: str) -> Optional[Any]:
    v = d.get(key, None)
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    return v

def fmt_money(x: Optional[float], currency: str = "") -> str:
    if x is None:
        return "N/A"
    absx = abs(x)
    if absx >= 1e12: return f"{x/1e12:.2f}T {currency}".strip()
    if absx >= 1e9:  return f"{x/1e9:.2f}B {currency}".strip()
    if absx >= 1e6:  return f"{x/1e6:.2f}M {currency}".strip()
    return f"{x:,.2f} {currency}".strip()

def fmt_num(x: Optional[float], decimals: int = 2) -> str:
    if x is None: return "N/A"
    return f"{x:.{decimals}f}"

def fmt_pct(x: Optional[float], decimals: int = 2) -> str:
    if x is None: return "N/A"
    return f"{x*100:.{decimals}f}%"

def color_pct(x: Optional[float], decimals: int = 2) -> Text:
    if x is None:
        return Text("N/A", style="dim")
    s = f"{x*100:.{decimals}f}%"
    if x > 0:  return Text(s, style="bold green")
    if x < 0:  return Text(s, style="bold red")
    return Text(s, style="bold yellow")

def sparkline(series: pd.Series, width: int = 34) -> str:
    s = series.dropna()
    if s.empty:
        return ""
    if len(s) > width:
        s = s.iloc[-width:]
    mn, mx = float(s.min()), float(s.max())
    if mx - mn < 1e-12:
        return "─" * len(s)
    ticks = "▁▂▃▄▅▆▇█"
    vals = (s - mn) / (mx - mn)
    idx = np.clip((vals * (len(ticks) - 1)).round().astype(int), 0, len(ticks) - 1)
    return "".join(ticks[i] for i in idx)

def max_drawdown(prices: pd.Series) -> Optional[float]:
    p = prices.dropna()
    if p.empty:
        return None
    cm = p.cummax()
    dd = (p / cm) - 1.0
    return float(dd.min())

def realized_vol(returns: pd.Series, window: int = 30) -> Optional[float]:
    r = returns.dropna()
    if len(r) < window:
        return None
    return float(r.iloc[-window:].std() * np.sqrt(252))

def returns_from_prices(prices: pd.Series) -> Dict[str, Optional[float]]:
    p = prices.dropna()
    if len(p) < 3:
        return {"1D": None, "1W": None, "1M": None, "3M": None, "1Y": None}
    horizons = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "1Y": 252}
    out: Dict[str, Optional[float]] = {}
    for k, n in horizons.items():
        out[k] = float(p.iloc[-1] / p.iloc[-(n + 1)] - 1.0) if len(p) > n else None
    return out


# ---------------------------
# Robust Yahoo calls with retries
# ---------------------------
def with_retries(fn, *, tries: int = 5, base_delay: float = 1.0, jitter: float = 0.2):
    last_err = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            # exponential backoff
            delay = base_delay * (2 ** i)
            delay = delay * (1.0 + np.random.uniform(-jitter, jitter))
            time.sleep(max(0.0, delay))
    raise last_err

def download_prices(ticker: str, period: str = "2y") -> pd.Series:
    def _dl():
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
        return df

    df = with_retries(_dl, tries=4, base_delay=0.8)
    if df is None or df.empty:
        return pd.Series(dtype=float)

    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].iloc[:, 0]
    else:
        close = df["Close"]
    close.name = "Close"
    return close

def get_info_safe(ticker: str) -> Dict[str, Any]:
    # Info endpoints are more likely to be rate limited than price history.
    # So: try a couple times, and if it fails, return {} and we still compute price-based KPIs.
    def _info():
        return yf.Ticker(ticker).info or {}

    try:
        return with_retries(_info, tries=2, base_delay=0.7)
    except Exception:
        return {}


@dataclass
class Facts:
    name: str
    ticker: str
    currency: str

    sector: Optional[str]
    industry: Optional[str]
    country: Optional[str]

    price: Optional[float]
    market_cap: Optional[float]
    w52_low: Optional[float]
    w52_high: Optional[float]

    perf: Dict[str, Optional[float]]
    beta: Optional[float]
    vol30: Optional[float]
    mdd1y: Optional[float]

    pe_ttm: Optional[float]
    pe_fwd: Optional[float]
    ps: Optional[float]
    pb: Optional[float]
    ev_ebitda: Optional[float]

    gross_margin: Optional[float]
    op_margin: Optional[float]
    net_margin: Optional[float]
    roe: Optional[float]

    debt_to_equity: Optional[float]
    current_ratio: Optional[float]

    div_yield: Optional[float]
    payout: Optional[float]

    spark: str
    fundamentals_available: bool


def build_factsheet(ticker: str) -> Facts:
    prices = download_prices(ticker, period="2y")
    if prices.dropna().empty:
        raise RuntimeError("No price data returned (ticker may be invalid or data source blocked).")

    info = get_info_safe(ticker)

    currency = safe_get(info, "currency") or ""
    name = safe_get(info, "shortName") or safe_get(info, "longName") or ticker
    last_price = safe_get(info, "currentPrice")
    if last_price is None:
        last_price = float(prices.dropna().iloc[-1])

    prices_1y = prices.dropna().iloc[-252:] if len(prices.dropna()) >= 50 else prices
    rets = prices.pct_change()

    perf = returns_from_prices(prices)

    fundamentals_available = bool(info)  # if info failed, we show N/A but still show price/risk

    return Facts(
        name=name,
        ticker=ticker,
        currency=currency,

        sector=safe_get(info, "sector"),
        industry=safe_get(info, "industry"),
        country=safe_get(info, "country"),

        price=last_price,
        market_cap=safe_get(info, "marketCap"),
        w52_low=safe_get(info, "fiftyTwoWeekLow"),
        w52_high=safe_get(info, "fiftyTwoWeekHigh"),

        perf=perf,
        beta=safe_get(info, "beta"),
        vol30=realized_vol(rets, window=30),
        mdd1y=max_drawdown(prices_1y),

        pe_ttm=safe_get(info, "trailingPE"),
        pe_fwd=safe_get(info, "forwardPE"),
        ps=safe_get(info, "priceToSalesTrailing12Months"),
        pb=safe_get(info, "priceToBook"),
        ev_ebitda=safe_get(info, "enterpriseToEbitda"),

        gross_margin=safe_get(info, "grossMargins"),
        op_margin=safe_get(info, "operatingMargins"),
        net_margin=safe_get(info, "profitMargins"),
        roe=safe_get(info, "returnOnEquity"),

        debt_to_equity=safe_get(info, "debtToEquity"),
        current_ratio=safe_get(info, "currentRatio"),

        div_yield=safe_get(info, "dividendYield"),
        payout=safe_get(info, "payoutRatio"),

        spark=sparkline(prices, width=34),
        fundamentals_available=fundamentals_available,
    )


# ---------------------------
# Rich UI
# ---------------------------
def header_panel(f: Facts) -> Panel:
    title = Text.assemble(("AlphaSheet", "bold cyan"), ("  •  ", "dim"), (f.ticker, "bold white"))
    meta_bits = [x for x in [f.sector, f.industry, f.country] if x]
    meta = Text(" | ".join(meta_bits), style="dim") if meta_bits else Text("—", style="dim")

    price_line = Text.assemble(
        ("Price: ", "dim"),
        (fmt_money(f.price, f.currency), "bold"),
        ("   ", "dim"),
        ("Chart: ", "dim"),
        (f.spark or "N/A", "cyan" if f.spark else "dim"),
    )

    note = Text(
        "Fundamentals limited (rate-limited) — showing price-based KPIs only."
        if not f.fundamentals_available
        else "",
        style="yellow" if not f.fundamentals_available else "dim",
    )

    body = Text.assemble(meta, "\n", price_line, "\n", note)
    return Panel(body, title=title, border_style="cyan", padding=(1, 2))

def table_returns(f: Facts) -> Panel:
    t = Table(box=box.SIMPLE, header_style="bold", expand=True)
    t.add_column("Returns", style="bold")
    for k in ["1D", "1W", "1M", "3M", "1Y"]:
        t.add_column(k, justify="right")
    t.add_row("Total", *(color_pct(f.perf.get(k)) for k in ["1D", "1W", "1M", "3M", "1Y"]))
    return Panel(t, title="Performance", border_style="green", padding=(1, 1))

def table_risk(f: Facts) -> Panel:
    t = Table(box=box.SIMPLE, show_header=False, expand=True)
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")
    t.add_row("Vol (30D, ann.)", color_pct(f.vol30))
    t.add_row("Max Drawdown (≈1Y)", color_pct(f.mdd1y))
    t.add_row("Beta", Text(fmt_num(f.beta), style="bold") if f.beta is not None else Text("N/A", style="dim"))
    return Panel(t, title="Risk", border_style="magenta", padding=(1, 1))

def table_snapshot(f: Facts) -> Panel:
    t = Table(box=box.SIMPLE, show_header=False, expand=True)
    t.add_column("Snapshot", style="bold")
    t.add_column("Value", justify="right")
    t.add_row("Market Cap", Text(fmt_money(f.market_cap, f.currency), style="bold") if f.market_cap else Text("N/A", style="dim"))
    t.add_row("52W Low", Text(fmt_money(f.w52_low, f.currency), style="bold") if f.w52_low else Text("N/A", style="dim"))
    t.add_row("52W High", Text(fmt_money(f.w52_high, f.currency), style="bold") if f.w52_high else Text("N/A", style="dim"))
    return Panel(t, title="Snapshot", border_style="blue", padding=(1, 1))

def table_valuation(f: Facts) -> Panel:
    t = Table(box=box.SIMPLE, show_header=False, expand=True)
    t.add_column("Valuation", style="bold")
    t.add_column("Value", justify="right")
    t.add_row("P/E (TTM)", Text(fmt_num(f.pe_ttm), style="bold") if f.pe_ttm else Text("N/A", style="dim"))
    t.add_row("P/E (Fwd)", Text(fmt_num(f.pe_fwd), style="bold") if f.pe_fwd else Text("N/A", style="dim"))
    t.add_row("P/S", Text(fmt_num(f.ps), style="bold") if f.ps else Text("N/A", style="dim"))
    t.add_row("P/B", Text(fmt_num(f.pb), style="bold") if f.pb else Text("N/A", style="dim"))
    return Panel(t, title="Valuation", border_style="yellow", padding=(1, 1))

def table_quality(f: Facts) -> Panel:
    t = Table(box=box.SIMPLE, show_header=False, expand=True)
    t.add_column("Quality", style="bold")
    t.add_column("Value", justify="right")
    t.add_row("Gross Margin", Text(fmt_pct(f.gross_margin), style="bold") if f.gross_margin else Text("N/A", style="dim"))
    t.add_row("Op Margin", Text(fmt_pct(f.op_margin), style="bold") if f.op_margin else Text("N/A", style="dim"))
    t.add_row("Net Margin", Text(fmt_pct(f.net_margin), style="bold") if f.net_margin else Text("N/A", style="dim"))
    t.add_row("ROE", Text(fmt_pct(f.roe), style="bold") if f.roe else Text("N/A", style="dim"))
    return Panel(t, title="Fundamentals", border_style="white", padding=(1, 1))

def make_layout(f: Facts) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=6),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(Layout(name="left", ratio=2), Layout(name="right", ratio=3))
    layout["left"].split_column(Layout(name="returns", size=7), Layout(name="risk", ratio=1))
    layout["right"].split_column(Layout(name="snap", size=7), Layout(name="val", size=7), Layout(name="qual", ratio=1))

    layout["header"].update(header_panel(f))
    layout["returns"].update(table_returns(f))
    layout["risk"].update(table_risk(f))
    layout["snap"].update(table_snapshot(f))
    layout["val"].update(table_valuation(f))
    layout["qual"].update(table_quality(f))

    footer = Panel(
        Align.center(
            Text.assemble(
                ("Enter ticker ", "bold"),
                ("(AAPL, AIR.PA, ^GSPC). ", "dim"),
                ("Type ", "dim"),
                ("quit", "bold"),
                (" to exit.", "dim"),
            )
        ),
        border_style="dim",
    )
    layout["footer"].update(footer)
    return layout


# ---------------------------
# Main loop
# ---------------------------
def fetch_with_spinner(ticker: str) -> Facts:
    sp = Spinner("dots", text=f"Loading {ticker} (cached + retries)…", style="cyan")
    with Live(Panel(sp, border_style="cyan", padding=(1, 2)), console=console, refresh_per_second=30):
        return build_factsheet(ticker)

def main():
    console.clear()
    console.print(
        Panel(
            Align.center(Text("AlphaSheet — Terminal Stock Factsheets\n\nTry: AAPL • AIR.PA • MC.PA • ^GSPC • ^FCHI", style="bold cyan")),
            border_style="cyan",
            padding=(1, 2),
        )
    )

    while True:
        s = console.input("\n[bold cyan]Ticker>[/bold cyan] ").strip()
        if not s:
            continue
        if s.lower() in {"q", "quit", "exit"}:
            console.print("\n[dim]Bye.[/dim]")
            return
        try:
            f = fetch_with_spinner(s)
            console.clear()
            console.print(make_layout(f))
        except Exception as e:
            console.print(
                Panel(
                    Text(
                        f"Could not load data for '{s}'.\n\n"
                        f"Reason: {e}\n\n"
                        "Tips:\n"
                        "- Wait 1–2 minutes and retry (rate limits cool down)\n"
                        "- Try another ticker\n"
                        "- The cache will reduce repeated requests",
                        style="red",
                    ),
                    title="Error",
                    border_style="red",
                    padding=(1, 2),
                )
            )

if __name__ == "__main__":
    main()
