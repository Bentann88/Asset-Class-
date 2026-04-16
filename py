import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta

st.set_page_config(layout="wide")

# ==== CONFIG ====
BASELINE = st.selectbox("Baseline (optional)", [None, "SPY"])

HORIZONS = {
    "1D": lambda today: today - relativedelta(days=1),
    "1W": lambda today: today - relativedelta(weeks=1),
    "1M": lambda today: today - relativedelta(months=1),
    "3M": lambda today: today - relativedelta(months=3),
    "6M": lambda today: today - relativedelta(months=6),
    "YTD": lambda today: datetime(today.year, 1, 1),
    "1Y": lambda today: today - relativedelta(years=1),
    "3Y": lambda today: today - relativedelta(years=3),
}

MULTIYEAR_HORIZONS = {"1Y", "3Y"}

TICKERS = {
    "US Large Cap": "SPY",
    "US Mid Cap": "VO",
    "US Small Cap": "IWM",
    "Developed ex-US": "EFA",
    "Emerging Markets": "EEM",
    "Global ex-US Small": "VSS",
    "US Aggregate Bond": "AGG",
    "US Treasuries (Int)": "IEF",
    "US Treasuries (LT)": "TLT",
    "US TIPS": "TIP",
    "US High Yield": "HYG",
    "Intl Agg Bond": "BNDX",
    "Commodities": "DBC",
    "Gold": "GLD",
    "REITs": "VNQ",
    "Cash": "BIL",
}

# ==== FUNCTIONS ====
def nearest_prior_index(ts, dt):
    idx = ts.index
    loc = idx.get_indexer([dt], method="pad")
    if loc.size and loc[0] != -1:
        return idx[loc[0]]
    before = idx[idx <= dt]
    return before[-1] if len(before) else None

def compute_returns(prices, today):
    results = {}
    end_date = nearest_prior_index(prices.iloc[:, 0], pd.Timestamp(today))

    for label, fn in HORIZONS.items():
        start_target = pd.Timestamp(fn(today))
        start_date = nearest_prior_index(prices.iloc[:, 0], start_target)

        if start_date is None:
            results[label] = pd.Series(np.nan, index=prices.columns)
            continue

        start_vals = prices.loc[start_date]
        end_vals = prices.loc[end_date]

        total_return = end_vals / start_vals - 1

        if label in MULTIYEAR_HORIZONS:
            days = (end_date - start_date).days
            years = days / 365.25
            total_return = (end_vals / start_vals) ** (1 / years) - 1

        results[label] = total_return * 100

    return pd.DataFrame(results)

# ==== LOAD DATA ====
@st.cache_data
def load_data():
    today = pd.Timestamp.today().normalize()
    earliest_start = min(fn(today) for fn in HORIZONS.values())

    tickers = list(TICKERS.values()) + (["SPY"] if BASELINE else [])

    data = yf.download(
        tickers,
        start=earliest_start.date(),
        end=today.date(),
        auto_adjust=True,
        progress=False
    )

    prices = data["Close"] if "Close" in data else data
    return prices.dropna(how="all")

prices = load_data()
today = pd.Timestamp.today().normalize()

returns = compute_returns(prices[list(TICKERS.values())], today)

# ==== BASELINE ====
if BASELINE:
    base = compute_returns(prices[[BASELINE]], today).iloc[0]
    returns = returns.subtract(base, axis=1)
    title = f"Excess Returns vs {BASELINE}"
else:
    title = "Absolute Returns"

# Map labels
ticker_to_label = {v: k for k, v in TICKERS.items()}
returns.index = returns.index.map(ticker_to_label)

# ==== UI ====
st.title("📊 Asset Class Return Dashboard")
st.caption(title)

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        returns,
        annot=True,
        fmt=".1f",
        center=0,
        cmap="RdYlGn",
        linewidths=0.5,
        linecolor="white",
        ax=ax
    )
    ax.xaxis.tick_top()
    st.pyplot(fig)

with col2:
    st.subheader("📈 Rankings")
    ranked = returns.rank(ascending=False)
    st.dataframe(ranked)

st.download_button(
    "Download CSV",
    returns.round(2).to_csv().encode(),
    "asset_returns.csv",
    "text/csv"
)
