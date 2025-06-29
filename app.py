# FreqMoE Streamlit Demo – Polygon.io Edition (balanced & zoomed)
# Author: Tina Truong
# --------------------------------------------------
# * Automatic 30‑step forecast
# * Energy‑balanced frequency bands → 3 experts with stable, non‑zero weights
# * Robust soft‑max (uniform fallback)
# * Larger plot, zoomed on recent history so the forecast is clearly visible
# --------------------------------------------------
# Requirements: streamlit, numpy, pandas, scipy, scikit‑learn, matplotlib, requests
# --------------------------------------------------

from __future__ import annotations

import datetime as dt
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.fft import rfft, irfft
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# -------------------- Streamlit config --------------------
st.set_page_config(
    page_title="FreqMoE Forecaster",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔮 FreqMoE Time‑Series Forecaster")
st.sidebar.markdown("**Author :** Tina Truong")

st.markdown(
    """Automatically decomposes the input time series into **frequency bands**, trains
    one expert per band, and blends their predictions with a soft‑max gate.
    The chart focuses on the **most recent history** so the 30‑step forecast is
    always visible.
    """
)

# -------------------- Constants --------------------
DEFAULT_POLYGON_KEY = "p46qnFerUpAecBAsNFBHzUhuhKGrYGM5"
POLY_ENDPOINT = "https://api.polygon.io/v2/aggs/ticker/{tic}/range/1/day/{start}/{end}"
HORIZON = 30  # forecast length

# -------------------- Helpers --------------------

def fetch_polygon(ticker: str, start: str, end: str, key: str) -> pd.Series:
    url = POLY_ENDPOINT.format(tic=ticker.upper(), start=start, end=end)
    params = {"adjusted": "true", "limit": 50000, "apiKey": key}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    js = r.json()
    if "results" not in js or not js["results"]:
        raise ValueError("Polygon returned no data.")
    closes = [item["c"] for item in js["results"]]
    idx = pd.to_datetime([item["t"] for item in js["results"]], unit="ms")
    return pd.Series(closes, index=idx, name="close").astype(float).ffill()


def sliding_windows(vals: np.ndarray, lkbk: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(vals) - lkbk - h + 1
    if n <= 0:
        raise ValueError("Series too short for look‑back/horizon.")
    X = np.lib.stride_tricks.sliding_window_view(vals, lkbk)[:n]
    y = np.vstack([vals[i + lkbk : i + lkbk + h] for i in range(n)])
    return X, y


# -------------------- Frequency tools --------------------

def energy_balanced_slices(mag: np.ndarray, n_exp: int) -> List[slice]:
    total = mag.sum()
    target = total / n_exp if total > 0 else 1.0
    cuts, acc = [0], 0.0
    for i, e in enumerate(mag):
        acc += e
        if acc >= target and len(cuts) < n_exp:
            cuts.append(i + 1)
            acc = 0.0
    cuts.append(len(mag))
    while len(cuts) < n_exp + 1:  # fill if too short
        cuts.insert(-1, cuts[-2])
    return [slice(cuts[i], cuts[i + 1]) for i in range(n_exp)]


def decompose(sig: np.ndarray, n_exp: int):
    coeffs = rfft(sig)
    mag = np.abs(coeffs)
    bands = energy_balanced_slices(mag, n_exp)
    comps, energy = [], []
    for sl in bands:
        mask = np.zeros_like(coeffs, dtype=bool)
        mask[sl] = True
        comps.append(irfft(np.where(mask, coeffs, 0), n=len(sig)))
        energy.append(mag[sl].sum())
    return comps, np.array(energy), bands


# -------------------- Expert --------------------

class Expert:
    def __init__(self, typ: str, lkbk: int, h: int, seed: int):
        self.lk, self.h = lkbk, h
        self.sx, self.sy = StandardScaler(), StandardScaler()
        if typ == "Linear Regression":
            self.m = LinearRegression()
        else:
            self.m = MLPRegressor(hidden_layer_sizes=(64,), max_iter=500, random_state=seed)

    def fit(self, s: np.ndarray):
        X, y = sliding_windows(s, self.lk, self.h)
        self.m.fit(self.sx.fit_transform(X), self.sy.fit_transform(y))

    def predict(self, recent: np.ndarray) -> np.ndarray:
        return self.sy.inverse_transform(self.m.predict(self.sx.transform(recent.reshape(1, -1))))[0]


# -------------------- Sidebar data input --------------------
st.sidebar.header("Data Source 📈")
mode = st.sidebar.radio("Source", ("CSV Upload", "Polygon API"))

if mode == "CSV Upload":
    up = st.sidebar.file_uploader("Upload CSV", ["csv"])
    if up is None:
        st.info("Upload a CSV to begin.")
        st.stop()
    df = pd.read_csv(up)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.error("No numeric column found.")
        st.stop()
    col = st.sidebar.selectbox("Column", num_cols)
    s = pd.Series(df[col]).dropna().astype(float).reset_index(drop=True)
else:
    tic = st.sidebar.text_input("Ticker", "AAPL")
    key = st.sidebar.text_input("Polygon key", DEFAULT_POLYGON_KEY, type="password")
    end, start = dt.date.today(), dt.date.today() - dt.timedelta(days=365 * 2)
    try:
        s = fetch_polygon(tic, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), key)
    except Exception as e:
        st.error(f"Polygon error: {e}")
        st.stop()

# -------------------- Parameters --------------------
st.sidebar.header("Model Parameters ⚙️")
N = st.sidebar.slider("Experts", 1, 10, 3)
model_kind = st.sidebar.selectbox("Expert model", ["Linear Regression", "Neural Network (MLP)"])
lookback = st.sidebar.slider("Look‑back", 20, 200, 100, step=5)

# -------------------- Pipeline --------------------
vals = s.values.astype(float)
if len(vals) < lookback + HORIZON + 10:  # a bit of slack
    st.error("Series too short for these settings.")
    st.stop()

comps, ener, slices = decompose(vals, N)

# Robust soft‑max
ener = np.where(~np.isfinite(ener), 0, ener)
if ener.sum() == 0:
    w = np.full(N, 1 / N)
else:
    exp_e = np.exp(ener - ener.max())
    w = exp_e / exp_e.sum()

recent = vals[-lookback:]
all_preds = []
for i, c in enumerate(comps):
    ex = Expert(model_kind, lookback, HORIZON, seed=i)
    ex.fit(c)
    all_preds.append(ex.predict(c[-lookback:]))
all_preds = np.array(all_preds)
forecast = (w[:, None] * all_preds).sum(axis=0)

# -------------------- Plot --------------------
zoom_pts = max(lookback * 2, 180)
start_idx = max(0, len(s) - zoom_pts)
plot_x = s.index[start_idx:]
plot_y = s.values[start_idx:]

if isinstance(s.index, pd.DatetimeIndex):
    freq = pd.infer_freq(s.index) or "D"
    fut_idx = pd.date_range(s.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=HORIZON, freq=freq)
else:
    fut_idx = np.arange(len(s), len(s) + HORIZON)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(plot_x, plot_y, color="black", label="Historical", linewidth=1.5)
ax.plot(fut_idx, forecast, color="red", label="Forecast", linewidth=2.2)
ax.axvline(s.index[-1], ls="--", color="gray", alpha=0.6)
for i in range(N):
    ax.plot(
        fut_idx,
        all_preds[i],
        ls="--",
        label=f"Expert {i+1} ({w[i]*100:.1f}%)",
        alpha=0.7,
    )
ax.set_title("FreqMoE – 30‑step Forecast (zoomed)")
ax.legend(loc="upper left")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# -------------------- Details table --------------------
starts = [sl.start for sl in slices]
ends = [sl.stop for sl in slices]
bwidth = [e - s for s, e in zip(starts, ends)]

details = pd.DataFrame(
    {
        "Freq start": starts,
        "Freq end": ends,
        "Bandwidth": bwidth,
        "Energy": ener.round(2),
        "Weight %": (w * 100).round(2),
        "Model": model_kind,
    },
    index=[f"Expert {i+1}" for i in range(N)],
)

st.subheader("Expert Band & Weight Details")
st.dataframe(details)

st.success("Forecast complete ✔️")
