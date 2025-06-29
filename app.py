# FreqMoE Streamlit Demo (Polygon.io Edition)
# Author: Tina Truong
# --------------------------------------------------
# Automatically forecasts the next 30 steps of a univariate series using
# Frequency‑Decomposition Mixture‑of‑Experts (FreqMoE).  Users can upload
# a CSV **or** fetch daily OHLC from Polygon.io.
# --------------------------------------------------
# Highlights
#   • Energy‑balanced frequency bands for more even expert focus
#   • Linear or MLP experts trained on‑the‑fly (no button needed)
#   • Soft‑max gate combines expert forecasts
#   • Detailed table shows each expert's band, bandwidth, energy share,
#     learned weight and model configuration
# --------------------------------------------------
#  Requirements (see requirements.txt):
#   streamlit, numpy, pandas, scipy, scikit-learn, matplotlib, requests
# --------------------------------------------------

from __future__ import annotations

import datetime as dt
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.ticker import MaxNLocator
from scipy.fft import rfft, irfft
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# -------------------- Streamlit page config --------------------
st.set_page_config(
    page_title="FreqMoE Forecaster – Polygon.io",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔮 FreqMoE Time‑Series Forecaster")
st.sidebar.markdown("**Author :** Tina Truong")

st.markdown(
    """
    **FreqMoE** splits the input series into frequency bands, trains one expert per
    band, then gates their forecasts by the input's frequency energy.  All steps run
    automatically as soon as data & parameters are available—no extra clicks.
    """
)

# -------------------- Constants --------------------
DEFAULT_POLYGON_KEY = "p46qnFerUpAecBAsNFBHzUhuhKGrYGM5"
API_ENDPOINT = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
HORIZON = 30  # forecast length

# -------------------- Helper functions --------------------

def fetch_polygon_series(ticker: str, start: dt.date, end: dt.date, api_key: str) -> pd.Series:
    """Fetch daily closes from Polygon.io."""
    url = API_ENDPOINT.format(ticker=ticker.upper(), start=start, end=end)
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }
    res = requests.get(url, params=params, timeout=10)
    res.raise_for_status()
    data = res.json()
    if not data.get("results"):
        raise ValueError("Polygon returned no data for this period.")
    closes = [item["c"] for item in data["results"]]
    ts = [item["t"] for item in data["results"]]  # unix ms
    idx = pd.to_datetime(ts, unit="ms")
    return pd.Series(closes, index=idx, name="close").astype(float)


def sliding_windows(values: np.ndarray, lookback: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(values) - lookback - horizon + 1
    if n <= 0:
        raise ValueError("Series too short for chosen look‑back/horizon.")
    X = np.lib.stride_tricks.sliding_window_view(values, lookback)[:n]
    y = np.array([values[i + lookback : i + lookback + horizon] for i in range(n)])
    return X, y


# ---------- Energy‑balanced band splitter ----------

def energy_balanced_bands(energy: np.ndarray, n_experts: int) -> List[slice]:
    """Return slices where each band has ~equal cumulative energy."""
    total = energy.sum()
    target = total / n_experts
    cuts = [0]
    acc = 0.0
    for i, e in enumerate(energy):
        acc += e
        if acc >= target and len(cuts) < n_experts:
            cuts.append(i + 1)
            acc = 0.0
    cuts.append(len(energy))
    # ensure monotonic increasing indices & length = n_experts + 1
    while len(cuts) < n_experts + 1:
        cuts.insert(-1, cuts[-2])
    bands = [slice(cuts[i], cuts[i + 1]) for i in range(n_experts)]
    return bands


def decompose(series: np.ndarray, n_experts: int):
    coeffs = rfft(series)
    mag = np.abs(coeffs)
    # Build energy array for each single frequency bin (ignoring symmetry as rfft output already half)
    bands = energy_balanced_bands(mag, n_experts)
    components, band_energy = [], []
    for sl in bands:
        mask = np.zeros_like(coeffs, dtype=bool)
        mask[sl] = True
        comp_coeffs = np.where(mask, coeffs, 0)
        components.append(irfft(comp_coeffs, n=len(series)))
        band_energy.append(mag[sl].sum())
    return components, np.array(band_energy), bands


# ---------- Expert Wrapper ----------

class Expert:
    def __init__(self, model_type: str, lookback: int, horizon: int, seed: int):
        self.lookback, self.horizon = lookback, horizon
        self.scaler_x, self.scaler_y = StandardScaler(), StandardScaler()
        self.model_type = model_type
        if model_type == "Linear Regression":
            self.model = LinearRegression()
        else:
            self.model = MLPRegressor(hidden_layer_sizes=(64,), max_iter=500, random_state=seed)

    def fit(self, series: np.ndarray):
        X, y = sliding_windows(series, self.lookback, self.horizon)
        Xs = self.scaler_x.fit_transform(X)
        ys = self.scaler_y.fit_transform(y)
        self.model.fit(Xs, ys)

    def predict(self, recent: np.ndarray) -> np.ndarray:
        xs = self.scaler_x.transform(recent.reshape(1, -1))
        ys = self.model.predict(xs)
        return self.scaler_y.inverse_transform(ys).flatten()


# -------------------- Sidebar: inputs --------------------
st.sidebar.header("Data Source 📈")
mode = st.sidebar.radio("Select source", ("CSV Upload", "Polygon API"))

if mode == "CSV Upload":
    file = st.sidebar.file_uploader("Upload CSV", ["csv"])
    if file is None:
        st.info("Upload a CSV to begin.")
        st.stop()
    df = pd.read_csv(file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric column found in CSV.")
        st.stop()
    col = st.sidebar.selectbox("Column", numeric_cols)
    series = pd.Series(df[col].values, index=np.arange(len(df[col])))
else:
    ticker = st.sidebar.text_input("Ticker", "AAPL")
    key = st.sidebar.text_input("Polygon API key", DEFAULT_POLYGON_KEY, type="password")
    end_d = dt.date.today()
    start_d = end_d - dt.timedelta(days=365 * 2)
    try:
        series = fetch_polygon_series(ticker, start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d"), key)
    except Exception as exc:
        st.error(f"Polygon error: {exc}")
        st.stop()

# Model params
st.sidebar.header("Model Parameters ⚙️")
N = st.sidebar.slider("Number of experts", 1, 10, 3)
model_choice = st.sidebar.selectbox("Expert model", ["Linear Regression", "Neural Network (MLP)"])
lookback = st.sidebar.slider("Look‑back window", 20, 200, 100, step=5)

# -------------------- Run pipeline automatically --------------------
values = series.values.astype(float)
if len(values) < lookback + HORIZON:
    st.error("Series too short for chosen parameters.")
    st.stop()

components, band_energy, band_slices = decompose(values, N)
weights = np.exp(band_energy) / np.exp(band_energy).sum()

recent = values[-lookback:]
expert_preds = []
experts = []
for i, comp in enumerate(components):
    exp = Expert(model_choice, lookback, HORIZON, seed=i)
    exp.fit(comp)
    experts.append(exp)
    expert_preds.append(exp.predict(comp[-lookback:]))

a_preds = np.array(expert_preds)
forecast = (weights[:, None] * a_preds).sum(axis=0)

# -------------------- Visuals --------------------
# Future index
if isinstance(series.index, pd.DatetimeIndex):
    freq = pd.infer_freq(series.index) or "D"
    fut_idx = pd.date_range(series.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=HORIZON, freq=freq)
else:
    fut_idx = np.arange(len(series), len(series) + HORIZON)

st.subheader("Forecast & Experts")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(series.index, series.values, label="Historical", color="black")
ax.plot(fut_idx, forecast, label="Forecast", color="red")
ax.axvline(series.index[-1], ls="--", color="gray", alpha=0.5)
for i in range(N):
    ax.plot(fut_idx, a_preds[i], ls="--", alpha=0.7, label=f"Expert {i+1} ({weights[i]*100:.1f}%)")
ax.set_title("FreqMoE 30‑step Forecast")
ax.legend(loc="upper left")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
st.pyplot(fig)

# Expert details table
start_end = [(sl.start, sl.stop) for sl in band_slices]
bandwidths = [sl.stop - sl.start for sl in band_slices]

details = pd.DataFrame(
    {
        "Expert": np.arange(1, N + 1),
        "Freq idx start": [s for s, _ in start_end],
        "Freq idx end": [e for _, e in start_end],
        "Bandwidth (bins)": bandwidths,
        "Band energy": band_energy.round(2),
        "Gate weight %": (weights * 100).round(2),
        "Model": model_choice,
        "Look‑back": lookback,
    }
).set_index("Expert")

st.subheader("Expert Details")
st.dataframe(details)

st.success("Automatic forecast complete ✔️")
