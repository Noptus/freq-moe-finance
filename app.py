# FreqMoE Streamlit Demo
# Author: Tina Truong
# --------------------------------------------------
# A lightweight, deploy‑friendly implementation of the Frequency‑Decomposition Mixture‑of‑Experts
# (FreqMoE) model for time‑series forecasting.  Users can choose between uploading a CSV file or
# fetching financial price data from Finnhub.  The model splits the input series into N frequency
# bands, trains one expert per band, weights their forecasts with a soft‑max gate, and returns a
# 30‑step forecast together with intuitive visualisations of each expert’s contribution.
#
# --------------------------------------------------
# 3rd‑party dependencies (see requirements.txt):
#   streamlit, numpy, pandas, scipy, scikit‑learn, matplotlib, requests (for Finnhub)
# --------------------------------------------------

import io
import os
import time
import json
import math
import textwrap
import datetime as dt
from typing import List, Tuple

import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.fft import rfft, irfft
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# -------------------- Streamlit page config --------------------
st.set_page_config(
    page_title="FreqMoE Forecaster",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔮 FreqMoE Time‑Series Forecaster")
st.markdown(
    """Interactively forecast the next **30 steps** of a univariate time series using
    **Frequency‑Decomposition Mixture‑of‑Experts (FreqMoE)**.  Each expert focuses on a
    different frequency band, and a soft‑max gate combines their predictions.
    """
)

st.sidebar.markdown("**Author :** Tina Truong")

# -------------------- Utility functions --------------------

DEFAULT_FINNHUB_KEY = "sandbox_finnhub_demo_key"  # Replace with your demo key if desired
FINNHUB_CANDLE_ENDPOINT = "https://finnhub.io/api/v1/stock/candle"

HORIZON = 30  # forecast horizon in steps (configurable here)


def fetch_finnhub_series(symbol: str, start: dt.date, end: dt.date, token: str) -> pd.Series:
    """Fetches daily close prices from Finnhub and returns them as a pandas Series indexed by date."""

    params = {
        "symbol": symbol.upper(),
        "resolution": "D",
        "from": int(time.mktime(start.timetuple())),
        "to": int(time.mktime(end.timetuple())),
        "token": token,
    }
    r = requests.get(FINNHUB_CANDLE_ENDPOINT, params=params, timeout=10)
    r.raise_for_status()
    payload = r.json()

    if payload.get("s") != "ok":
        raise ValueError("Finnhub replied with an error or no data (status=%s)." % payload.get("s"))

    closes = payload["c"]
    timestamps = payload["t"]  # unix epoch seconds
    dates = pd.to_datetime(timestamps, unit="s")
    series = pd.Series(closes, index=dates, name="close").astype(float)
    return series


def prepare_sliding_windows(values: np.ndarray, lookback: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Creates (X, y) pairs for supervised learning from a 1‑D array."""
    n_total = len(values)
    n_samples = n_total - lookback - horizon + 1
    if n_samples <= 0:
        raise ValueError("Time series too short for the chosen look‑back and horizon.")
    X = np.zeros((n_samples, lookback))
    y = np.zeros((n_samples, horizon))
    for i in range(n_samples):
        X[i] = values[i : i + lookback]
        y[i] = values[i + lookback : i + lookback + horizon]
    return X, y


# -------- Frequency decomposition / expert helpers --------

def split_frequency_bands(n_freq: int, n_experts: int) -> List[slice]:
    """Returns list of slice objects partitioning the positive‑frequency indices into n_experts bands."""
    # Equal‑width partitioning in the frequency domain.
    band_width = n_freq // n_experts
    slices = []
    for i in range(n_experts):
        start = i * band_width
        end = (i + 1) * band_width if i < n_experts - 1 else n_freq  # last band takes remainder
        slices.append(slice(start, end))
    return slices


def decompose_series(series: np.ndarray, n_experts: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """Decomposes signal into n_experts band‑limited components and returns (components, band_energies)."""
    fft_coeffs = rfft(series)  # real FFT (length n_freq)
    mag = np.abs(fft_coeffs)
    n_freq = len(fft_coeffs)
    bands = split_frequency_bands(n_freq, n_experts)

    components = []
    energies = np.zeros(n_experts)
    for idx, sl in enumerate(bands):
        mask = np.zeros_like(fft_coeffs, dtype=bool)
        mask[sl] = True
        masked_coeffs = np.where(mask, fft_coeffs, 0)
        component = irfft(masked_coeffs, n=len(series))  # back to time domain
        components.append(component)
        energies[idx] = mag[sl].sum()
    return components, energies


# -------------- Expert model wrapper --------------

class ExpertModel:
    """Wraps a regression model (linear or MLP) for convenience."""

    def __init__(self, model_type: str, lookback: int, horizon: int, random_state: int = 42):
        self.model_type = model_type
        self.lookback = lookback
        self.horizon = horizon
        self.random_state = random_state
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self._build_model()

    def _build_model(self):
        if self.model_type == "Linear Regression":
            self.model = LinearRegression(n_jobs=1)
        else:  # MLP
            self.model = MLPRegressor(
                hidden_layer_sizes=(64,),
                activation="relu",
                solver="adam",
                max_iter=500,
                random_state=self.random_state,
                verbose=False,
            )

    def fit(self, series: np.ndarray):
        X, y = prepare_sliding_windows(series, self.lookback, self.horizon)
        # Standardise features & targets for MLP stability; harmless for linear.
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        self.model.fit(X_scaled, y_scaled)

    def predict(self, recent_window: np.ndarray) -> np.ndarray:
        assert len(recent_window) == self.lookback
        X = recent_window.reshape(1, -1)
        X_scaled = self.scaler_x.transform(X)
        y_scaled_pred = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_scaled_pred).flatten()
        return y_pred  # length horizon


# -------------- Sidebar: data source selection --------------

st.sidebar.header("1  Choose Data Source 📈")
data_source = st.sidebar.radio("Select source", ("CSV Upload", "Finnhub API"))

if data_source == "CSV Upload":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("CSV must contain at least one numeric column.")
            st.stop()
        col_choice = st.sidebar.selectbox("Column to forecast", numeric_cols)
        series = pd.Series(df[col_choice].values, index=np.arange(len(df[col_choice])))
    else:
        st.info("Please upload a CSV file.")
        st.stop()
else:
    st.sidebar.write("Fetch daily OHLC prices from Finnhub (free tier).")
    default_symbol = "AAPL"
    symbol = st.sidebar.text_input("Ticker symbol", value=default_symbol)
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=365 * 2)  # 2 years
    token_input = st.sidebar.text_input(
        "Finnhub API key (leave blank for demo key)",
        value="",
        type="password",
    )
    api_key = token_input.strip() or DEFAULT_FINNHUB_KEY
    try:
        series = fetch_finnhub_series(symbol, start_date, end_date, api_key)
    except Exception as e:
        st.error(f"Finnhub error: {e}")
        st.stop()

# Show raw data preview
st.subheader("Input Time Series")
st.line_chart(series)

# -------------- Sidebar: model parameters --------------

st.sidebar.header("2  Model Parameters ⚙️")
num_experts = st.sidebar.slider("Number of experts", min_value=1, max_value=10, value=3, step=1)
model_type = st.sidebar.selectbox("Expert model type", ["Linear Regression", "Neural Network (MLP)"])
lookback = st.sidebar.slider("Look‑back window (history points)", min_value=20, max_value=200, value=100, step=5)

run = st.sidebar.button("🚀 Run Forecast")

if run:
    st.subheader("Results")
    data = series.values.astype(float)
    if len(data) < lookback + HORIZON:
        st.error("Time series too short for the given look‑back and forecast horizon.")
        st.stop()

    # -------- Decompose into frequency bands --------
    components, band_energy = decompose_series(data, num_experts)
    weights = np.exp(band_energy) / np.exp(band_energy).sum()  # soft‑max

    # -------- Train experts & forecast --------
    expert_forecasts = []
    recent_idx_start = len(data) - lookback
    for idx, comp in enumerate(components):
        expert = ExpertModel(model_type=model_type, lookback=lookback, horizon=HORIZON, random_state=idx)
        expert.fit(comp)
        forecast = expert.predict(comp[recent_idx_start:])
        expert_forecasts.append(forecast)

    expert_forecasts = np.array(expert_forecasts)  # shape (n_experts, HORIZON)
    final_forecast = (weights[:, None] * expert_forecasts).sum(axis=0)

    # -------- Visualisation --------
    # Prepare future index (numeric or datetime depending on input)
    if isinstance(series.index, pd.DatetimeIndex):
        freq = pd.infer_freq(series.index) or "D"
        fut_index = pd.date_range(series.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=HORIZON, freq=freq)
    else:
        fut_index = np.arange(len(series), len(series) + HORIZON)

    # Main plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series.index, series.values, label="Historical", color="black")
    ax.plot(fut_index, final_forecast, label="Forecast", color="red")
    ax.axvline(series.index[-1], linestyle="--", color="gray", alpha=0.5)

    # Plot expert components (optional)
    for i in range(num_experts):
        ax.plot(
            fut_index,
            expert_forecasts[i],
            label=f"Expert {i+1} ({weights[i]*100:.1f}% weight)",
            linestyle="--",
            alpha=0.7,
        )

    ax.set_title("FreqMoE Forecast (30 steps ahead)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    st.pyplot(fig)

    # Gating weights bar
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.bar(range(1, num_experts + 1), weights * 100)
    ax2.set_xlabel("Expert #")
    ax2.set_ylabel("Weight (%)")
    ax2.set_title("Gating Weights")
    ax2.set_xticks(range(1, num_experts + 1))
    st.pyplot(fig2)

    # Expert energy information
    energy_df = pd.DataFrame({"Expert": np.arange(1, num_experts + 1), "Band Energy": band_energy, "Weight": weights})
    st.write("### Expert Band Energies & Weights")
    st.dataframe(energy_df.round(4))

    st.success("Forecast complete ✔️")
