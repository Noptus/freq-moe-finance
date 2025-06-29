# FreqMoE Streamlit Demo – Polygon.io Edition (energy‑quantile bands)
# Author: Tina Truong
# --------------------------------------------------
# Band selection is now **FFT‑energy quantile** based:
#   • We cut the rFFT magnitude spectrum into N slices so that each slice carries
#     ~equal cumulative energy (quantile boundaries).  This yields meaningful
#     non‑empty bands no matter the series length.
#   • Fallback to energy-balanced slices if total energy is 0.
# Adds a final table summarising each expert: start, end freq idx, bandwidth,
#   band energy, back‑weight, future‑weight, model.
# Keeps 30‑step back‑test, forecast, and per‑expert plots.
# --------------------------------------------------
# Requirements: streamlit numpy pandas scipy scikit-learn matplotlib requests
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
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import multiprocessing as mp

# ---------- Config ----------
st.set_page_config(page_title="FreqMoE Forecaster", page_icon="🔮", layout="wide")

st.title("🔮 FreqMoE Time‑Series Forecaster – Energy‑Quantile Bands")
st.sidebar.markdown("**Streamlit and implementation Author:** Tina Truong")
st.sidebar.markdown("[Original FreqMoE paper, by Ziqi Liu](https://arxiv.org/abs/2501.15125)")
st.sidebar.markdown("[Connect with me on LinkedIn](https://www.linkedin.com/in/tina-truong-nguyen/)")

DEFAULT_POLY_KEY = "p46qnFerUpAecBAsNFBHzUhuhKGrYGM5"
DEFAULT_POLY_KEY = ""
POLY_EP = "https://api.polygon.io/v2/aggs/ticker/{tic}/range/1/day/{start}/{end}"
HORIZON = 30
TEMP, EPS = 1.0, 1e-3

# ---------- Helpers ----------

# ---------- Weather helper (Open-Meteo ERA-5) ----------
def fetch_weather(latitude: float,
                  longitude: float,
                  start: str,
                  end: str,
                  variable: str = "temperature_2m") -> pd.Series:
    """
    Download hourly ERA-5 reanalysis for a given lat/lon & date range.
    variable options: temperature_2m, relative_humidity_2m, precipitation,
                      wind_speed_10m, surface_pressure … (see open-meteo.com)
    """
    base = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start,
        "end_date": end,
        "hourly": variable,
        "timezone": "UTC",
    }
    for attempt in range(3):
        try:
            r = requests.get(base, params=params, timeout=10)
            r.raise_for_status()
            break
        except requests.exceptions.ReadTimeout:
            if attempt == 2:
                raise
    js = r.json()
    if "hourly" not in js or variable not in js["hourly"]:
        raise ValueError("No data returned.")
    times = pd.to_datetime(js["hourly"]["time"])
    data  = pd.Series(js["hourly"][variable], index=times, name=variable).astype(float)
    return data

def fetch_poly(tic: str, start: str, end: str, key: str) -> pd.Series:
    url = POLY_EP.format(tic=tic.upper(), start=start, end=end)
    r = requests.get(url, params={"adjusted": "true", "limit": 50000, "apiKey": key}, timeout=10)
    r.raise_for_status()
    js = r.json()
    if not js.get("results"):
        raise ValueError("Polygon returned no data.")
    closes = [it["c"] for it in js["results"]]
    idx = pd.to_datetime([it["t"] for it in js["results"]], unit="ms")
    return pd.Series(closes, index=idx, name="close").astype(float).ffill()


def make_windows(arr: np.ndarray, lkbk: int, h: int):
    n = len(arr) - lkbk - h + 1
    if n <= 0:
        raise ValueError("Series too short.")
    X = np.lib.stride_tricks.sliding_window_view(arr, lkbk)[:n]
    y = np.vstack([arr[i + lkbk : i + lkbk + h] for i in range(n)])
    return X, y

# ---------- Band selection (energy quantile) ----------

def quantile_slices(mag: np.ndarray, n_exp: int) -> List[slice]:
    n_freq = len(mag)

    # --- 1. Reserve DC bin (0) --------------------------
    dc_width = max(5, len(mag)//256)        # at least 5 bins or ~0.4 %
    dc_slice = slice(0, dc_width)
    mag_no_dc = mag[dc_width:]              # energy excluding widened DC           # energy excluding DC

    # --- 2. Quantile cuts on the remaining spectrum ----
    cum = np.cumsum(mag_no_dc)
    total = cum[-1]
    if total == 0.0:                # completely flat → equal-width fallback
        step = max(1, n_freq // n_exp)
        return [slice(i*step,
                      (i+1)*step if i < n_exp-1 else n_freq)
                for i in range(n_exp)]
    bands_needed = n_exp - 1        # one slot already taken by DC
    targets = np.linspace(0, total, bands_needed + 1)[1:-1]
    cuts, last = [dc_width], dc_width            # start counting after DC
    for t in targets:
        idx = int(np.searchsorted(cum, t, side="right")) + 1 
        cuts.append(max(idx, last + 1))
        last = cuts[-1]
    cuts.append(n_freq)

    # --- 3. Enforce min width ---------------------------
    min_bins = max(3, n_freq // 100)   # ≥3 bins or ≥1 % of spectrum
    for i in range(1, len(cuts)):
        if cuts[i] - cuts[i-1] < min_bins:
            cuts[i] = min(cuts[i-1] + min_bins, n_freq)
    for i in range(len(cuts)-2, -1, -1):             # backward pass in case of collision
        if cuts[i+1] - cuts[i] < min_bins:
            cuts[i] = max(cuts[i+1] - min_bins, 1)

    # Re-insert DC slice at the front
    slices = [dc_slice] + [slice(cuts[i], cuts[i+1]) for i in range(len(cuts)-1)]
    # Trim / pad to exactly n_exp bands
    if len(slices) > n_exp:
        slices = slices[:n_exp-1] + [slice(slices[n_exp-1].start, n_freq)]
    elif len(slices) < n_exp:        # merge last slice repeatedly until count matches
        while len(slices) < n_exp:
            s_last = slices.pop()
            s_penult = slices.pop()
            merged = slice(s_penult.start, s_last.stop)
            slices.append(merged)

    return slices


def decompose(sig: np.ndarray, n_exp: int):
    coeffs = rfft(sig)
    mag = np.abs(coeffs)
    bands = quantile_slices(mag, n_exp)
    comps, ener = [], []
    for sl in bands:
        mask = np.zeros_like(coeffs, dtype=bool); mask[sl] = True
        comps.append(irfft(np.where(mask, coeffs, 0), n=len(sig)))
        ener.append(float(mag[sl].sum()))
    return comps, np.array(ener), bands

# ---------- Expert ----------
class Expert:
    def __init__(self, kind: str, lk: int, h: int, seed: int):
        self.lk, self.h = lk, h
        self.sx, self.sy = StandardScaler(), StandardScaler()
        if kind == "Neural Network (MLP)":
            hid1, hid2 = max(32, lk//4), max(16, lk//8)
            self.m = MLPRegressor(hidden_layer_sizes=(hid1, hid2),
                                   activation="relu",
                                   solver="adam",
                                   learning_rate="adaptive",
                                   alpha=1e-4,
                                   early_stopping=True,
                                   validation_fraction=0.1,
                                   n_iter_no_change=10,
                                   max_iter=400,
                                   random_state=seed)
        else:
            self.m = LinearRegression()

    def fit(self, series: np.ndarray):
        # 1. Interpolate / fill missing values
        s_clean = (
            pd.Series(series)
            .interpolate("linear", limit_direction="both")
            .bfill()
            .ffill()
            .values
        )

        # 2. Build windows; if none remain we fall back to last‑value baseline
        if len(s_clean) < self.lk + self.h:
            self._fallback_val = s_clean[-1]
            self._use_fallback = True
            return

        X, y = make_windows(s_clean, self.lk, self.h)
        good = (~np.isnan(X).any(axis=1)) & (~np.isnan(y).any(axis=1))
        X, y = X[good], y[good]

        if len(X) == 0:
            self._fallback_val = s_clean[-1]
            self._use_fallback = True
            return

        # 3. Train model
        self._use_fallback = False
        if len(X) < 10:
            self.m = LinearRegression()
        self.m.fit(self.sx.fit_transform(X), self.sy.fit_transform(y))

    def pred(self, recent: np.ndarray):
        if getattr(self, "_use_fallback", False):
            return np.full(self.h, self._fallback_val)
        xs = self.sx.transform(recent.reshape(1, -1))
        return self.sy.inverse_transform(self.m.predict(xs))[0]

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def benchmark_table(backcast: np.ndarray,
                    actual_back: np.ndarray,
                    prev_history: np.ndarray,
                    season_len: int | None = None,
                    ma_window: int = 7) -> pd.DataFrame:
    """
    Compare your back-test (backcast) against several naïve baselines.

    Parameters
    ----------
    backcast       : model's 30-step prediction (length = HORIZON)
    actual_back    : realised values for those 30 steps
    prev_history   :  look-back window *ending one step before* actual_back
                     (used to build naïve forecasts and MASE denominator)
    season_len     : length of seasonality (e.g. 5 for weekly trading days).
                     If None → seasonal naïve is skipped.
    ma_window      : window length for moving-average naïve

    Returns
    -------
    pandas.DataFrame  with MAE, RMSE, MAPE, MASE, Directional-Accuracy
    for each method.
    """

    # --- clean NaNs in backcast ------------------------------------
    backcast = (
        pd.Series(backcast, dtype=float)
        .interpolate("linear", limit_direction="both")
        .fillna(method="ffill")
        .fillna(method="bfill")
        .values
    )

    # --- clean NaNs -------------------------------------------------
    prev_series = (
        pd.Series(prev_history)
        .interpolate("linear", limit_direction="both")
        .bfill()
        .ffill()
    )
    prev_history = prev_series.values  # replace with NaN‑free version

    # --- clean NaNs in actual_back ------------------------------------
    actual_back = (
        pd.Series(actual_back, dtype=float)
        .interpolate("linear", limit_direction="both")
        .fillna(method="ffill")
        .fillna(method="bfill")
    )

    # If any NaNs survive (all‑NaN slice or leading/trailing), fill with the series mean
    if actual_back.isna().any():
        actual_back = actual_back.fillna(actual_back.mean())

    actual_back = actual_back.values

    H = len(actual_back)
    last_val  = prev_history[-1]
    drift_val = prev_history[-1] + np.arange(1, H + 1) * (
                (prev_history[-1] - prev_history[0]) / (len(prev_history) - 1))
    ma_val    = np.full(H, prev_history[-ma_window:].mean())

    naive_dict = {
        "Your FreqMoE": backcast,
        "Last value"  : np.full(H, last_val),
        "Drift (Theil)" : drift_val,
        f"{ma_window}-day mean": ma_val,
    }

    if season_len and len(prev_history) >= season_len:
        seasonal_history = prev_history[-season_len:]
        naive_dict["Seasonal naïve"] = np.tile(seasonal_history, int(np.ceil(H/season_len)))[:H]

    # denominator for MASE: MAE of naive 'random walk' on prev_history
    mase_denom = mean_absolute_error(prev_history[1:], prev_history[:-1])

    rows = []
    for name, raw in naive_dict.items():
        pred = (
            pd.Series(raw, dtype=float)
            .interpolate("linear", limit_direction="both")
            .fillna(method="ffill")
            .fillna(method="bfill")
            .values
        )

        # --- final NaN mask (drop any positions still NaN) ---------------
        valid = (~np.isnan(actual_back)) & (~np.isnan(pred))
        if valid.sum() == 0:
            continue  # Skip this row; no valid data points
        y_true = actual_back[valid]
        y_pred = pred[valid]

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mase = mae / mase_denom if mase_denom else np.nan
        da   = (np.sign(np.diff(y_pred)) == np.sign(np.diff(y_true))).mean() * 100

        rows.append(dict(Method=name, MAE=mae,
                         RMSE=rmse, MAPE=mape, MASE=mase, DA=da))


    if not rows:                       # no valid rows → return empty frame
        return pd.DataFrame(
            {"Info": ["All benchmark rows skipped (no valid data after NaN filtering)"]}
        )

    df = pd.DataFrame(rows).set_index("Method").round(3)
    return df


# ---------- UI inputs ----------

st.sidebar.header("Data Source 📈")
mode = st.sidebar.radio("Source", ("Weather (Open-Meteo)", "Polygon API", "CSV Upload"), index=0)
if mode == "CSV Upload":
    up = st.sidebar.file_uploader("Upload CSV", ["csv"])
    if up is None:
        st.stop()
    df = pd.read_csv(up)
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num:
        st.error("No numeric col."); st.stop()
    col = st.sidebar.selectbox("Column", num)
    series = pd.Series(df[col]).dropna().astype(float).reset_index(drop=True)
elif mode == "Weather (Open-Meteo)":
    city = st.sidebar.text_input("Location (lat,lon)", "40.71,-74.01")   # NYC default
    lat, lon = map(float, city.split(","))
    years = st.sidebar.slider("Years of history", 1, 5, 2)
    variable = st.sidebar.selectbox("Variable",
        ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
         "surface_pressure", "precipitation"])
    end_d = dt.date.today()
    start_d = end_d - dt.timedelta(days=365*years)
    try:
        series = fetch_weather(lat, lon,
                               start_d.strftime("%Y-%m-%d"),
                               end_d.strftime("%Y-%m-%d"),
                               variable)
    except requests.exceptions.ReadTimeout:
        st.error("Open‑Meteo timed‑out. Please reduce the date range or try again later.")
        st.stop()
    except Exception as e:
        st.error(str(e))
        st.stop()
    # ---- make index uniform & naïve ----
    if series.index.tz is not None:
        series = series.tz_localize(None)

    # --- regular 1‑hour grid, but **avoid flat tail** -------------------
    series = (
        series
        .resample("1h")  # pandas v3 prefers lowercase
        .mean()
        # interpolate only *inside* the data, not beyond the first / last real point
        .interpolate("linear", limit_direction="both", limit_area="inside")
        .dropna()                 # drop any trailing NaNs so we don’t ffill them
    )

    st.sidebar.write(f"Weather rows loaded after resample: {len(series)}")
    
else:
    tic = st.sidebar.text_input("Ticker", "AAPL")
    key = st.sidebar.text_input("Polygon API key", DEFAULT_POLY_KEY, type="password")
    ed = dt.date.today(); sd = ed - dt.timedelta(days=365*10)
    try:
        series = fetch_poly(tic, sd.strftime("%Y-%m-%d"), ed.strftime("%Y-%m-%d"), key)
    except Exception as e:
        st.error(e); st.stop()

# ---------- default params depending on source ----------
if mode == "Weather (Open-Meteo)":
    default_lookback = 720     # 30 days of hourly data
    default_N = 4
else:  # stocks or CSV
    default_lookback = 360     # roughly one year of daily closes
    default_N = 3

st.sidebar.header("Model Params ⚙️")
N = st.sidebar.slider("Experts", 2, 8, default_N)
kind = st.sidebar.selectbox("Expert model", ["Neural Network (MLP)", "Linear Regression"], index=0)
lookback = st.sidebar.slider("Look‑back (past timesteps)", 30, 2000, default_lookback, step=10)

# ---------- rough training‑time estimate ----------
vals = series.values.astype(float)
samples = max(0, len(vals) - lookback - HORIZON)
per_sample = 0.0001 if kind == "Neural Network (MLP)" else 0.00003  # seconds
est_time = samples * N * per_sample
est_time = max(0.1, est_time)
st.sidebar.info(f"Estimated run time: ~{est_time:,.1f} s")

if len(vals) < lookback + HORIZON*2 + 10:
    st.error("Series too short."); st.stop()

vals = series.values.astype(float)
if len(vals) < lookback + HORIZON*2 + 10:
    st.error("Series too short."); st.stop()

# ---------- FreqMoE run ----------

def run(data_arr: np.ndarray):
    comps, ener, bands = decompose(data_arr, N)
    #logits = (np.log(ener + 1e-8) - np.log(ener + 1e-8).max()) / TEMP
    #w = np.exp(logits) + EPS; w /= w.sum()
    w = ener / ener.sum()

    def _train_predict(i_comp):
        i, comp = i_comp
        ex = Expert(kind, lookback, HORIZON, i)
        ex.fit(comp)
        return ex.pred(comp[-lookback:])

    n_jobs = min(N, max(1, mp.cpu_count() - 1))
    preds = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_train_predict)(ic) for ic in enumerate(comps)
    )
    return np.array(preds), w, bands, ener

back_preds, w_b, bands, ener_b = run(vals[:-HORIZON])
fore_preds, w_f, _, ener_f = run(vals)
hist_anchor_bt = vals[-HORIZON-1]
raw_backcast   = back_preds.sum(axis=0) 
offset_bt      = hist_anchor_bt - raw_backcast[0]
backcast       = raw_backcast + offset_bt

# Forecast alignment: match last observed value
hist_anchor_fc = vals[-1]
raw_forecast   = fore_preds.sum(axis=0)
offset_fc      = hist_anchor_fc - raw_forecast[0]
forecast       = raw_forecast + offset_fc

# ---------- Plot main ----------
zoom = max(lookback*2 + HORIZON, 240)
start_idx = max(0, len(series) - zoom)
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(series.index[start_idx:-HORIZON], vals[start_idx:-HORIZON], color="black", label="Historical")
ax.plot(series.index[-HORIZON:], vals[-HORIZON:], color="black")
ax.plot(series.index[-HORIZON:], backcast, color="orange", ls="--", label="Back-test pred")
if isinstance(series.index, pd.DatetimeIndex):
    # Try to use the native frequency; if missing, infer from median diff
    if series.index.freq:
        freq_off = series.index.freq
    else:
        # Median time delta (robust against irregular gaps)
        median_delta = series.index.to_series().diff().median()
        if pd.isna(median_delta) or median_delta == pd.Timedelta(0):
            median_delta = pd.Timedelta(days=1)  # conservative fallback
        freq_off = pd.tseries.frequencies.to_offset(median_delta)

    fut_idx = pd.date_range(series.index[-1] + freq_off,
                            periods=HORIZON,
                            freq=freq_off)
else:
    fut_idx = np.arange(len(series), len(series) + HORIZON)
ax.plot(fut_idx, forecast, color="red", label="Forecast (+30)")
ax.set_title("FreqMoE – Energy-Quantile Bands")
ax.legend(); ax.grid(alpha=0.3)
st.pyplot(fig)

# ---------- Metrics & details table ----------
mae = np.mean(np.abs(backcast - vals[-HORIZON:]))
rmse = np.sqrt(np.mean((backcast - vals[-HORIZON:])**2))

st.write(f"### Back-test MAE: {mae:.3f} | RMSE: {rmse:.3f}")
# After computing backcast vs. actual

# Ground‑truth segment for back‑test metrics
actual_back = vals[-HORIZON:]

mape  = np.mean(np.abs((actual_back - backcast) / actual_back)) * 100
mase  = mae / np.mean(np.abs(np.diff(actual_back, prepend=actual_back[0])))

naive = np.concatenate(([vals[-HORIZON-1]], actual_back[:-1]))
rmse_naive = np.sqrt(np.mean((actual_back - naive)**2))
theil_u = rmse / rmse_naive

da = (np.sign(np.diff(backcast)) == np.sign(np.diff(actual_back))).mean() * 100

st.write(f"MAPE: {mape:.2f}% | MASE: {mase:.2f} | Theil U: {theil_u:.2f} | Directional Acc: {da:.1f}%")
bench_df = benchmark_table(backcast=backcast,
                           actual_back=vals[-HORIZON:],
                           prev_history=vals[:-HORIZON],
                           season_len=5,          # 5 trading days = weekly season
                           ma_window=7)

st.subheader("Back-test Metrics vs. Naïve Forecasts")
st.dataframe(bench_df)


starts = [s.start for s in bands]
ends   = [s.stop  for s in bands]
widths = [e - s for s,e in zip(starts,ends)]
expert_tbl = pd.DataFrame({
    "Freq start": starts,
    "Freq end": ends,
    "Bandwidth": widths,
    "Band energy (bt)": ener_b.round(2),
    "Weight % (bt)": (w_b*100).round(2),
    "Weight % (fut)": (w_f*100).round(2),
    "Model": kind,
}, index=[f"Expert {i+1}" for i in range(N)])
st.subheader("Expert Summary Table")
st.dataframe(expert_tbl.style.format({
    "Freq start": "{:d}",
    "Freq end": "{:d}",
    "Bandwidth": "{:d}",
    "Band energy (bt)": "{:.2f}",
    "Weight % (bt)": "{:.2f}%",
    "Weight % (fut)": "{:.2f}%",
    "Model": lambda x: x.replace("Neural Network (MLP)", "MLP").replace("Linear Regression", "LR")
}))

# ---------- Expert plots ----------
st.subheader("Expert Contributions")
for i in range(N):
    fig_i, ax_i = plt.subplots(figsize=(12,3))
    ax_i.plot(series.index[-HORIZON:], back_preds[i], ls="--", color="orange")
    ax_i.plot(fut_idx, fore_preds[i], ls="--", color="blue")
    ax_i.set_title(f"Expert {i+1}  | band bins {bands[i].start}–{bands[i].stop} |\n"
                   f"back w={w_b[i]*100:.1f}%  future w={w_f[i]*100:.1f}%")
    ax_i.grid(alpha=0.3)
    st.pyplot(fig_i)