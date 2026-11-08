"""Загрузка данных прототипа и общие расчёты."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP = Path(__file__).resolve().parent
DATA = APP / "data"
RESULTS = APP.parent / "results"
EPS = 10.0

KIND_VENUE = {"mlb_yankees": "Yankee Stadium", "mlb_mets": "Citi Field / USTA", "usopen_main": "Citi Field / USTA",
              "usopen_fanweek": "Citi Field / USTA", "milb_cyclones": "Maimonides Park", "nba_nets": "Barclays Center",
              "wnba_liberty": "Barclays Center"}
MODEL_NAMES = {"hyb": "Гибрид", "lgbm": "LightGBM", "chronos": "Chronos-2"}
WEEKDAYS = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]
MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"]


def fmt_day(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.day} {MONTHS[ts.month - 1]}, {WEEKDAYS[ts.dayofweek]}"


@st.cache_data
def stations() -> pd.DataFrame:
    s = pd.read_parquet(DATA / "stations.parquet")
    s["short"] = s["name"].str.replace(r"\s*\(.*\)$", "", regex=True)
    s["линии"] = s["lines"]
    return s


@st.cache_data
def edges() -> np.ndarray:
    f = DATA / "edges.npy"
    return np.load(f) if f.exists() else np.zeros((0, 2), int)


@st.cache_data
def series() -> dict:
    z = np.load(DATA / "series.npz")
    out = {k: (z[k].astype("float32") if z[k].dtype == np.float16 else z[k]) for k in z.files}
    out["hours"] = pd.DatetimeIndex(out["hours"])
    return out


@st.cache_data
def scenarios() -> dict | None:
    f = DATA / "scenarios.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return {"days": pd.DatetimeIndex(z["days"]), "keys": list(z["keys"]), "names": list(z["names"]),
            "ratio": z["ratio"]}


@st.cache_data
def events() -> pd.DataFrame:
    f = DATA / "events.csv"
    return pd.read_csv(f, parse_dates=["start"]) if f.exists() else pd.DataFrame(columns=["kind", "venue", "start", "status"])


@st.cache_data
def result(name: str) -> pd.DataFrame | None:
    f = RESULTS / name
    return pd.read_csv(f) if f.exists() else None


def days() -> pd.DatetimeIndex:
    return series()["hours"].normalize().unique()


def day_slice(day) -> slice:
    h = series()["hours"]
    i = h.searchsorted(pd.Timestamp(day))
    return slice(i, min(i + 24, len(h)))


def wape(y, p) -> float:
    m = ~np.isnan(y) & ~np.isnan(p)
    return float(np.abs(p[m] - y[m]).sum() / max(y[m].sum(), 1e-9))


def anomalies(t: int, window: int = 3, thr: float = 2.5) -> pd.DataFrame:
    """Станции, где факт за последние window часов (до часа t включительно) сильно расходится с прогнозом на час.

    Оценка — среднее логарифма (факт + 10) / (прогноз + 10), делённое на типичный разброс этой величины для станции
    на валидации; |z| > thr — провал (z < 0) или всплеск (z > 0)."""
    s = series()
    sl = slice(max(0, t - window + 1), t + 1)
    y, p = s["actual"][sl], s["h1_hyb"][sl]
    r = np.log((y + EPS) / (p + EPS))
    z = np.nanmean(r, 0) / np.maximum(s["sd_h1"], 1e-3)
    st_ = stations()
    df = pd.DataFrame({"id": st_["id"], "станция": st_["short"], "район": st_["borough"], "линии": st_["lines"],
                       "lat": st_["lat"], "lon": st_["lon"], "z": z,
                       "факт": np.nansum(y, 0), "прогноз": np.nansum(p, 0)})
    df["разница"] = df["факт"] - df["прогноз"]
    big = df["прогноз"] >= 30 * window                          # малые станции дают шумные отклонения
    return df[(df["z"].abs() > thr) & big].sort_values("z")


@st.cache_data
def anomaly_counts(thr: float = 2.5) -> pd.Series:
    """Число станций с провалом по часам теста (окно 3 ч) — для поиска эпизодов сбоев."""
    s = series()
    y, p = s["actual"], s["h1_hyb"]
    r = np.log((y + EPS) / (p + EPS))
    k = 3
    c = np.nancumsum(np.nan_to_num(r), 0)
    roll = (c[k - 1:] - np.vstack([np.zeros((1, c.shape[1])), c[:-k]])) / k
    z = roll / np.maximum(s["sd_h1"], 1e-3)
    big = np.vstack([np.nansum(p[i:i + k], 0) for i in range(len(p) - k + 1)]) >= 30 * k
    cnt = ((z < -thr) & big).sum(1)
    return pd.Series(np.r_[np.zeros(k - 1), cnt], index=s["hours"])
