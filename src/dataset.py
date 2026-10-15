"""
Загрузка собранного датасета: панель станция × час + календарь + признаки станций + события + погода.

Пример (из ноутбука в notebooks/):
    import sys; sys.path.append("../src")
    from dataset import load_dataset
    df = load_dataset(weather="0.1")                 # погода по сетке 0.1°
    df = load_dataset(weather="city", start="2025-01-01")
    train = df[df["split"] == "train"]

weather:
    "city" — одна точка (Центральный парк) на весь город
    "0.1"  — сетка 0.1° (~12 узлов), каждой станции свой узел
    "0.25" — сетка 0.25° (~6 узлов)
    None   — без погоды

Если build_features.py ещё не запускался, грузится базовый вариант (calendar.parquet, stations.parquet)
без признаков станций и событий.

Важно про погоду: это фактическая погода (реанализ), а не прогноз. В реальном прогнозе на 1–24 ч
была бы доступна только прогнозная погода — это допущение («идеальный прогноз погоды»).
"""

import pandas as pd

from config import PROCESSED, RAW, split_of

WEATHER_COLS = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "precipitation", "rain", "snowfall", "snow_depth",
    "cloud_cover", "wind_speed_10m", "weather_code",
]
STATION_COLS = ["borough", "latitude", "longitude", "in_cbd", "scale_weekday", "commute_index",
                "cluster", "n_lines", "venue", "od_out_flow", "od_in_flow", "od_out_degree"]


def _weather(level: str) -> pd.DataFrame:
    if level == "city":
        w = pd.read_parquet(RAW / "weather_hourly.parquet")
        return w[["transit_timestamp"] + WEATHER_COLS]
    w = pd.read_parquet(RAW / f"weather_grid_{level}.parquet")
    return w[["grid_id", "transit_timestamp"] + WEATHER_COLS]


def load_dataset(weather: str | None = "city", start=None, end=None, stations=None) -> pd.DataFrame:
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    if start is not None:
        panel = panel[panel["transit_timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        panel = panel[panel["transit_timestamp"] <= pd.Timestamp(end)]
    if stations is not None:
        panel = panel[panel["station_complex_id"].isin([str(s) for s in stations])]
    panel = panel.copy()
    panel["station_complex_id"] = panel["station_complex_id"].astype(str)

    cal_path = PROCESSED / "calendar_ext.parquet"
    cal = pd.read_parquet(cal_path if cal_path.exists() else PROCESSED / "calendar.parquet")
    st_path = PROCESSED / "station_static.parquet"
    st = pd.read_parquet(st_path if st_path.exists() else PROCESSED / "stations.parquet")

    df = panel.merge(cal, on="transit_timestamp", how="left")
    cols = [c for c in STATION_COLS if c in st.columns] + [c for c in st.columns if c.startswith("grid_")]
    df = df.merge(st[["station_complex_id"] + cols], on="station_complex_id", how="left")

    ev_path = PROCESSED / "events_station_hour.parquet"
    if ev_path.exists():
        ev = pd.read_parquet(ev_path)
        ev["station_complex_id"] = ev["station_complex_id"].astype(str)
        df = df.merge(ev, on=["station_complex_id", "transit_timestamp"], how="left")
        df["ev_today"] = df["ev_today"].fillna(0).astype("int8")
        df["ev_active"] = df["ev_active"].fillna(0).astype("int8")
        df["ev_kind"] = df["ev_kind"].astype(object).fillna("").astype("category")

    cl_path = PROCESSED / "closure_lag.parquet"
    if cl_path.exists():                     # закрытия, известные в 00:00 (по итогам вчерашнего дня)
        cl = pd.read_parquet(cl_path)
        df["date"] = df["transit_timestamp"].dt.normalize()
        df = df.merge(cl, on=["station_complex_id", "date"], how="left").drop(columns="date")
        df["closed_prev_day"] = df["closed_prev_day"].fillna(0).astype("int8")
        df["closed_run_days"] = df["closed_run_days"].fillna(0).astype("int16")

    if weather == "city":
        df = df.merge(_weather("city"), on="transit_timestamp", how="left")
    elif weather in ("0.1", "0.25"):
        col = "grid_" + weather.replace(".", "")
        w = _weather(weather).rename(columns={"grid_id": col})
        df = df.merge(w, on=[col, "transit_timestamp"], how="left")
    elif weather is not None:
        raise ValueError("weather: 'city', '0.1', '0.25' или None")

    for c in WEATHER_COLS:
        if c in df:
            df[c] = df[c].astype("float32")
    df["split"] = split_of(df["transit_timestamp"])
    for c in ["station_complex_id", "borough", "venue"]:
        if c in df:
            df[c] = df[c].astype("category")
    return df.drop(columns=[c for c in df.columns if c.startswith("grid_")])
