"""
Загрузка собранного датасета с погодой выбранного уровня.

Пример (из ноутбука в notebooks/):
    import sys; sys.path.append("../src")
    from dataset import load_dataset
    df = load_dataset(weather="0.1")                 # погода по сетке 0.1°
    df = load_dataset(weather="city", start="2025-01-01")

weather:
    "city" — одна точка (Центральный парк) на весь город
    "0.1"  — сетка 0.1° (~12 узлов), каждой станции свой узел
    "0.25" — сетка 0.25° (~6 узлов)
    None   — без погоды
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

WEATHER_COLS = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "precipitation", "rain", "snowfall", "snow_depth",
    "cloud_cover", "wind_speed_10m", "weather_code",
]


def _weather(level: str) -> pd.DataFrame:
    if level == "city":
        w = pd.read_parquet(RAW / "weather_hourly.parquet")
        return w[["transit_timestamp"] + WEATHER_COLS]
    w = pd.read_parquet(RAW / f"weather_grid_{level}.parquet")
    return w[["grid_id", "transit_timestamp"] + WEATHER_COLS]


def load_dataset(weather: str | None = "city", start=None, end=None,
                 stations=None) -> pd.DataFrame:
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    if start is not None:
        panel = panel[panel["transit_timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        panel = panel[panel["transit_timestamp"] <= pd.Timestamp(end)]
    if stations is not None:
        panel = panel[panel["station_complex_id"].isin([str(s) for s in stations])]

    cal = pd.read_parquet(PROCESSED / "calendar.parquet")
    st = pd.read_parquet(PROCESSED / "stations.parquet")

    df = panel.merge(cal, on="transit_timestamp", how="left")
    st_cols = ["station_complex_id", "borough", "latitude", "longitude", "in_cbd"]
    df["station_complex_id"] = df["station_complex_id"].astype(str)
    df = df.merge(st[st_cols + [c for c in st.columns if c.startswith("grid_")]],
                  on="station_complex_id", how="left")

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
    df["station_complex_id"] = df["station_complex_id"].astype("category")
    df["borough"] = df["borough"].astype("category")
    return df.drop(columns=[c for c in df.columns if c.startswith("grid_")])