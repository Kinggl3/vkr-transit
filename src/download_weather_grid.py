"""
Почасовая погода на сетке узлов вокруг станций метро (Open-Meteo Historical API).

Идея: вместо одной точки на город берём регулярную сетку с шагом STEP градусов,
оставляем только узлы, к которым ближе всего хотя бы одна станция, и для каждой
станции запоминаем её узел. Шаг 0.1° (~11 x 8.5 км) — родное разрешение ERA5-Land,
мельче брать нет смысла: соседние точки внутри ячейки модели дадут те же значения.

Для эксперимента «нужна ли пространственная детализация погоды» можно скачать
несколько вариантов:
  python src/download_weather_grid.py --step 0.1
  python src/download_weather_grid.py --step 0.25

Результаты (на примере шага 0.1):
  data/raw/weather_grid_0.1.parquet        — погода: grid_id, transit_timestamp, переменные
  data/raw/station_weather_map_0.1.csv     — станция -> узел сетки и расстояние до него
Нужен data/raw/stations.csv (его создаёт download_mta.py).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "precipitation", "rain", "snowfall", "snow_depth",
    "cloud_cover", "wind_speed_10m", "weather_code",
]
SUM_VARS = {"precipitation", "rain", "snowfall"}
BATCH = 10  # узлов в одном запросе


def build_grid(stations: pd.DataFrame, step: float):
    """Привязывает каждую станцию к ближайшему узлу сетки с шагом step."""
    st = stations.copy()
    st["grid_lat"] = (np.round(st["latitude"] / step) * step).round(4)
    st["grid_lon"] = (np.round(st["longitude"] / step) * step).round(4)
    nodes = (st[["grid_lat", "grid_lon"]].drop_duplicates()
               .sort_values(["grid_lat", "grid_lon"]).reset_index(drop=True))
    nodes["grid_id"] = [f"g{i:02d}" for i in range(len(nodes))]
    st = st.merge(nodes, on=["grid_lat", "grid_lon"])

    # расстояние станция -> узел в км (для контроля)
    dlat = (st["latitude"] - st["grid_lat"]) * 111.0
    dlon = (st["longitude"] - st["grid_lon"]) * 111.0 * np.cos(np.radians(st["latitude"]))
    st["dist_km"] = np.sqrt(dlat ** 2 + dlon ** 2).round(2)
    return nodes, st


def fetch(nodes: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": ",".join(map(str, nodes["grid_lat"])),
        "longitude": ",".join(map(str, nodes["grid_lon"])),
        "start_date": start, "end_date": end,
        "hourly": ",".join(VARIABLES),
        "timezone": "GMT",
    }
    for attempt in range(1, 6):
        try:
            r = requests.get(URL, params=params, timeout=180)
            if r.status_code == 429:  # превышен лимит запросов
                raise requests.RequestException("429 Too Many Requests")
            r.raise_for_status()
            break
        except requests.RequestException as e:
            wait = 30 * attempt
            print(f"  попытка {attempt}/5 не удалась ({e}); жду {wait} с", flush=True)
            time.sleep(wait)
    else:
        raise RuntimeError("Open-Meteo не отвечает, попробуйте позже.")

    data = r.json()
    if isinstance(data, dict):  # один узел -> объект, несколько -> список
        data = [data]
    frames = []
    for gid, item in zip(nodes["grid_id"], data):
        df = pd.DataFrame(item["hourly"])
        df["grid_id"] = gid
        frames.append(df)
    out = pd.concat(frames)
    out["time"] = pd.to_datetime(out["time"], utc=True)
    return out


def to_local(w: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """UTC -> местное время Нью-Йорка без пояса, с учётом переходов на летнее/зимнее время."""
    w = w.copy()
    w["transit_timestamp"] = w["time"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    agg = {v: ((lambda x: x.sum(min_count=1)) if v in SUM_VARS else "mean") for v in VARIABLES}
    agg["weather_code"] = "max"
    w = w.groupby(["grid_id", "transit_timestamp"]).agg(agg)

    full = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
    parts = []
    for gid, g in w.groupby(level="grid_id"):
        g = g.droplevel("grid_id").reindex(full).ffill(limit=1)  # дыра весеннего перехода
        g.index.name = "transit_timestamp"
        g["grid_id"] = gid
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.1, help="шаг сетки в градусах")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--stations", default="data/raw/stations.csv")
    ap.add_argument("--outdir", default="data/raw")
    args = ap.parse_args()

    stations = pd.read_csv(args.stations, dtype={"station_complex_id": str})
    nodes, st_map = build_grid(stations, args.step)
    counts = st_map.groupby("grid_id").size()
    print(f"Шаг {args.step}°: {len(nodes)} узлов на {len(stations)} станций; "
          f"станций на узел от {counts.min()} до {counts.max()}; "
          f"макс. расстояние станция-узел {st_map['dist_km'].max()} км")

    start = pd.Timestamp(args.start) - pd.Timedelta(days=1)
    end = pd.Timestamp(args.end) + pd.Timedelta(days=1)

    raw = []
    for y in range(start.year, end.year + 1):
        a = max(start, pd.Timestamp(f"{y}-01-01"))
        b = min(end, pd.Timestamp(f"{y}-12-31"))
        for i in range(0, len(nodes), BATCH):
            chunk = nodes.iloc[i:i + BATCH]
            print(f"{a:%Y-%m-%d} – {b:%Y-%m-%d}: узлы {i + 1}–{i + len(chunk)}", flush=True)
            raw.append(fetch(chunk, f"{a:%Y-%m-%d}", f"{b:%Y-%m-%d}"))
            time.sleep(5)  # бережём бесплатный лимит Open-Meteo
    raw = pd.concat(raw).drop_duplicates(["grid_id", "time"])

    w = to_local(raw, args.start, args.end)
    w = w.merge(nodes, on="grid_id")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.step:g}"
    w.to_parquet(outdir / f"weather_grid_{tag}.parquet", index=False)
    st_map[["station_complex_id", "station_complex", "borough", "latitude", "longitude",
            "grid_id", "grid_lat", "grid_lon", "dist_km"]].to_csv(
        outdir / f"station_weather_map_{tag}.csv", index=False)

    tail = w[w["temperature_2m"].isna()]["transit_timestamp"]
    print(f"Строк: {len(w):,}. Пропусков по температуре: {len(tail)}"
          + (f" (с {tail.min()}, это отставание архива ~5 дней)" if len(tail) else ""))

    # насколько узлы реально различаются — пригодится для текста ВКР
    piv = w.pivot_table(index="transit_timestamp", columns="grid_id", values="precipitation")
    rainy = piv[(piv > 0.1).any(axis=1)]
    share = ((rainy > 0.1).sum(axis=1) < rainy.shape[1]).mean()
    print(f"Часы с дождём хотя бы в одном узле: {len(rainy):,}; "
          f"из них дождь шёл не во всех узлах: {share:.0%}")
    print(f"Сохранено в {outdir.resolve()}")


if __name__ == "__main__":
    main()