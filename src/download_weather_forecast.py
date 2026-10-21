"""
Прогноз погоды «на сутки вперёд» из Open-Meteo Previous Model Runs API.

Зачем. В download_weather*.py лежит фактическая погода (реанализ ERA5). Модель, которая в 00:00
строит прогноз пассажиропотока на сутки, фактической погоды завтрашнего дня знать не может —
только прогноз. Этот скрипт скачивает значения, которые погодная модель предсказывала
за 24 часа до каждого часа (переменные *_previous_day1). Все они выпущены не позже
00:00 дня прогноза, то есть были известны в момент прогноза спроса.

Схема использования в экспериментах:
  * обучение — на фактической погоде (прогнозов до 2024 года в архиве нет);
  * val / test — на прогнозной погоде; разница в ошибке с вариантом «факт» = цена допущения.

Архив прогнозов у Open-Meteo начинается в январе 2024 (по большинству моделей), поэтому период
по умолчанию 2024-01-01 – 2026-09-24. Высоты снежного покрова (snow_depth) в архиве прогнозов нет.

Запуск (из корня проекта, через VPN, если Open-Meteo недоступен):
  python src/download_weather_forecast.py                    # точка Центральный парк
  python src/download_weather_forecast.py --step 0.1         # сетка 0.1°, как weather_grid_0.1
Результаты:
  data/raw/weather_forecast_hourly.parquet       — transit_timestamp, переменные
  data/raw/weather_forecast_grid_0.1.parquet     — grid_id, transit_timestamp, переменные
Имена колонок те же, что у фактической погоды, чтобы одно можно было подставить вместо другого.
Повторный запуск докачивает только недостающие месяцы (части в data/raw/weather_forecast_parts/).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from download_weather_grid import build_grid

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LAT, LON = 40.7812, -73.9665          # Центральный парк, как в download_weather.py
LEAD = "previous_day1"                # прогноз, выпущенный за 24 ч до часа
VARIABLES = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "precipitation", "rain", "snowfall",
    "cloud_cover", "wind_speed_10m", "weather_code",
]
SUM_VARS = {"precipitation", "rain", "snowfall"}
BATCH = 10


def fetch(lats, lons, start: str, end: str, model: str) -> list:
    params = {
        "latitude": ",".join(map(str, lats)), "longitude": ",".join(map(str, lons)),
        "start_date": start, "end_date": end,
        "hourly": ",".join(f"{v}_{LEAD}" for v in VARIABLES),
        "timezone": "GMT",
    }
    if model != "best_match":
        params["models"] = model
    for attempt in range(1, 6):
        try:
            r = requests.get(URL, params=params, timeout=180)
            if r.status_code == 400:                   # неверный запрос — повтор не поможет
                raise SystemExit(f"Open-Meteo отклонил запрос: {r.text[:300]}")
            if r.status_code == 429:
                raise requests.RequestException("429 Too Many Requests")
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, list) else [data]
        except requests.RequestException as e:
            wait = 30 * attempt
            print(f"  попытка {attempt}/5 не удалась ({e}); жду {wait} с", flush=True)
            time.sleep(wait)
    raise RuntimeError("Open-Meteo не отвечает, попробуйте позже.")


def to_frame(item: dict, gid: str) -> pd.DataFrame:
    df = pd.DataFrame(item["hourly"])
    df = df.rename(columns={f"{v}_{LEAD}": v for v in VARIABLES})
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["grid_id"] = gid
    return df


def to_local(w: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """UTC -> местное время Нью-Йорка без пояса (как в download_weather*.py)."""
    w = w.copy()
    w["transit_timestamp"] = w["time"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    agg = {v: ((lambda x: x.sum(min_count=1)) if v in SUM_VARS else "mean") for v in VARIABLES}
    agg["weather_code"] = "max"
    w = w.groupby(["grid_id", "transit_timestamp"]).agg(agg)
    full = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
    parts = []
    for gid, g in w.groupby(level="grid_id"):
        g = g.droplevel("grid_id").reindex(full).ffill(limit=1)   # дыра весеннего перехода
        g.index.name = "transit_timestamp"
        g["grid_id"] = gid
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)


def compare(fc: pd.DataFrame, actual_path: Path, keys: list):
    """Насколько прогноз на сутки расходится с фактической погодой (для текста ВКР)."""
    if not actual_path.exists():
        print(f"  {actual_path.name} нет — сравнение с фактом пропущено")
        return
    act = pd.read_parquet(actual_path)
    m = fc.merge(act, on=keys, suffixes=("_fc", "_act")).dropna(
        subset=["temperature_2m_fc", "temperature_2m_act", "precipitation_fc", "precipitation_act"])
    if not len(m):
        return
    t_mae = (m["temperature_2m_fc"] - m["temperature_2m_act"]).abs().mean()
    t_bias = (m["temperature_2m_fc"] - m["temperature_2m_act"]).mean()
    rain_a, rain_f = m["precipitation_act"] >= 0.5, m["precipitation_fc"] >= 0.5
    hit = (rain_a & rain_f).sum() / max(rain_a.sum(), 1)
    false = (~rain_a & rain_f).sum() / max(rain_f.sum(), 1)
    # суточные суммы осадков — так их видит пассажир
    day = m.assign(d=m["transit_timestamp"].dt.date).groupby(keys[:-1] + ["d"])[
        ["precipitation_fc", "precipitation_act"]].sum()
    day_corr = np.corrcoef(day["precipitation_fc"], day["precipitation_act"])[0, 1]
    print(f"  Прогноз против факта ({len(m):,} часов):")
    print(f"    температура: MAE {t_mae:.2f} °C, смещение {t_bias:+.2f} °C")
    print(f"    часы с осадками ≥ 0,5 мм: прогноз поймал {hit:.0%}, ложных тревог {false:.0%}")
    print(f"    корреляция суточных сумм осадков: {day_corr:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=None, help="шаг сетки; без параметра — одна точка")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-09-24")
    ap.add_argument("--model", default="best_match", help="best_match, gfs_seamless, ecmwf_ifs025 и др.")
    ap.add_argument("--stations", default="data/raw/stations.csv")
    ap.add_argument("--outdir", default="data/raw")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if args.step is None:
        nodes = pd.DataFrame({"grid_id": ["city"], "grid_lat": [LAT], "grid_lon": [LON]})
        tag = "hourly"
    else:
        stations = pd.read_csv(args.stations, dtype={"station_complex_id": str})
        nodes, _ = build_grid(stations, args.step)      # те же узлы, что в weather_grid_<шаг>
        tag = f"grid_{args.step:g}"
    print(f"Прогноз погоды ({LEAD}, модель {args.model}): {len(nodes)} узл., {args.start} – {args.end}")

    parts_dir = outdir / "weather_forecast_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.start) - pd.Timedelta(days=1)    # запас под перевод UTC -> Нью-Йорк
    end = pd.Timestamp(args.end) + pd.Timedelta(days=1)
    months = pd.period_range(start, end, freq="M")
    raw = []
    for p in months:
        a, b = max(start, p.start_time), min(end, p.end_time.normalize())
        for i in range(0, len(nodes), BATCH):
            chunk = nodes.iloc[i:i + BATCH]
            path = parts_dir / f"{tag}_{args.model}_{p}_{i:03d}.parquet"
            if not path.exists():
                data = fetch(chunk["grid_lat"], chunk["grid_lon"], f"{a:%Y-%m-%d}", f"{b:%Y-%m-%d}", args.model)
                pd.concat([to_frame(it, g) for it, g in zip(data, chunk["grid_id"])]).to_parquet(path, index=False)
                print(f"  {p}: узлы {i + 1}–{i + len(chunk)}", flush=True)
                time.sleep(3)
            raw.append(pd.read_parquet(path))
    raw = pd.concat(raw).drop_duplicates(["grid_id", "time"])

    w = to_local(raw, args.start, args.end)
    for c in VARIABLES:
        w[c] = w[c].astype("float32")

    # Покрытие по месяцам: если в начале периода пусто — у выбранной модели архив короче
    cov = w.assign(m=w["transit_timestamp"].dt.to_period("M")).groupby("m")["temperature_2m"].apply(
        lambda s: s.notna().mean())
    empty = cov[cov < 0.95]
    print(f"  часов: {w['transit_timestamp'].nunique():,}; заполнено {w['temperature_2m'].notna().mean():.1%}")
    if len(empty):
        print("  месяцы с пропусками (> 5%), в скобках доля пропусков: "
              + ", ".join(f"{m} ({1 - v:.0%})" for m, v in empty.items()))

    if args.step is None:
        w = w.drop(columns="grid_id")
        out = outdir / "weather_forecast_hourly.parquet"
        w.to_parquet(out, index=False)
        compare(w, outdir / "weather_hourly.parquet", ["transit_timestamp"])
    else:
        w = w.merge(nodes, on="grid_id")
        out = outdir / f"weather_forecast_{tag}.parquet"
        w.to_parquet(out, index=False)
        compare(w.drop(columns=["grid_lat", "grid_lon"]), outdir / f"weather_grid_{args.step:g}.parquet",
                ["grid_id", "transit_timestamp"])
    print(f"Сохранено: {out.resolve()}")


if __name__ == "__main__":
    main()
