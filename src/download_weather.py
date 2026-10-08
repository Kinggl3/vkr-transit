"""
Почасовая погода для Нью-Йорка из Open-Meteo Historical Weather API (реанализ ERA5).

Точка: Центральный парк (40.7812, -73.9665) — стандартная метеоточка для NYC.
Для одного города одной точки достаточно: метро покрывает ~40 x 40 км,
и погода в один час на этой территории почти одинакова.

Время приводится к местному времени Нью-Йорка (как в данных MTA):
  * Open-Meteo отдаёт UTC, переводим в America/New_York и убираем часовой пояс;
  * при переходе на зимнее время один местный час встречается дважды — усредняем
    (осадки суммируем);
  * при переходе на летнее время одного местного часа нет — заполняем соседним.

Запуск (из корня проекта):
  python src/download_weather.py --start 2023-01-01 --end 2026-09-30
Результат: data/raw/weather_hourly.parquet
"""

import argparse
from pathlib import Path

import pandas as pd
import requests

URL = "https://archive-api.open-meteo.com/v1/archive"
LAT, LON = 40.7812, -73.9665

VARIABLES = [
    "temperature_2m",        # температура, °C
    "apparent_temperature",  # «ощущается как», °C
    "relative_humidity_2m",  # влажность, %
    "precipitation",         # все осадки за час, мм
    "rain",                  # дождь, мм
    "snowfall",              # снег, см
    "snow_depth",            # высота снежного покрова, м
    "cloud_cover",           # облачность, %
    "wind_speed_10m",        # ветер, км/ч
    "weather_code",          # код погоды WMO (ясно/туман/гроза и т.д.)
]
SUM_VARS = {"precipitation", "rain", "snowfall"}


def fetch_year(start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": start, "end_date": end,
        "hourly": ",".join(VARIABLES),
        "timezone": "GMT",
    }
    r = requests.get(URL, params=params, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--out", default="data/raw/weather_hourly.parquet")
    args = ap.parse_args()

    # Берём на сутки шире, чтобы после перевода UTC -> Нью-Йорк не обрезать края.
    start = pd.Timestamp(args.start) - pd.Timedelta(days=1)
    end = pd.Timestamp(args.end) + pd.Timedelta(days=1)

    parts = []
    for y in range(start.year, end.year + 1):
        a = max(start, pd.Timestamp(f"{y}-01-01"))
        b = min(end, pd.Timestamp(f"{y}-12-31"))
        print(f"Загружаю {a:%Y-%m-%d} – {b:%Y-%m-%d}...", flush=True)
        parts.append(fetch_year(f"{a:%Y-%m-%d}", f"{b:%Y-%m-%d}"))
    w = pd.concat(parts).drop_duplicates("time")

    # UTC -> местное время без часового пояса, как transit_timestamp у MTA.
    w["transit_timestamp"] = (w["time"].dt.tz_convert("America/New_York")
                                       .dt.tz_localize(None))
    agg = {v: ((lambda x: x.sum(min_count=1)) if v in SUM_VARS else "mean") for v in VARIABLES}
    agg["weather_code"] = "max"
    w = w.groupby("transit_timestamp").agg(agg)

    full = pd.date_range(args.start, pd.Timestamp(args.end) + pd.Timedelta(hours=23), freq="h")
    w = w.reindex(full).ffill(limit=1)  # только дыра перехода на летнее время
    w.index.name = "transit_timestamp"

    gaps = w.isna().sum().sum()
    last_ok = w.dropna().index.max()
    print(f"Часов: {len(w):,}; пропусков: {gaps}; последние полные данные: {last_ok}")
    if gaps:
        print("Пропуски в конце — норма: архив Open-Meteo отстаёт от текущей даты на ~5 дней.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w.reset_index().to_parquet(out, index=False)
    print(f"Сохранено: {out.resolve()}")


if __name__ == "__main__":
    main()