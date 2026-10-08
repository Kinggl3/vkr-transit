"""
Загрузка почасового пассажиропотока метро Нью-Йорка (MTA) с data.ny.gov.

Что делает:
  * качает данные кусками по 7 дней через Socrata API (SoQL);
  * прямо на сервере суммирует все способы оплаты и тарифы,
    чтобы получить одну строку на (час, станционный комплекс);
  * берёт только метро (без Staten Island Railway и трамвая на Рузвельт-Айленд);
  * сохраняет каждый месяц в отдельный parquet;
  * попутно собирает справочник станций (название, район, координаты).

Запуск (из корня проекта):
  pip install requests pandas pyarrow tqdm
  python src/download_mta.py --start 2023-01 --end 2026-09

Необязательно, но снижает риск ограничений: бесплатный app token
(https://data.ny.gov/profile/edit/developer_settings) в переменной SOCRATA_APP_TOKEN.
Повторный запуск докачивает только недостающие месяцы.
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

BASE = "https://data.ny.gov/resource/{dataset}.json"
DATASET_2020_2024 = "wujg-7c2s"   # MTA Subway Hourly Ridership: 2020-2024
DATASET_2025_PLUS = "5wq4-mkjj"   # MTA Subway Hourly Ridership: Beginning 2025
PAGE = 50_000
CHUNK_DAYS = 7
TIMEOUT = 120

STATION_COLS = ["station_complex_id", "station_complex", "borough", "latitude", "longitude"]

HEADERS = {}
if os.getenv("SOCRATA_APP_TOKEN"):
    HEADERS["X-App-Token"] = os.environ["SOCRATA_APP_TOKEN"]


def dataset_for(day: pd.Timestamp) -> str:
    return DATASET_2025_PLUS if day.year >= 2025 else DATASET_2020_2024


def get(url: str, params: dict, retries: int = 5) -> list:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            wait = 15 * attempt
            tqdm.write(f"  попытка {attempt}/{retries} не удалась ({type(e).__name__}: {e}); "
                       f"жду {wait} с")
            time.sleep(wait)
    raise RuntimeError("Сервер не отвечает. Попробуйте позже или уменьшите CHUNK_DAYS.")


def download_range(start: pd.Timestamp, end: pd.Timestamp) -> list:
    """Агрегированные строки за [start, end): одна строка на (час, станция)."""
    group = "transit_timestamp, " + ", ".join(STATION_COLS)
    params = {
        "$select": group + ", sum(ridership) AS ridership, sum(transfers) AS transfers",
        "$where": f"transit_mode = 'subway' "
                  f"AND transit_timestamp >= '{start:%Y-%m-%dT%H:%M:%S}' "
                  f"AND transit_timestamp < '{end:%Y-%m-%dT%H:%M:%S}'",
        "$group": group,
        "$order": "transit_timestamp, station_complex_id",
        "$limit": PAGE,
    }
    url = BASE.format(dataset=dataset_for(start))
    rows, offset = [], 0
    while True:
        batch = get(url, {**params, "$offset": offset})
        rows.extend(batch)
        if len(batch) < PAGE:
            return rows
        offset += PAGE


def download_month(month: pd.Timestamp) -> pd.DataFrame:
    month_end = month + pd.offsets.MonthBegin(1)
    bounds = list(pd.date_range(month, month_end, freq=f"{CHUNK_DAYS}D"))
    if bounds[-1] != month_end:
        bounds.append(month_end)

    rows = []
    for a, b in tqdm(list(zip(bounds[:-1], bounds[1:])), desc=f"  {month:%Y-%m}", leave=False):
        rows.extend(download_range(a, b))

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["transit_timestamp"] = pd.to_datetime(df["transit_timestamp"])
    df["ridership"] = df["ridership"].astype(float).round().astype("int32")
    df["transfers"] = df["transfers"].astype(float).round().astype("int32")
    df[["latitude", "longitude"]] = df[["latitude", "longitude"]].astype(float)
    return df


def build_stations(out_dir: Path, stations_path: Path) -> None:
    """Справочник станций из всех скачанных месяцев (последние данные важнее)."""
    files = sorted(out_dir.glob("*_stations.csv"))
    if not files:
        return
    st = (pd.concat(pd.read_csv(f, dtype={"station_complex_id": str}) for f in files)
            .drop_duplicates("station_complex_id", keep="last")
            .sort_values("station_complex_id"))
    st.to_csv(stations_path, index=False)
    print(f"Справочник: {len(st)} станций -> {stations_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01", help="первый месяц, ГГГГ-ММ")
    ap.add_argument("--end", default="2026-09", help="последний месяц, ГГГГ-ММ")
    ap.add_argument("--out", default="data/raw/mta_hourly")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Сохраняю в: {out.resolve()}")
    print("Проверяю связь с data.ny.gov...", flush=True)
    get(BASE.format(dataset=DATASET_2025_PLUS), {"$limit": 1})
    print("Связь есть, начинаю загрузку.\n", flush=True)

    months = pd.date_range(args.start, args.end, freq="MS")
    for m in tqdm(months, desc="Месяцы"):
        path = out / f"{m:%Y-%m}.parquet"
        if path.exists():
            continue
        df = download_month(m)
        if df.empty:
            tqdm.write(f"{m:%Y-%m}: данных нет (возможно, ещё не опубликованы)")
            continue

        df[STATION_COLS].drop_duplicates("station_complex_id", keep="last").to_csv(
            out / f"{m:%Y-%m}_stations.csv", index=False)

        # Название/координаты станции могут меняться внутри месяца, и тогда сервер
        # возвращает несколько строк на один (час, id). Досуммируем до одной строки.
        df = (df.groupby(["transit_timestamp", "station_complex_id"], as_index=False)
                [["ridership", "transfers"]].sum())
        dups = df.duplicated(["transit_timestamp", "station_complex_id"]).sum()
        assert dups == 0, f"остались дубликаты: {dups}"
        df.to_parquet(path, index=False)

        hours = df["transit_timestamp"].nunique()
        tqdm.write(f"{m:%Y-%m}: {len(df):,} строк, станций {df['station_complex_id'].nunique()}, "
                   f"часов {hours}/{m.days_in_month * 24}")

    build_stations(out, out.parent / "stations.csv")


if __name__ == "__main__":
    main()