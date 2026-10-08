"""
Матрица пассажиропотоков между станциями (origin–destination) от MTA.

Набор MTA содержит оценку среднего числа поездок для каждой пары станций по месяцу × дню недели × часу
(в 2024 году — 116 млн строк). Для графа станций весь набор не нужен: на сервере суммируем
по часам и получаем для каждой пары средний суточный поток в каждый день недели. Отдельно —
поток в утренний пик (6–9 ч), он показывает направленные связи «дом → работа».
Запросы идут по частям (день недели × интервал часов), каждая часть сохраняется сразу,
поэтому при обрыве повторный запуск продолжит с места остановки.

Берём два типичных месяца (по умолчанию март и октябрь): без праздников и летнего сезона.

Запуск (из корня проекта, через VPN):
  python src/download_od.py                    # 2024 год, март и октябрь
  python src/download_od.py --months 3 4 10 11 # больше месяцев — надёжнее, но дольше
Результат: data/raw/od/od_<год>_<месяц>.parquet
  origin, destination, day_of_week, flow_day (поездок в средний такой день), flow_am (из них 6–9 ч)
Повторный запуск докачивает только недостающие месяцы.
"""

import argparse
import os
import time

import pandas as pd
import requests

from config import RAW

DATASETS = {2024: "jsu2-fbtj", 2025: "y2qv-fytt"}
PAGE = 50_000
HEADERS = {"X-App-Token": os.environ["SOCRATA_APP_TOKEN"]} if os.getenv("SOCRATA_APP_TOKEN") else {}


def get(url, params, retries=6):
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=300)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            wait = 20 * attempt
            print(f"  попытка {attempt}/{retries} не удалась ({type(e).__name__}); жду {wait} с", flush=True)
            time.sleep(wait)
    raise RuntimeError("Сервер не отвечает. Попробуйте позже.")


def query_all(url, params):
    rows, offset = [], 0
    while True:
        batch = get(url, {**params, "$limit": PAGE, "$offset": offset})
        rows.extend(batch)
        if len(batch) < PAGE:
            return rows
        offset += PAGE


# Часовые интервалы: сервер суммирует по ~250 тыс. строк за запрос и укладывается в тайм-аут.
# Интервал 6–9 ч одновременно даёт поток в утренний пик.
BANDS = [(0, 5), (6, 9), (10, 15), (16, 19), (20, 23)]


def download_part(url, month, dow, lo, hi):
    return query_all(url, {
        "$select": "origin_station_complex_id AS origin, destination_station_complex_id AS destination, "
                   "sum(estimated_average_ridership) AS flow",
        "$where": f"month = {month} AND day_of_week = '{dow}' AND hour_of_day >= {lo} AND hour_of_day <= {hi}",
        "$group": "origin_station_complex_id, destination_station_complex_id",
        "$order": "origin_station_complex_id, destination_station_complex_id",
    })


def download_month(url, year, month, parts_dir):
    dows = [r["day_of_week"] for r in get(url, {
        "$select": "day_of_week", "$where": f"month = {month}", "$group": "day_of_week"})]
    print(f"  дни недели в данных: {dows}", flush=True)
    frames = []
    for dow in dows:
        for lo, hi in BANDS:
            path = parts_dir / f"{year}_{month:02d}_{dow}_{lo:02d}-{hi:02d}.parquet"
            if not path.exists():                       # докачка: готовые части не перекачиваются
                d = pd.DataFrame(download_part(url, month, dow, lo, hi))
                d.to_parquet(path, index=False)
                print(f"  {dow} {lo:02d}–{hi:02d} ч: {len(d):,} пар", flush=True)
            d = pd.read_parquet(path)
            if len(d):
                d["day_of_week"], d["band"] = dow, f"{lo}-{hi}"
                frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["origin"] = df["origin"].astype(str)
    df["destination"] = df["destination"].astype(str)
    df["flow"] = df["flow"].astype(float)
    keys = ["origin", "destination", "day_of_week"]
    day = df.groupby(keys)["flow"].sum().rename("flow_day")
    am = df[df["band"] == "6-9"].groupby(keys)["flow"].sum().rename("flow_am")
    return pd.concat([day, am], axis=1).fillna({"flow_am": 0}).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024, choices=sorted(DATASETS))
    ap.add_argument("--months", type=int, nargs="+", default=[3, 10])
    args = ap.parse_args()

    url = f"https://data.ny.gov/resource/{DATASETS[args.year]}.json"
    out_dir = RAW / "od"
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    for m in args.months:
        path = out_dir / f"od_{args.year}_{m:02d}.parquet"
        if path.exists():
            print(f"{path.name} уже есть, пропускаю")
            continue
        print(f"{args.year}-{m:02d}:", flush=True)
        df = download_month(url, args.year, m, parts_dir)
        df.to_parquet(path, index=False)
        print(f"  сохранено {len(df):,} строк, суммарный поток в средний будний день: "
              f"{df[~df['day_of_week'].isin(['Saturday', 'Sunday'])].groupby('day_of_week')['flow_day'].sum().mean():,.0f}")


if __name__ == "__main__":
    main()