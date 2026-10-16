"""
Загрузка служебных оповещений MTA (MTA Service Alerts: Beginning April 2020) с data.ny.gov.

Набор: https://data.ny.gov/d/7kct-peq7 (тексты оповещений о задержках, плановых работах,
изменениях маршрутов; у каждого оповещения есть время публикации и обновления).

Что делает:
  * смотрит, какие колонки и какие значения agency есть в наборе (вывод на экран);
  * качает только оповещения метро (agency содержит "Subway") за [--start, --end) помесячно
    через Socrata API, не скачивая автобусы и электрички;
  * сохраняет каждый месяц в data/raw/alerts/alerts_YYYY-MM.parquet;
  * в конце печатает короткую сводку и пишет выборку из 2000 строк
    в data/raw/alerts_sample.csv (маленький файл, его можно прислать для проверки).

Запуск (из корня проекта):
  python src/download_alerts.py --start 2023-01 --end 2026-10

Повторный запуск докачивает только недостающие месяцы.
Необязательно: app token в переменной SOCRATA_APP_TOKEN (как в download_mta.py).
"""

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

URL = "https://data.ny.gov/resource/7kct-peq7.json"
PAGE = 50_000
TIMEOUT = 120
OUT = Path("data/raw/alerts")
SAMPLE = Path("data/raw/alerts_sample.csv")

HEADERS = {}
if os.getenv("SOCRATA_APP_TOKEN"):
    HEADERS["X-App-Token"] = os.environ["SOCRATA_APP_TOKEN"]


def get(params: dict, retries: int = 5) -> list:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(URL, params=params, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            wait = 15 * attempt
            tqdm.write(f"  попытка {attempt}/{retries} не удалась ({type(e).__name__}: {e}); жду {wait} с")
            time.sleep(wait)
    raise RuntimeError("Сервер не отвечает, попробуйте позже.")


def detect_columns() -> tuple[str, str | None, list[str]]:
    """Возвращает (колонка даты, колонка agency, все колонки) по одной строке набора."""
    row = get({"$limit": 1})[0]
    cols = list(row)
    print("Колонки набора:", cols)
    date_col = next((c for c in cols if c == "date"), None) or next(c for c in cols if "date" in c)
    agency_col = next((c for c in cols if "agency" in c), None)
    return date_col, agency_col, cols


def subway_filter(agency_col: str | None) -> str:
    if agency_col is None:
        print("Колонки agency нет: качаю всё, фильтровать будем потом.")
        return "1=1"
    stats = get({"$select": f"{agency_col}, count(*) AS n", "$group": agency_col,
                 "$order": "n DESC", "$limit": 100})
    print("\nОповещений по agency (весь набор):")
    for s in stats:
        print(f"  {str(s.get(agency_col)):<45} {s['n']}")
    return f"upper({agency_col}) like '%SUBWAY%'"


def download_month(month: pd.Period, date_col: str, where_agency: str) -> pd.DataFrame:
    start, end = month.start_time, (month + 1).start_time
    params = {
        "$where": f"{where_agency} AND {date_col} >= '{start:%Y-%m-%dT%H:%M:%S}' "
                  f"AND {date_col} < '{end:%Y-%m-%dT%H:%M:%S}'",
        "$order": f"{date_col}, :id",
        "$limit": PAGE,
    }
    rows, offset = [], 0
    while True:
        batch = get({**params, "$offset": offset})
        rows.extend(batch)
        if len(batch) < PAGE:
            return pd.DataFrame(rows)
        offset += PAGE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--end", default="2026-10", help="месяц не включается")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    date_col, agency_col, _ = detect_columns()
    where_agency = subway_filter(agency_col)

    months = pd.period_range(a.start, pd.Period(a.end) - 1, freq="M")
    for m in tqdm(months, desc="месяцы"):
        f = OUT / f"alerts_{m}.parquet"
        if f.exists():
            continue
        df = download_month(m, date_col, where_agency)
        df.to_parquet(f, index=False)
        tqdm.write(f"  {m}: {len(df):>7} строк")

    df = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob("alerts_*.parquet"))], ignore_index=True)
    df[date_col] = pd.to_datetime(df[date_col])
    print("\n=== Сводка ===")
    print("Строк:", len(df))
    print("Период:", df[date_col].min(), "—", df[date_col].max())
    for c in df.columns:
        nun = df[c].nunique(dropna=True)
        print(f"  {c:<22} пропусков {df[c].isna().mean():6.1%}   уникальных {nun}")
    print("\nПо годам:\n", df[date_col].dt.year.value_counts().sort_index().to_string())
    for c in df.columns:
        if df[c].nunique() < 60 and c != date_col:
            print(f"\nЗначения {c}:\n", df[c].value_counts().head(25).to_string())
    size = sum(f.stat().st_size for f in OUT.glob("alerts_*.parquet")) / 2**20
    print(f"\nРазмер папки {OUT}: {size:.1f} МБ")
    df.sample(min(2000, len(df)), random_state=42).sort_values(date_col).to_csv(SAMPLE, index=False)
    print(f"Выборка для проверки: {SAMPLE}")


if __name__ == "__main__":
    main()
