"""
Сборка очищенного датасета из сырых файлов.

Сырые файлы в data/raw не меняются. Скрипт создаёт в data/processed:
  panel.parquet     — станция x час: ridership, transfers, is_closed
  calendar.parquet  — час: календарные признаки (праздники, школа, плата за въезд, DST)
  stations.parquet  — станции: название, район, координаты, зона платы за въезд,
                      узлы погодных сеток
  closures.csv      — найденные закрытия станций (для проверки и текста ВКР)

Погода в panel не вклеивается: её уровень (город / сетка 0.1 / сетка 0.25)
выбирается при загрузке через src/dataset.py, чтобы не держать три копии в памяти.

Запуск (из корня проекта):
  pip install holidays
  python src/build_dataset.py
"""

from datetime import date
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")

START = "2023-01-01"
END = "2026-09-24 23:00"          # последний час, опубликованный MTA на момент загрузки

DROP_STATIONS = {"501", "502"}    # Staten Island Railway (St George, Tompkinsville) — не метро
CLOSED_SHARE = 0.10               # день закрыт, если поток < 10% обычного уровня станции
CP_START = pd.Timestamp("2025-01-05 00:00")   # начало платы за въезд в центр Манхэттена

# Известные из новостей закрытия: используются для ПРОВЕРКИ автоматического правила.
KNOWN_CLOSURES = [
    # Рокавэй: метро до полуострова остановлено на ремонт, январь–май 2025
    {"name": "Rockaway 2025", "ids": [str(i) for i in range(199, 210)],
     "from": "2025-01-18", "to": "2025-05-14"},
]

# ---------------------------------------------------------------------------
# Школьный календарь NYC (официальные календари NYC Public Schools).
# Неучебными считаются дни, когда школы закрыты или ученики не приходят.
# Частичные дни (ранний роспуск, выходной только у части школ) считаются учебными.
# ---------------------------------------------------------------------------
SCHOOL_YEARS = [
    {   # 2022-23 (нужна только часть с января 2023)
        "first": "2022-09-08", "last": "2023-06-27",
        "closed": ["2023-01-02", "2023-01-16", ("2023-02-20", "2023-02-24"),
                   "2023-04-06", "2023-04-07", ("2023-04-10", "2023-04-14"), "2023-04-21",
                   "2023-05-29", "2023-06-08", "2023-06-19"],
    },
    {   # 2023-24
        "first": "2023-09-07", "last": "2024-06-26",
        "closed": ["2023-09-25", "2023-10-09", "2023-11-07", ("2023-11-23", "2023-11-24"),
                   ("2023-12-25", "2024-01-01"), "2024-01-15",
                   "2024-02-13",  # снегопад: удалённый день, ученики дома
                   ("2024-02-19", "2024-02-23"), ("2024-03-29", "2024-04-01"), "2024-04-10",
                   ("2024-04-22", "2024-04-30"), "2024-05-27", "2024-06-06", "2024-06-17",
                   "2024-06-19"],
    },
    {   # 2024-25
        "first": "2024-09-05", "last": "2025-06-26",
        "closed": [("2024-10-03", "2024-10-04"), "2024-10-14", "2024-11-01", "2024-11-05",
                   "2024-11-11", ("2024-11-28", "2024-11-29"), ("2024-12-23", "2025-01-01"),
                   "2025-01-20", "2025-01-29", ("2025-02-17", "2025-02-21"), "2025-03-31",
                   ("2025-04-14", "2025-04-18"), "2025-05-26", "2025-06-05", "2025-06-19"],
    },
    {   # 2025-26
        "first": "2025-09-04", "last": "2026-06-26",
        "closed": [("2025-09-23", "2025-09-24"), "2025-10-02", "2025-10-13", "2025-10-20",
                   "2025-11-04", "2025-11-11", ("2025-11-27", "2025-11-28"),
                   ("2025-12-24", "2026-01-02"), "2026-01-19", ("2026-02-16", "2026-02-20"),
                   "2026-03-20", ("2026-04-02", "2026-04-10"), "2026-05-25", "2026-05-27",
                   "2026-06-04", "2026-06-19"],
    },
    {   # 2026-27 (нужен только сентябрь 2026)
        "first": "2026-09-10", "last": "2027-06-25",
        "closed": ["2026-09-21"],
    },
]


def school_days() -> set:
    days = set()
    for y in SCHOOL_YEARS:
        closed = set()
        for c in y["closed"]:
            a, b = (c, c) if isinstance(c, str) else c
            closed.update(pd.date_range(a, b, freq="D").date)
        for d in pd.bdate_range(y["first"], y["last"]).date:
            if d not in closed:
                days.add(d)
    return days


# ---------------------------------------------------------------------------
# Календарь
# ---------------------------------------------------------------------------
def build_calendar() -> pd.DataFrame:
    naive = pd.date_range(START, END, freq="h")
    local = naive.tz_localize("America/New_York", nonexistent="NaT", ambiguous="NaT")
    cal = pd.DataFrame({"transit_timestamp": naive})
    # весной один час не существует -> выкидываем; осенью один час повторяется -> флаг
    nonexistent = local.isna() & (naive.month == 3)
    cal["is_dst_double_hour"] = (local.isna() & (naive.month == 11)).astype("int8")
    cal = cal[~nonexistent].reset_index(drop=True)

    ts = cal["transit_timestamp"]
    d = ts.dt.date
    cal["hour"] = ts.dt.hour.astype("int8")
    cal["dow"] = ts.dt.dayofweek.astype("int8")            # 0 = понедельник
    cal["is_weekend"] = (cal["dow"] >= 5).astype("int8")

    years = range(2022, 2028)
    fed = holidays.US(years=years, observed=True)           # федеральные праздники
    names = {k: v for k, v in fed.items()}
    # дни с сильно изменённым спросом, которых нет в официальном списке
    for y in years:
        thanksgiving = [k for k, v in fed.items() if k.year == y and "Thanksgiving" in v][0]
        names.setdefault(thanksgiving + pd.Timedelta(days=1), "Day after Thanksgiving")
        names.setdefault(date(y, 12, 24), "Christmas Eve")
        names.setdefault(date(y, 12, 31), "New Year's Eve")

    hol_dates = set(names)
    cal["is_holiday"] = d.isin(hol_dates).astype("int8")
    cal["holiday_name"] = d.map(names).fillna("").astype("category")
    cal["is_day_before_holiday"] = (d + pd.Timedelta(days=1)).isin(hol_dates).astype("int8")
    cal["is_day_after_holiday"] = (d - pd.Timedelta(days=1)).isin(hol_dates).astype("int8")

    cal["is_school_day"] = d.isin(school_days()).astype("int8")
    cal["cp_active"] = (ts >= CP_START).astype("int8")
    return cal


# ---------------------------------------------------------------------------
# Станции
# ---------------------------------------------------------------------------
def in_cbd(st: pd.DataFrame) -> pd.Series:
    """Зона платы за въезд: Манхэттен южнее 60-й улицы (включительно).
    60-я улица аппроксимирована прямой через 5-ю авеню (40.7651, -73.9724)
    с наклоном манхэттенской сетки (~29° от параллели)."""
    line_lat = 40.7651 - 0.42 * (st["longitude"] + 73.9724) + 0.0004
    return ((st["borough"] == "Manhattan") & (st["latitude"] < line_lat)).astype("int8")


def build_stations() -> pd.DataFrame:
    st = pd.read_csv(RAW / "stations.csv", dtype={"station_complex_id": str})
    st = st[~st["station_complex_id"].isin(DROP_STATIONS)].copy()
    st["in_cbd"] = in_cbd(st)
    for tag in ("0.1", "0.25"):
        path = RAW / f"station_weather_map_{tag}.csv"
        if path.exists():
            m = pd.read_csv(path, dtype={"station_complex_id": str})
            col = "grid_" + tag.replace(".", "")
            st = st.merge(m[["station_complex_id", "grid_id"]].rename(columns={"grid_id": col}),
                          on="station_complex_id", how="left")
    return st.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Панель станция x час и закрытия
# ---------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    files = sorted((RAW / "mta_hourly").glob("20*.parquet"))
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    n0 = len(df)
    df = df[~df["station_complex_id"].isin(DROP_STATIONS)]
    df = df[(df["transit_timestamp"] >= START) & (df["transit_timestamp"] <= END)]
    print(f"Сырых строк: {n0:,}; после удаления SIR и обрезки периода: {len(df):,}")
    return df


def build_panel(df: pd.DataFrame, stations: pd.DataFrame, hours: pd.Series) -> pd.DataFrame:
    ids = sorted(stations["station_complex_id"])
    idx = pd.MultiIndex.from_product([ids, hours], names=["station_complex_id", "transit_timestamp"])
    panel = (df.set_index(["station_complex_id", "transit_timestamp"])[["ridership", "transfers"]]
               .reindex(idx, fill_value=0).reset_index())
    panel["ridership"] = panel["ridership"].astype("int32")
    panel["transfers"] = panel["transfers"].astype("int32")
    return panel


def detect_closures(panel: pd.DataFrame) -> pd.DataFrame:
    """День закрыт, если суточный поток станции < CLOSED_SHARE от её медианы
    за тот же год и тот же день недели. Годовая медиана устойчива даже к
    закрытиям на несколько месяцев."""
    daily = (panel.assign(date=panel["transit_timestamp"].dt.normalize())
                  .groupby(["station_complex_id", "date"], observed=True)["ridership"].sum()
                  .reset_index())
    daily["year"] = daily["date"].dt.year
    daily["dow"] = daily["date"].dt.dayofweek
    daily["baseline"] = (daily.groupby(["station_complex_id", "year", "dow"], observed=True)
                              ["ridership"].transform("median"))
    daily["is_closed"] = ((daily["baseline"] > 0) &
                          (daily["ridership"] < CLOSED_SHARE * daily["baseline"])).astype("int8")
    return daily


def closure_segments(daily: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    c = daily[daily["is_closed"] == 1].sort_values(["station_complex_id", "date"])
    if c.empty:
        return pd.DataFrame(columns=["station_complex_id", "from", "to", "days"])
    new_seg = (c["station_complex_id"] != c["station_complex_id"].shift()) | \
              (c["date"].diff() != pd.Timedelta(days=1))
    c = c.assign(seg=new_seg.cumsum())
    seg = (c.groupby("seg").agg(station_complex_id=("station_complex_id", "first"),
                                **{"from": ("date", "min"), "to": ("date", "max")},
                                days=("date", "size")).reset_index(drop=True))
    seg = seg.merge(stations[["station_complex_id", "station_complex", "borough"]],
                    on="station_complex_id", how="left")
    return seg.sort_values(["from", "station_complex_id"]).reset_index(drop=True)


def check_known(daily: pd.DataFrame) -> None:
    for k in KNOWN_CLOSURES:
        m = (daily["station_complex_id"].isin(k["ids"]) &
             daily["date"].between(k["from"], k["to"]))
        share = daily.loc[m, "is_closed"].mean()
        print(f"Проверка «{k['name']}»: автоматическое правило пометило закрытыми "
              f"{share:.1%} станционных дней из известного периода")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("Календарь...")
    cal = build_calendar()
    print(f"  часов: {len(cal):,}; праздничных дней: {cal.loc[cal.is_holiday == 1, 'transit_timestamp'].dt.date.nunique()}; "
          f"учебных дней: {cal.loc[cal.is_school_day == 1, 'transit_timestamp'].dt.date.nunique()}")

    print("Станции...")
    stations = build_stations()
    print(f"  станций: {len(stations)}; в зоне платы за въезд: {stations['in_cbd'].sum()}")
    border = stations[(stations["borough"] == "Manhattan") &
                      stations["station_complex"].str.contains(r"\b(?:5[7-9]|6[0-3]) St", regex=True)]
    print("  станции у границы зоны (проверьте глазами):")
    for _, r in border.sort_values("latitude").iterrows():
        print(f"    {'В ЗОНЕ ' if r.in_cbd else 'вне   '} {r.station_complex}")

    print("Панель станция x час...")
    raw = load_raw()
    panel = build_panel(raw, stations, cal["transit_timestamp"])
    del raw
    print(f"  строк: {len(panel):,} = {stations.shape[0]} станций x {len(cal):,} часов")

    print("Закрытия станций...")
    daily = detect_closures(panel)
    check_known(daily)
    segs = closure_segments(daily, stations)
    segs.to_csv(OUT / "closures.csv", index=False)
    print(f"  закрытых станционных дней: {daily['is_closed'].sum():,} "
          f"({daily['is_closed'].mean():.2%}); отрезков: {len(segs)}; "
          f"станций с закрытиями: {segs['station_complex_id'].nunique()}")

    panel["date"] = panel["transit_timestamp"].dt.normalize()
    panel = panel.merge(daily[["station_complex_id", "date", "is_closed"]],
                        on=["station_complex_id", "date"], how="left").drop(columns="date")
    panel["station_complex_id"] = panel["station_complex_id"].astype("category")

    panel.to_parquet(OUT / "panel.parquet", index=False)
    cal.to_parquet(OUT / "calendar.parquet", index=False)
    stations.to_parquet(OUT / "stations.parquet", index=False)
    print(f"Готово: {OUT.resolve()}")


if __name__ == "__main__":
    main()