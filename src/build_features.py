"""
Финальная сборка признаков поверх результата build_dataset.py.

Создаёт в data/processed:
  calendar_ext.parquet     — час: календарь + режимы + городские события + циклические признаки
  station_static.parquet   — станция: масштаб, тип (commute_index, кластер), линии, зона, площадка, OD-степени
  events_station_hour.parquet — станция × час: события на площадке рядом (только часы с событиями)
  graph/                   — граф станций: порядок узлов и матрицы смежности (расстояние, линии, OD
                             полный и разреженный top-10)
  lab/                     — экспорт для моделей временных рядов (широкий формат: час × станция)

Всё, что «учится» на данных (масштаб станции, commute_index, кластеры), считается ТОЛЬКО
по обучающему периоду (до config.TRAIN_END), чтобы в признаки не протекала информация из val/test.

Запуск (из корня проекта, после build_dataset.py; события и OD — по желанию):
  python src/download_events.py      # необязательно: data/raw/events.csv
  python src/download_od.py          # необязательно: data/raw/od/*.parquet
  python src/build_features.py
"""

import json
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from config import (CONGESTION_PRICING_START, DATA_END, DATA_START, PROCESSED, RAW,
                    STUDENT_OMNY_START, SPLITS, TRAIN_END)

GRAPH = PROCESSED / "graph"
LAB = PROCESSED / "lab"

# ---------------------------------------------------------------------------
# Городские события (день целиком, для всех станций). Подтверждены всплесками в данных
# (ноутбуки 02–03) или известны по правилу (дата парада).
# ---------------------------------------------------------------------------
CITY_EVENTS = {
    "marathon": ["2023-11-05", "2024-11-03", "2025-11-02"],              # первое воскресенье ноября
    "pride_march": ["2023-06-25", "2024-06-30", "2025-06-29", "2026-06-28"],  # последнее вс июня
    "puerto_rican_parade": ["2023-06-11", "2024-06-09", "2025-06-08", "2026-06-14"],  # 2-е вс июня
    "halloween_parade": ["2023-10-31", "2024-10-31", "2025-10-31"],
    "santacon": ["2023-12-09", "2024-12-14", "2025-12-13"],
    # ЧМ по футболу 2026, MetLife Stadium (болельщики едут через Penn Station). Только val/test!
    "world_cup_2026": ["2026-06-13", "2026-06-16", "2026-06-22", "2026-06-25", "2026-06-27",
                       "2026-06-30", "2026-07-05", "2026-07-19"],
}

# Площадка -> станции рядом (по подстроке названия станционного комплекса)
VENUE_STATIONS = {
    "Yankee Stadium": ["161 St-Yankee Stadium"],
    "Citi Field / USTA": ["Mets-Willets Point"],
    "Maimonides Park": ["Coney Island-Stillwell Av", "W 8 St-NY Aquarium"],
    "Barclays Center": ["Atlantic Av-Barclays Ctr"],
}
KIND_VENUE = {
    "mlb_yankees": "Yankee Stadium", "mlb_mets": "Citi Field / USTA",
    "usopen_main": "Citi Field / USTA", "usopen_fanweek": "Citi Field / USTA",
    "milb_cyclones": "Maimonides Park", "nba_nets": "Barclays Center", "wnba_liberty": "Barclays Center",
}
EVENT_WINDOW = (-2, 5)
OD_TOP_K = 10            # разреженный OD-граф: 10 главных направлений поездок с каждой станции   # часы относительно начала события, в которые ставится признак


# ---------------------------------------------------------------------------
# Календарь
# ---------------------------------------------------------------------------
def build_calendar_ext() -> pd.DataFrame:
    cal = pd.read_parquet(PROCESSED / "calendar.parquet")
    ts = cal["transit_timestamp"]
    d = ts.dt.normalize()

    cal["month"] = ts.dt.month.astype("int8")
    cal["dayofyear"] = ts.dt.dayofyear.astype("int16")
    cal["week_of_year"] = ts.dt.isocalendar().week.astype("int8").values
    for name, x, period in [("hour", cal["hour"], 24), ("dow", cal["dow"], 7), ("doy", cal["dayofyear"], 365.25)]:
        cal[f"{name}_sin"] = np.sin(2 * np.pi * x / period).astype("float32")
        cal[f"{name}_cos"] = np.cos(2 * np.pi * x / period).astype("float32")

    cal["regime_student_omny"] = (ts >= STUDENT_OMNY_START).astype("int8")
    cal["cp_active"] = (ts >= CONGESTION_PRICING_START).astype("int8")

    md = ts.dt.month * 100 + ts.dt.day
    cal["xmas_week"] = ((md >= 1224) | (md <= 101)).astype("int8")                 # 24.12 – 01.01
    cal["holiday_shopping_weekend"] = ((ts.dt.month == 12) & (ts.dt.day <= 23) &
                                       (cal["dow"] >= 5)).astype("int8")

    ev = {pd.Timestamp(x): k for k, days in CITY_EVENTS.items() for x in days}
    cal["city_event"] = d.map(ev).fillna("").astype("category")
    cal["is_city_event"] = (cal["city_event"] != "").astype("int8")
    return cal


# ---------------------------------------------------------------------------
# Станции
# ---------------------------------------------------------------------------
def lines_of(name: str) -> list:
    groups = re.findall(r"\(([^)]*)\)", name)
    return sorted({x.strip() for g in groups for x in g.split(",") if x.strip()})


def build_station_static(panel: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    st = pd.read_parquet(PROCESSED / "stations.parquet")
    st["lines"] = st["station_complex"].map(lambda s: ",".join(lines_of(s)))
    st["n_lines"] = st["lines"].map(lambda s: len(s.split(",")) if s else 0).astype("int8")

    # Обучающий период, открытые станции, будни без праздников
    hol_ts = cal.loc[cal["is_holiday"] == 1, "transit_timestamp"]
    tr = panel[(panel["transit_timestamp"] <= TRAIN_END) & (panel["is_closed"] == 0)].copy()
    tr["dow"] = tr["transit_timestamp"].dt.dayofweek
    tr["hour"] = tr["transit_timestamp"].dt.hour
    tr = tr[(tr["dow"] < 5) & ~tr["transit_timestamp"].isin(hol_ts)]

    daily = tr.groupby(["station_complex_id", tr["transit_timestamp"].dt.normalize()],
                       observed=True)["ridership"].sum()
    scale = daily.groupby("station_complex_id", observed=True).mean().rename("scale_weekday")

    prof = tr.groupby(["station_complex_id", "hour"], observed=True)["ridership"].mean().unstack().fillna(0)
    shares = prof.div(prof.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    commute = (shares.loc[:, 6:9].sum(axis=1) - shares.loc[:, 16:19].sum(axis=1)).rename("commute_index")

    km = KMeans(4, n_init=20, random_state=42).fit(shares)
    lab = pd.Series(km.labels_, index=shares.index)
    order = shares.loc[:, 6:9].sum(axis=1).groupby(lab).mean().sort_values(ascending=False).index
    cluster = lab.map({old: new for new, old in enumerate(order)}).rename("cluster")   # 0 = утренние ... 3 = деловые

    feats = pd.concat([scale, commute, cluster], axis=1)
    feats.index = feats.index.astype(str)
    st = st.merge(feats, left_on="station_complex_id", right_index=True, how="left")
    st["scale_weekday"] = st["scale_weekday"].astype("float32")
    st["commute_index"] = st["commute_index"].astype("float32")

    st["venue"] = ""
    for venue, pats in VENUE_STATIONS.items():
        for p in pats:
            st.loc[st["station_complex"].str.contains(p, regex=False), "venue"] = venue
    missing = [v for v in VENUE_STATIONS if not (st["venue"] == v).any()]
    if missing:
        print(f"  ВНИМАНИЕ: не найдены станции для площадок {missing}")
    return st


# ---------------------------------------------------------------------------
# События на площадках
# ---------------------------------------------------------------------------
def build_events(st: pd.DataFrame) -> pd.DataFrame | None:
    path = RAW / "events.csv"
    if not path.exists():
        print("  data/raw/events.csv нет — признаки событий пропущены (запустите download_events.py)")
        return None
    ev = pd.read_csv(path, parse_dates=["start"])
    ev = ev[(ev["start"] >= DATA_START - pd.Timedelta(days=1)) & (ev["start"] <= DATA_END + pd.Timedelta(days=1))]
    ev["venue_key"] = ev["kind"].map(KIND_VENUE)
    venue_st = st[st["venue"] != ""][["station_complex_id", "venue"]]

    rows = []
    lo, hi = EVENT_WINDOW
    for _, e in ev.iterrows():
        start_h = e["start"].floor("h")
        for sid in venue_st.loc[venue_st["venue"] == e["venue_key"], "station_complex_id"]:
            for k in range(lo, hi + 1):
                rows.append((sid, start_h + pd.Timedelta(hours=k), k, e["kind"]))
    w = pd.DataFrame(rows, columns=["station_complex_id", "transit_timestamp", "ev_hours_from_start", "ev_kind"])

    # Несколько событий в один час (US Open + матч Mets): считаем число, берём ближайшее по времени
    w["absk"] = w["ev_hours_from_start"].abs()
    w = w.sort_values("absk")
    agg = (w.groupby(["station_complex_id", "transit_timestamp"])
             .agg(ev_active=("ev_kind", "size"), ev_hours_from_start=("ev_hours_from_start", "first"),
                  ev_kind=("ev_kind", "first"))
             .reset_index())

    # Событие сегодня на площадке у станции (число событий за день) — на все часы дня
    day = (ev.assign(date=ev["start"].dt.normalize())
             .merge(venue_st, left_on="venue_key", right_on="venue")
             .groupby(["station_complex_id", "date"]).size().rename("ev_today").reset_index())
    hours = pd.DataFrame({"h": range(24)})
    day = day.merge(hours, how="cross")
    day["transit_timestamp"] = day["date"] + pd.to_timedelta(day["h"], unit="h")
    out = day[["station_complex_id", "transit_timestamp", "ev_today"]].merge(
        agg, on=["station_complex_id", "transit_timestamp"], how="outer")
    out["ev_today"] = out["ev_today"].fillna(0).astype("int8")
    out["ev_active"] = out["ev_active"].fillna(0).astype("int8")
    out["ev_hours_from_start"] = out["ev_hours_from_start"].astype("float32")
    out["ev_kind"] = out["ev_kind"].fillna("").astype("category")
    out = out[(out["transit_timestamp"] >= DATA_START) & (out["transit_timestamp"] <= DATA_END)]
    print(f"  событий: {len(ev)}; станция×часов с признаками: {len(out):,}")
    print("  " + ev.groupby("kind").size().to_string().replace("\n", "\n  "))
    return out


# ---------------------------------------------------------------------------
# Граф станций
# ---------------------------------------------------------------------------
def distances_km(st: pd.DataFrame) -> np.ndarray:
    lat, lon = np.radians(st["latitude"].values), np.radians(st["longitude"].values)
    dlat = lat[:, None] - lat[None, :]
    dlon = (lon[:, None] - lon[None, :]) * np.cos((lat[:, None] + lat[None, :]) / 2)
    return 6371 * np.sqrt(dlat ** 2 + dlon ** 2)


def save_edges(name: str, A: np.ndarray, ids: list):
    i, j = np.nonzero(A)
    pd.DataFrame({"src": np.array(ids)[i], "dst": np.array(ids)[j], "weight": A[i, j]}).to_csv(
        GRAPH / f"edges_{name}.csv", index=False)
    np.save(GRAPH / f"adj_{name}.npy", A.astype("float32"))
    n = len(ids)
    print(f"  adj_{name}: рёбер {len(i):,} (плотность {len(i) / (n * (n - 1)):.2%}), "
          f"изолированных узлов {int(((A > 0).sum(axis=1) == 0).sum())}")


def build_graph(st: pd.DataFrame) -> pd.DataFrame:
    GRAPH.mkdir(parents=True, exist_ok=True)
    st = st.sort_values("station_complex_id").reset_index(drop=True)
    ids = st["station_complex_id"].tolist()
    st[["station_complex_id", "station_complex", "borough", "latitude", "longitude", "lines"]].rename_axis(
        "node").to_csv(GRAPH / "nodes.csv")
    D = distances_km(st)
    np.save(GRAPH / "dist_km.npy", D.astype("float32"))

    # 1. Расстояние: гауссово ядро exp(-(d/σ)²), σ = медиана расстояния до 5-го ближайшего соседа;
    #    веса < 0.1 обнуляются (как в DCRNN/STGCN).
    sigma = np.median(np.sort(D, axis=1)[:, 5])
    A = np.exp(-(D / sigma) ** 2)
    A[A < 0.1] = 0
    np.fill_diagonal(A, 0)
    print(f"  σ для гауссова ядра: {sigma:.2f} км")
    save_edges("dist", A, ids)

    # 2. Линии: каждая станция соединена с 2 ближайшими станциями на каждой своей линии
    #    (приближение последовательности станций на линии).
    L = np.zeros_like(D)
    lines = st["lines"].map(lambda s: set(s.split(",")) if s else set())
    for line in sorted(set().union(*lines)):
        idx = np.where(lines.map(lambda x: line in x))[0]
        if len(idx) < 2:
            continue
        sub = D[np.ix_(idx, idx)]
        for a, row in zip(idx, sub):
            for b in idx[np.argsort(row)[1:3]]:
                L[a, b] = L[b, a] = 1
    save_edges("line", L, ids)

    # 3. OD: средний поток поездок i -> j в будний день (по скачанным месяцам), и в утренний пик
    files = sorted((RAW / "od").glob("od_*.parquet"))
    if not files:
        print("  OD-файлов нет — граф по потокам пропущен (запустите download_od.py)")
        return st
    od = pd.concat(pd.read_parquet(f).assign(src_file=f.name) for f in files)
    od = od[~od["day_of_week"].isin(["Saturday", "Sunday"])]
    od = od.groupby(["origin", "destination"])[["flow_day", "flow_am"]].sum() / \
         od.groupby(["src_file", "day_of_week"]).ngroups              # среднее по (месяц × будний день)
    od = od.reset_index()
    pos = {s: k for k, s in enumerate(ids)}
    lost = (~od["origin"].isin(pos) | ~od["destination"].isin(pos)).mean()
    od = od[od["origin"].isin(pos) & od["destination"].isin(pos) & (od["origin"] != od["destination"])]
    for col, name in [("flow_day", "od"), ("flow_am", "od_am")]:
        W = np.zeros_like(D)
        W[od["origin"].map(pos).values, od["destination"].map(pos).values] = od[col].values
        save_edges(name, W, ids)
        # MTA даёт оценку почти для каждой пары станций (плотность ~98%), такой граф бесполезен для
        # графовых моделей. Разреженная версия: для каждой станции K направлений с наибольшим потоком.
        top = np.argsort(-W, axis=1)[:, :OD_TOP_K]
        Wk = np.zeros_like(W)
        rows = np.arange(len(ids))[:, None]
        Wk[rows, top] = W[rows, top]
        Wk[Wk < 1] = 0                                   # меньше 1 поездки в день — не связь
        save_edges(f"{name}_top{OD_TOP_K}", Wk, ids)
    print(f"  OD-пар со станциями вне панели: {lost:.2%} (отброшены)")
    W = np.load(GRAPH / "adj_od.npy")
    st["od_out_flow"] = W.sum(axis=1).astype("float32")
    st["od_in_flow"] = W.sum(axis=0).astype("float32")
    st["od_out_degree"] = (W >= 1).sum(axis=1).astype("int16")      # куда едут хотя бы 1 раз в день
    return st


# ---------------------------------------------------------------------------
# Экспорт для лабораторной (широкий формат)
# ---------------------------------------------------------------------------
def export_lab(panel: pd.DataFrame, cal: pd.DataFrame, st: pd.DataFrame):
    LAB.mkdir(parents=True, exist_ok=True)
    ids = sorted(st["station_complex_id"])
    p = panel.copy()
    p["station_complex_id"] = p["station_complex_id"].astype(str)
    y = p.pivot(index="transit_timestamp", columns="station_complex_id", values="ridership")[ids]
    closed = p.pivot(index="transit_timestamp", columns="station_complex_id", values="is_closed")[ids]
    y.astype("int32").to_parquet(LAB / "ridership_wide.parquet")
    closed.astype("bool").to_parquet(LAB / "closed_wide.parquet")

    w = pd.read_parquet(RAW / "weather_hourly.parquet")
    cov = cal.merge(w, on="transit_timestamp", how="left")
    cov = cov.drop(columns=["holiday_name", "city_event"]).set_index("transit_timestamp")
    cov.astype("float32").to_parquet(LAB / "covariates_time.parquet")
    st.to_parquet(LAB / "station_static.parquet", index=False)

    meta = {k: [str(a), str(b)] for k, (a, b) in SPLITS.items()}
    meta.update({"n_stations": len(ids), "n_hours": len(y), "freq": "1h", "timezone": "America/New_York (naive)",
                 "station_order": "columns of ridership_wide = nodes.csv order (sorted station_complex_id)"})
    (LAB / "split.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ridership_wide: {y.shape[0]:,} часов × {y.shape[1]} станций; "
          f"covariates_time: {cov.shape[1]} признаков")


def main():
    print("Календарь...")
    cal = build_calendar_ext()
    cal.to_parquet(PROCESSED / "calendar_ext.parquet", index=False)
    print(f"  дней с городскими событиями: {cal.loc[cal['is_city_event'] == 1, 'transit_timestamp'].dt.date.nunique()}")

    print("Станции...")
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    panel["station_complex_id"] = panel["station_complex_id"].astype(str)
    st = build_station_static(panel, cal)

    print("Граф...")
    st = build_graph(st)
    st.to_parquet(PROCESSED / "station_static.parquet", index=False)
    print(f"  станций: {len(st)}; признаков станции: {st.shape[1]}")

    print("События...")
    ev = build_events(st)
    if ev is not None:
        ev.to_parquet(PROCESSED / "events_station_hour.parquet", index=False)

    print("Экспорт для лабораторной...")
    export_lab(panel, cal, st)
    print(f"Готово: {PROCESSED.resolve()}")


if __name__ == "__main__":
    main()
