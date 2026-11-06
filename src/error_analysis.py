"""
Анализ ошибок выбранной модели (гибрид hyb_blend) в сравнении с LightGBM и эталоном.

  python src/error_analysis.py --split test
  python src/error_analysis.py --split val --alerts data/raw/alerts

Что считает (results/errors_<split>_<раздел>.csv):
  sources   — откуда берётся ошибка: доля суммарной абсолютной ошибки и доля потока по категориям
              (ночь, выходные, особые дни, события у станции, сильный дождь, вчерашние закрытия, сбои по оповещениям)
  days      — ошибка и смещение по городу для каждого дня, тип дня, осадки, события, число оповещений
  stations  — ошибка каждой станции, смещение, масштаб, район, сдвиг уровня test/train
  events    — каждое событие у площадки: факт и прогноз моделей в окне события, статус (состоялось / перенесено)
  alerts    — сбои по оповещениям MTA: ошибка в часы со сбоем и без, доля крупнейших ошибок, профиль по сдвигу часов
  ev_blend  — вариант гибрида с LightGBM на станциях у площадок в часы событий (по расписанию);
              прогноз сохраняется как hyb_ev — правило проверяется на val, на test считается один раз

Оповещения MTA (data.ny.gov 7kct-peq7) не входят в набор для моделей: передайте папку с alerts_*.parquet
через --alerts. Время в наборе в UTC (проверка — профиль по сдвигу в разделе alerts).
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import _find, build_slices
from forecasting import H, PREDS, RESULTS, Data, load_preds, save_preds

MODELS = {"hyb_blend": "гибрид", "lgbm": "LightGBM", "lgbm_final": "LightGBM финальный", "norm_level": "эталон"}
PLANNED = re.compile(r"planned|weekend|scheduled|reroute", re.I)


def load(data: Data, split: str):
    out = {}
    all_o = data.origins(split)
    for m in MODELS:
        f = PREDS / f"{m}__{split}.npz"
        if f.exists():
            pr = load_preds(f, data)
            if len(pr["origins"]) == len(all_o):
                out[m] = pr["yhat"]
    return all_o, out


def wape(ae, y):
    return float(ae.sum() / max(y.sum(), 1e-9))


# ---------------------------------------------------------------------------
# Оповещения MTA
# ---------------------------------------------------------------------------
def load_alerts(data: Data, folder: str | None, tz: str):
    """(часы, станции) число оперативных оповещений по линиям станции, опубликованных в этот час; и сводка."""
    if not folder:
        return None, None
    files = sorted(Path(folder).glob("**/alerts_*.parquet")) or sorted(Path(folder).glob("**/*.parquet"))
    if not files:
        print(f"Оповещения не найдены в {folder}")
        return None, None
    al = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    date_col = "date" if "date" in al else next(c for c in al if "date" in c)
    ts = pd.to_datetime(al[date_col], utc=(tz == "utc"))
    if tz == "utc":
        ts = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
    al["hour"] = ts.dt.floor("h")
    kind_col = next((c for c in ("status_label", "status", "alert_type") if c in al), None)
    text = al[kind_col].fillna("").astype(str) if kind_col else pd.Series("", index=al.index)
    hdr = al["header"].fillna("").astype(str) if "header" in al else pd.Series("", index=al.index)
    al["planned"] = text.str.contains(PLANNED) | hdr.str.contains(PLANNED)
    if "update_number" in al:                                          # только первая публикация оповещения
        al = al[pd.to_numeric(al["update_number"], errors="coerce").fillna(0) == 0]
    lines = data.static["lines"].fillna("").str.split(",")
    known = sorted({x.strip() for ls in lines for x in ls if x.strip()})
    aff_col = next((c for c in ("affected", "affected_routes", "routes") if c in al), None)
    aff = al[aff_col].fillna("").astype(str) if aff_col else hdr
    tok = aff.str.findall(r"\b[A-Z0-9]{1,2}\b")
    op = al[~al["planned"]].assign(routes=tok[~al["planned"]])
    pos = data.pos.reindex(op["hour"]).to_numpy()
    by_line = {ln: np.zeros(len(data.index), "float32") for ln in known}
    for p, rs in zip(pos, op["routes"]):
        if np.isnan(p):
            continue
        for r in set(rs):
            if r in by_line:
                by_line[r][int(p)] += 1
    A = np.zeros(data.y.shape, "float32")
    for j, ls in enumerate(lines):
        for ln in ls:
            ln = ln.strip()
            if ln in by_line:
                A[:, j] += by_line[ln]
    info = {"оповещений всего": len(al), "оперативных (не плановых)": int((~al["planned"]).sum()),
            "колонка типа": kind_col, "колонка линий": aff_col, "часовой пояс": tz}
    if kind_col:
        print(al[kind_col].value_counts().head(15).to_string())
    return A, info


# ---------------------------------------------------------------------------
def sources(data, o, y, m, preds, sl, alert_mask):
    cats = {
        "всё": np.ones_like(m),
        "ночь 0–5": sl["часы: ночь 0–5"],
        "вечер 20–23": sl["часы: вечер 20–23"],
        "выходной обычный": sl["день: выходной обычный"],
        "праздник, до/после праздника": sl["день: праздник"] | sl["день: до/после праздника"],
        "городское событие": sl["день: городское событие"],
        "событие у станции (по расписанию)": sl.get("событие у станции: состоялось", False) | sl.get("событие у станции: перенесено/отменено", False),
        "день с осадками ≥ 15 мм": sl["погода: день с осадками ≥ 15 мм"],
        "станция была закрыта вчера": sl.get("станция была закрыта вчера", np.zeros_like(m)),
        "обычный будний день 6–19": sl["день: будний обычный"] & ~sl["часы: ночь 0–5"] & ~sl["часы: вечер 20–23"],
    }
    if alert_mask is not None:
        cats["сбой на линии станции (оповещение за 2 ч)"] = alert_mask
    rows = []
    ytot = np.where(m, y, 0).sum()
    for k, c in cats.items():
        c = np.broadcast_to(c, m.shape) & m
        r = {"категория": k, "доля потока": np.where(c, y, 0).sum() / ytot}
        for name, p in preds.items():
            ae = np.where(c, np.abs(p - y), 0)
            aet = np.where(m, np.abs(p - y), 0).sum()
            r[f"{name}: доля ошибки"] = ae.sum() / aet
            r[f"{name}: WAPE"] = ae.sum() / max(np.where(c, y, 0).sum(), 1e-9)
            r[f"{name}: смещение"] = np.where(c, p - y, 0).sum() / max(np.where(c, y, 0).sum(), 1e-9)
        rows.append(r)
    return pd.DataFrame(rows)


def days(data, o, y, m, preds, A):
    """По дню прогнозируемого часа; ошибки станция × час, все горизонты."""
    tidx = data.target_index(o)
    day = data.index.normalize().values[tidx]
    cov = data.cov
    out = pd.DataFrame(index=pd.DatetimeIndex(np.unique(day), name="день"))
    ys = pd.Series(np.where(m, y, 0).sum(-1).ravel()).groupby(day.ravel()).sum()
    for name, p in preds.items():
        ae = pd.Series(np.where(m, np.abs(p - y), 0).sum(-1).ravel()).groupby(day.ravel()).sum()
        sp = pd.Series(np.where(m, p - y, 0).sum(-1).ravel()).groupby(day.ravel()).sum()
        out[f"{name}: WAPE"] = (ae / ys).values
        out[f"{name}: смещение"] = (sp / ys).values
    d0 = data.index.normalize()
    daily = lambda col, f="max": cov[col].groupby(d0).agg(f).reindex(out.index).values
    out["день недели"] = out.index.dayofweek
    out["праздник"] = daily("is_holiday")
    out["до/после праздника"] = np.maximum(daily("is_day_before_holiday"), daily("is_day_after_holiday"))
    out["городское событие"] = daily("is_city_event")
    out["осадки, мм"] = daily("precipitation", "sum")
    out["снег, см"] = daily("snowfall", "sum")
    out["входов, тыс."] = ys.values / 1e3 / H
    if A is not None:
        out["оперативных оповещений (станция × час)"] = pd.Series(A.sum(1)).groupby(d0).sum().reindex(out.index).values
    return out.reset_index()


def stations(data, o, y, m, preds):
    tr = data.index <= pd.Timestamp("2026-03-31 23:00")
    tidx = data.target_index(o)
    # сдвиг уровня: среднее за выборку / среднее за последние 8 недель train в те же часы суток
    last = tr & (data.index > pd.Timestamp("2026-02-03"))
    lvl_tr = np.nanmean(np.where(data.open[last], data.y[last], np.nan), 0)
    sel = np.zeros(len(data.index), bool)
    sel[np.unique(tidx)] = True
    lvl = np.nanmean(np.where(data.open[sel], data.y[sel], np.nan), 0)
    s = data.static
    out = pd.DataFrame({"станция": s["station_complex"].values, "район": s["borough"].values, "линии": s["lines"].values,
                        "площадка": s["venue"].fillna("").values, "входов в будний день": s["scale_weekday"].round().values,
                        "уровень к концу train": lvl / np.maximum(lvl_tr, 1e-9)}, index=data.stations)
    ysum = np.where(m, y, 0).sum((0, 1))
    for name, p in preds.items():
        out[f"{name}: WAPE"] = np.where(m, np.abs(p - y), 0).sum((0, 1)) / np.maximum(ysum, 1e-9)
        out[f"{name}: смещение"] = np.where(m, p - y, 0).sum((0, 1)) / np.maximum(ysum, 1e-9)
        out[f"{name}: доля ошибки"] = np.where(m, np.abs(p - y), 0).sum((0, 1)) / np.where(m, np.abs(p - y), 0).sum()
    return out.reset_index(names="id")


def events(data, o, y, m, preds):
    path = _find("events.csv", data)
    if path is None:
        return pd.DataFrame()
    ev = pd.read_csv(path, parse_dates=["start"])
    from evaluate import EVENT_WINDOW, KIND_VENUE
    tidx = data.target_index(o)
    t0, t1 = tidx.min(), tidx.max()
    venue = data.static["venue"].fillna("").values
    lo, hi = EVENT_WINDOW
    rows = []
    for _, e in ev.iterrows():
        cols = np.where(venue == KIND_VENUE.get(e["kind"], "?"))[0]
        st = e["start"].floor("h")
        if not len(cols) or st not in data.pos.index:
            continue
        hrs = data.pos.reindex(pd.date_range(st + pd.Timedelta(hours=lo), st + pd.Timedelta(hours=hi), freq="h")).dropna().astype(int).values
        if hrs.min() < t0 or hrs.max() > t1:
            continue
        sel = np.isin(tidx, hrs)                                   # все (точка, горизонт), попадающие в окно
        r = {"событие": e["kind"], "площадка": KIND_VENUE.get(e["kind"]), "начало": st, "статус": e.get("status", "final"),
             "станций": len(cols)}
        yy = np.where(m[:, :, cols], y[:, :, cols], 0)[sel].sum() / max(sel.sum(), 1)
        r["факт, входов за окно"] = float(yy * len(hrs))
        for name, p in preds.items():
            pp = np.where(m[:, :, cols], p[:, :, cols], 0)[sel].sum() / max(sel.sum(), 1)
            r[f"{name}: прогноз / факт"] = float(pp / max(yy, 1e-9))
        rows.append(r)
    return pd.DataFrame(rows)


def alerts_section(data, o, y, m, preds, A):
    """Часы со сбоем на линиях станции (оповещение в течение 2 ч до часа); прогноз на 1 час."""
    tidx = data.target_index(o)
    t1 = tidx[:, 0]
    rows = []
    Aw = A + np.vstack([np.zeros((1, A.shape[1]), "float32"), A[:-1]])           # оповещение в этот или прошлый час
    am = Aw[t1] > 0
    y1, m1 = y[:, 0], m[:, 0]
    for name, p in preds.items():
        p1 = p[:, 0]
        ae = np.abs(p1 - y1)
        thr = np.quantile(ae[m1], 0.99)
        big = (ae >= thr) & m1
        over = big & (p1 > y1)
        rows.append({"модель": name,
                     "доля станция-часов со сбоем": am[m1].mean(),
                     "WAPE h1 со сбоем": wape(np.where(am & m1, ae, 0), np.where(am & m1, y1, 0)),
                     "WAPE h1 без сбоя": wape(np.where(~am & m1, ae, 0), np.where(~am & m1, y1, 0)),
                     "смещение h1 со сбоем": np.where(am & m1, p1 - y1, 0).sum() / np.where(am & m1, y1, 0).sum(),
                     "доля сбоев среди 1% крупнейших ошибок": am[big].mean(),
                     "доля сбоев среди крупных завышений": am[over].mean()})
    # профиль по сдвигу: связь городского остатка h1 с числом оповещений через k часов (проверка часового пояса)
    res = np.where(m1, y1 - preds.get("hyb_blend", next(iter(preds.values())))[:, 0], 0).sum(1)
    cnt = A.sum(1)
    prof = {}
    for k in range(-6, 7):
        c = cnt[np.clip(t1 + k, 0, len(cnt) - 1)]
        prof[f"сдвиг {k:+d} ч"] = float(np.corrcoef(res, c)[0, 1]) if c.std() > 0 else np.nan
    return pd.DataFrame(rows), pd.Series(prof, name="корреляция остатка h1 и числа оповещений")


def event_blend(data, split, o, y, m, preds, sl):
    """hyb_ev: гибрид, но в часы событий у площадок (по расписанию) — LightGBM финальный."""
    if "hyb_blend" not in preds or "lgbm_final" not in preds:
        return None
    ev = sl.get("событие у станции: состоялось", False) | sl.get("событие у станции: перенесено/отменено", False)
    ev = np.broadcast_to(ev, m.shape)
    p = np.where(ev, preds["lgbm_final"], preds["hyb_blend"]).astype("float32")
    save_preds("hyb_ev", split, data, o, p)
    rows = []
    for name, q in [("hyb_blend", preds["hyb_blend"]), ("lgbm_final", preds["lgbm_final"]), ("hyb_ev", p)]:
        rows.append({"модель": name, "WAPE": wape(np.where(m, np.abs(q - y), 0), np.where(m, y, 0)),
                     "WAPE события": wape(np.where(m & ev, np.abs(q - y), 0), np.where(m & ev, y, 0))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--alerts", default=None, help="папка с alerts_*.parquet")
    ap.add_argument("--alerts-tz", default="utc", choices=["utc", "local"])
    a = ap.parse_args()
    data = Data()
    o, preds = load(data, a.split)
    print("Модели:", sorted(preds))
    y, m = data.targets(o)
    tidx = data.target_index(o)
    sl = build_slices(data, tidx)
    A, info = load_alerts(data, a.alerts, a.alerts_tz)
    am = None
    if A is not None:
        Aw = A + np.vstack([np.zeros((1, A.shape[1]), "float32"), A[:-1]])
        am = Aw[tidx] > 0
        print("Оповещения:", info)
    RESULTS.mkdir(parents=True, exist_ok=True)
    parts = {"sources": sources(data, o, y, m, preds, sl, am), "days": days(data, o, y, m, preds, A),
             "stations": stations(data, o, y, m, preds), "events": events(data, o, y, m, preds)}
    if A is not None:
        parts["alerts"], prof = alerts_section(data, o, y, m, preds, A)
        prof.to_csv(RESULTS / f"errors_{a.split}_alerts_lag.csv")
        print("\nПрофиль по сдвигу (пик должен быть около 0, со знаком минус — сбой снижает поток):\n" + prof.round(3).to_string())
    eb = event_blend(data, a.split, o, y, m, preds, sl)
    if eb is not None:
        parts["ev_blend"] = eb
    pd.set_option("display.width", 220, "display.max_columns", 30)
    for k, t in parts.items():
        t.to_csv(RESULTS / f"errors_{a.split}_{k}.csv", index=False)
    print("\n=== Откуда берётся ошибка ===\n" + parts["sources"].round(3).to_string(index=False))
    d = parts["days"]
    if "hyb_blend: WAPE" in d:
        print("\n=== Худшие дни (гибрид) ===\n" + d.sort_values("hyb_blend: WAPE", ascending=False).head(12).round(3).to_string(index=False))
    s = parts["stations"]
    if "hyb_blend: доля ошибки" in s:
        print("\n=== Станции с наибольшей долей ошибки ===\n" + s.sort_values("hyb_blend: доля ошибки", ascending=False).head(12).round(3).to_string(index=False))
    if len(parts["events"]):
        print("\n=== События: прогноз / факт (медиана по статусу) ===\n" +
              parts["events"].groupby("статус")[[c for c in parts["events"] if "прогноз / факт" in c]].median().round(2).to_string())
    for k in ("alerts", "ev_blend"):
        if k in parts:
            print(f"\n=== {k} ===\n" + parts[k].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
