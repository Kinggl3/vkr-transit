"""
Дополнительные разрезы оценки для главы 3 (поверх evaluate.py, по тем же файлам прогнозов).

  python src/report.py --split test --pairs gnn_od_in:gnn_none lgbm_all:lgbm

Что считает:
  slices       — постановки, близкие к литературе: обычные будни, 06–24 ч, горизонты 1–3 ч (как HW-DMD)
                 и 05–23 ч, горизонт 1 ч (как Conv-GCN, Пекин)
  aggregation  — WAPE на разных уровнях: станция × час, станция × сутки, район × час, город × час, город × сутки
                 (сутки — прогноз из точки 00:00 на 24 часа)
  windows      — разброс WAPE по всем окнам из 5 и 10 обычных будних дней (как короткие тесты в статьях)
  pairs        — разница WAPE пар моделей и 95% интервал (бутстреп по дням)
Результаты: results/report_<split>_<раздел>.csv.
"""

import argparse

import numpy as np
import pandas as pd

from evaluate import bootstrap_diff, day_sums
from forecasting import PREDS, RESULTS, STRIDE_COMMON, Data, load_preds

SPECIAL = ["is_holiday", "is_day_before_holiday", "is_day_after_holiday", "xmas_week", "is_city_event"]


def wape(y, p, m):
    s = np.where(m, y, 0).sum()
    return float(np.where(m, np.abs(p - y), 0).sum() / s) if s > 0 else np.nan


def ordinary_weekday(data: Data) -> np.ndarray:
    """(часы,) True — будний день без праздников, особых и соседних с праздником дней."""
    c = data.cov
    special = np.zeros(len(c), bool)
    for col in SPECIAL:
        special |= c[col].to_numpy() > 0
    day = pd.Series(special, index=data.index.normalize()).groupby(level=0).transform("max").to_numpy()
    return (c["dow"].to_numpy() < 5) & ~day


def load_full(data: Data, split: str, models=None):
    """Модели с прогнозом на все точки выборки: имя → (O, 24, N)."""
    all_o = data.origins(split)
    out = {}
    for f in sorted(PREDS.glob(f"*__{split}.npz")):
        name = f.name.split("__")[0]
        if models and name not in models:
            continue
        pr = load_preds(f, data)
        if len(pr["origins"]) == len(all_o) and (pr["origins"] == all_o).all():
            out[name] = pr["yhat"]
    return out


def slices(data: Data, split: str, preds: dict) -> pd.DataFrame:
    o = data.origins(split)
    tidx = data.target_index(o)
    y, m = data.targets(o)
    hour = data.index.hour.to_numpy()[tidx]
    ok_day = ordinary_weekday(data)[tidx]
    rows = []
    for name, p in preds.items():
        r = {"model": name}
        for h in (1, 2, 3):
            sel = (ok_day & (hour >= 6))[:, h - 1, None] & m[:, h - 1]
            r[f"A: будни 06–24, h{h}"] = wape(y[:, h - 1], p[:, h - 1], sel)
        sel = (ok_day & (hour >= 5) & (hour <= 22))[:, 0, None] & m[:, 0]
        r["B: будни 05–23, h1"] = wape(y[:, 0], p[:, 0], sel)
        rows.append(r)
    return pd.DataFrame(rows).set_index("model")


def aggregation(data: Data, split: str, preds: dict) -> pd.DataFrame:
    o = data.origins(split)
    y, m = data.targets(o)
    midnight = data.index.hour.to_numpy()[o] == 0
    boro = data.static["borough"].fillna("?").to_numpy()
    rows = []
    for name, p in preds.items():
        ym, pm = np.where(m, y, 0), np.where(m, p, 0)
        r = {"model": name, "станция × час": wape(y, p, m)}
        yd, pd_ = ym[midnight].sum(1), pm[midnight].sum(1)                       # (дни, N)
        r["станция × сутки"] = float(np.abs(pd_ - yd).sum() / yd.sum())
        yb = np.stack([ym[..., boro == b].sum(-1) for b in np.unique(boro)], -1)
        pb = np.stack([pm[..., boro == b].sum(-1) for b in np.unique(boro)], -1)
        r["район × час"] = float(np.abs(pb - yb).sum() / yb.sum())
        yc, pc = ym.sum(-1), pm.sum(-1)
        r["город × час"] = float(np.abs(pc - yc).sum() / yc.sum())
        r["город × сутки"] = float(np.abs(pd_.sum(1) - yd.sum(1)).sum() / yd.sum())
        rows.append(r)
    return pd.DataFrame(rows).set_index("model")


def windows(data: Data, split: str, preds: dict) -> pd.DataFrame:
    """Срез A (h1) по каждому окну из 5 и 10 подряд идущих обычных будних дней."""
    o = data.origins(split)
    y, m = data.targets(o)
    tidx = data.target_index(o)
    hour = data.index.hour.to_numpy()[tidx[:, 0]]
    days = data.index[tidx[:, 0]].normalize()
    ok = ordinary_weekday(data)[tidx[:, 0]] & (hour >= 6)
    good_days = sorted(set(days[ok]))
    rows = []
    for name, p in preds.items():
        ae = np.where(m[:, 0], np.abs(p[:, 0] - y[:, 0]), 0).sum(1)
        ys = np.where(m[:, 0], y[:, 0], 0).sum(1)
        per_day = pd.DataFrame({"d": days[ok], "ae": ae[ok], "y": ys[ok]}).groupby("d").sum()
        for k in (5, 10):
            w = [per_day["ae"].iloc[i:i + k].sum() / per_day["y"].iloc[i:i + k].sum()
                 for i in range(len(per_day) - k + 1)]
            rows.append({"model": name, "окно, будних дней": k, "окон": len(w), "мин": np.min(w),
                         "медиана": np.median(w), "макс": np.max(w),
                         "весь тест": per_day["ae"].sum() / per_day["y"].sum()})
    out = pd.DataFrame(rows)
    out.attrs["days"] = len(good_days)
    return out


def pairs(data: Data, split: str, pairs_list) -> pd.DataFrame:
    """Разница WAPE (a − b) и интервал; набор точек — все, если у обеих моделей есть все точки, иначе общий подсет."""
    all_o = data.origins(split)
    common = data.origins(split, STRIDE_COMMON)
    rows = []
    for a, b in pairs_list:
        if not all((PREDS / f"{n}__{split}.npz").exists() for n in (a, b)):
            print(f"  пара {a}:{b} пропущена — нет прогноза на {split}")
            continue
        pa, pb = (load_preds(PREDS / f"{n}__{split}.npz", data) for n in (a, b))
        full = all(len(p["origins"]) == len(all_o) for p in (pa, pb))
        o = all_o if full else common
        y, m = data.targets(o)
        get = lambda pr: pr["yhat"][pd.Series(np.arange(len(pr["origins"])), index=pr["origins"]).reindex(o).to_numpy(int)]
        days = data.index[o].normalize()
        ae_a, ys = day_sums(y, get(pa), m, days)
        ae_b, _ = day_sums(y, get(pb), m, days)
        d, lo, hi = bootstrap_diff(ae_a, ae_b, ys)
        rows.append({"a": a, "b": b, "точки": "все" if full else "общий подсет", "wape_a": ae_a.sum() / ys.sum(),
                     "wape_b": ae_b.sum() / ys.sum(), "разница a−b": d, "ДИ низ": lo, "ДИ верх": hi,
                     "отн. изменение, %": 100 * d / (ae_b.sum() / ys.sum()), "значимо": bool(hi < 0 or lo > 0)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--pairs", nargs="*", default=[], help="пары a:b — значимость разницы WAPE")
    a = ap.parse_args()
    data = Data()
    pd.set_option("display.width", 200, "display.max_columns", 20)
    preds = load_full(data, a.split, a.models)
    print(f"Модели с прогнозом на все точки {a.split}: {sorted(preds)}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    parts = {"slices": slices(data, a.split, preds), "aggregation": aggregation(data, a.split, preds)}
    w = windows(data, a.split, preds)
    parts["windows"] = w
    if a.pairs:
        parts["pairs"] = pairs(data, a.split, [tuple(x.split(":")) for x in a.pairs])
    titles = {"slices": "Срезы, близкие к литературе (WAPE)", "aggregation": "WAPE по уровням агрегации",
              "windows": f"Разброс WAPE по коротким окнам (срез A, h1; обычных будних дней в выборке: {w.attrs['days']})",
              "pairs": "Пары моделей: разница WAPE и 95% интервал"}
    for k, t in parts.items():
        t.to_csv(RESULTS / f"report_{a.split}_{k}.csv")
        print(f"\n=== {titles[k]} ===\n" + t.round(4).to_string())


if __name__ == "__main__":
    main()
