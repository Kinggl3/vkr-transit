"""
Оценка прогнозов по протоколу (docs/PROTOCOL.md). Одинаково для всех моделей.

  python src/evaluate.py --split val                  # все модели из results/preds/*__val.npz
  python src/evaluate.py --split test --models lgbm snaive168
  python src/evaluate.py --split val --ref median4w   # с какой моделью сравнивать (по умолчанию лучший бейзлайн)

Результаты:
  results/metrics_<split>.csv   — длинная таблица: model, subset, slice, n, wape, mae, rmse, bias, mase, ...
  results/compare_<split>.csv   — разница WAPE с эталонной моделью и 95% доверительный интервал (бутстреп по дням)
и короткая сводка на экране.

Метрики считаются только по открытым станциям (маска closed_wide). Главная метрика — WAPE.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import RAW
from forecasting import H, PREDS, RESULTS, STRIDE_COMMON, Data, load_preds

BASELINES = ("snaive168", "median4w", "norm_level")
FULL, COMMON = "все точки", "общий подсет"
HORIZON_GROUPS = {"h1": (1, 1), "h2-3": (2, 3), "h4-6": (4, 6), "h7-12": (7, 12), "h13-24": (13, 24)}
HOUR_GROUPS = {"ночь 0–5": (0, 5), "утренний пик 6–9": (6, 9), "день 10–15": (10, 15),
               "вечерний пик 16–19": (16, 19), "вечер 20–23": (20, 23)}
EVENT_WINDOW = (-2, 7)                       # как в build_features.py
KIND_VENUE = {"mlb_yankees": "Yankee Stadium", "mlb_mets": "Citi Field / USTA",
              "usopen_main": "Citi Field / USTA", "usopen_fanweek": "Citi Field / USTA",
              "milb_cyclones": "Maimonides Park", "nba_nets": "Barclays Center",
              "wnba_liberty": "Barclays Center"}


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------
def point_metrics(y, p, m, scale=None) -> dict:
    """y, p, m: (O, 24, N). scale: (N,) знаменатель MASE по станциям."""
    e = np.where(m, p - y, 0.0)
    ys = np.where(m, y, 0.0)
    n = int(m.sum())
    if n == 0:
        return {"n": 0}
    out = {"n": n, "wape": np.abs(e).sum() / ys.sum(), "mae": np.abs(e).sum() / n,
           "rmse": np.sqrt((e ** 2).sum() / n), "bias": e.sum() / ys.sum()}
    if scale is not None:
        cnt = m.sum(axis=(0, 1))
        mae_st = np.abs(e).sum(axis=(0, 1)) / np.maximum(cnt, 1)
        ok = (cnt > 0) & (scale > 0)
        out["mase"] = float(np.mean(mae_st[ok] / scale[ok]))
    return out


def quantile_metrics(y, q, levels, m) -> dict:
    """q: (O, 24, N, Q). WQL — взвешенный квантильный лосс (≈ CRPS), покрытие и ширина 80%-интервала."""
    levels = np.asarray(levels, dtype="float32")
    ys = np.where(m, y, 0.0).sum()
    d = y[..., None] - q
    pin = np.maximum(levels * d, (levels - 1) * d)
    wql = float(np.mean([2 * np.where(m, pin[..., k], 0).sum() / ys for k in range(len(levels))]))
    out = {"wql": wql}
    lo, hi = np.argmin(np.abs(levels - 0.1)), np.argmin(np.abs(levels - 0.9))
    if np.isclose(levels[lo], 0.1) and np.isclose(levels[hi], 0.9):
        inside = (y >= q[..., lo]) & (y <= q[..., hi])
        out["cov80"] = float(inside[m].mean())
        out["width80"] = float(np.where(m, q[..., hi] - q[..., lo], 0).sum() / ys)
    return out


def mase_scale(data: Data) -> np.ndarray:
    """Знаменатель MASE: средняя ошибка «неделю назад» по станции на обучающем периоде (открытые часы)."""
    a, b = 168, data.split_bounds("train")[1] + 1
    y, op = data.y[:b], data.open[:b]
    ok = op[a:] & op[:-a]
    return (np.where(ok, np.abs(y[a:] - y[:-a]), 0).sum(axis=0) / np.maximum(ok.sum(axis=0), 1)).astype("float32")


# ---------------------------------------------------------------------------
# Разрезы
# ---------------------------------------------------------------------------
def _find(name: str, data: Data) -> Path | None:
    for p in (data.lab / name, RAW / name):
        if p.exists():
            return p
    return None


def event_masks(data: Data) -> dict[str, np.ndarray] | None:
    """(часы, станции): станция у площадки в окне события — состоявшегося или перенесённого/отменённого."""
    path = _find("events.csv", data)
    if path is None:
        return None
    ev = pd.read_csv(path, parse_dates=["start"])
    if "status" not in ev:
        ev["status"] = "final"
    venue = data.static["venue"].fillna("")
    out = {"final": np.zeros(data.y.shape, bool), "not_played": np.zeros(data.y.shape, bool)}
    lo, hi = EVENT_WINDOW
    for _, e in ev.iterrows():
        cols = np.where(venue.values == KIND_VENUE.get(e["kind"], "?"))[0]
        if not len(cols):
            continue
        t = e["start"].floor("h")
        rows = data.pos.reindex(pd.date_range(t + pd.Timedelta(hours=lo), t + pd.Timedelta(hours=hi), freq="h"))
        rows = rows.dropna().astype(int).values
        key = "final" if e["status"] in ("final", "suspended") else "not_played"
        out[key][np.ix_(rows, cols)] = True
    return out


def build_slices(data: Data, tidx: np.ndarray) -> dict[str, np.ndarray]:
    """Маски разрезов формы (O, 24, N) или совместимой для broadcasting."""
    O = len(tidx)
    ts = data.index[tidx.ravel()].to_frame(index=False).iloc[:, 0]
    hour = ts.dt.hour.to_numpy().reshape(O, H)
    cov = data.cov
    pick = lambda col: cov[col].to_numpy()[tidx] > 0 if col in cov else None
    s: dict[str, np.ndarray] = {}
    for k, (a, b) in HORIZON_GROUPS.items():
        m = np.zeros((1, H, 1), bool)
        m[0, a - 1:b, 0] = True
        s[f"горизонт {k}"] = m
    for k, (a, b) in HOUR_GROUPS.items():
        s[f"часы: {k}"] = ((hour >= a) & (hour <= b))[..., None]
    hol, bef, aft, xmas, cev = (pick(c) for c in
                                ["is_holiday", "is_day_before_holiday", "is_day_after_holiday", "xmas_week", "is_city_event"])
    wkend = (cov["dow"].to_numpy()[tidx] >= 5)
    special = hol | bef | aft | xmas | cev
    s["день: будний обычный"] = (~wkend & ~special)[..., None]
    s["день: выходной обычный"] = (wkend & ~special)[..., None]
    s["день: праздник"] = hol[..., None]
    s["день: до/после праздника"] = (bef | aft)[..., None]
    s["день: рождественская неделя"] = xmas[..., None]
    s["день: городское событие"] = cev[..., None]
    precip = cov["precipitation"].to_numpy()
    day = data.index.normalize()
    daily = pd.Series(precip).groupby(day).transform("sum").to_numpy()
    s["погода: день с осадками ≥ 15 мм"] = (daily[tidx] >= 15)[..., None]
    s["погода: снег"] = (pd.Series(cov["snowfall"].to_numpy()).groupby(day).transform("sum").to_numpy()[tidx] > 1)[..., None]

    scale = data.static["scale_weekday"].to_numpy()
    qs = np.nanquantile(scale, [0.25, 0.5, 0.75])
    grp = np.digitize(scale, qs)
    for g, nm in enumerate(["малые (нижние 25%)", "средние 25–50%", "крупные 50–75%", "крупнейшие (верхние 25%)"]):
        s[f"станции: {nm}"] = (grp == g)[None, None, :]
    for b in sorted(data.static["borough"].dropna().unique()):
        s[f"район: {b}"] = (data.static["borough"].to_numpy() == b)[None, None, :]

    ev = event_masks(data)
    if ev is not None:
        s["событие у станции: состоялось"] = ev["final"][tidx]
        s["событие у станции: перенесено/отменено"] = ev["not_played"][tidx]

    cpd = data.lab / "closed_prev_day_wide.parquet"
    if cpd.exists():
        w = pd.read_parquet(cpd)
        w.columns = w.columns.astype(str)
        w = w.reindex(index=day.unique(), columns=data.stations).fillna(0).to_numpy(bool)
        day_pos = pd.Series(np.arange(len(day.unique())), index=day.unique())
        s["станция была закрыта вчера"] = w[day_pos[day[tidx.ravel()]].to_numpy().reshape(O, H)]
    return s


# ---------------------------------------------------------------------------
# Сравнение с эталоном: бутстреп по дням
# ---------------------------------------------------------------------------
def day_sums(y, p, m, origin_days):
    """Суммы |ошибки| и факта по дням точки прогноза — для бутстрепа WAPE."""
    ae = np.where(m, np.abs(p - y), 0).sum(axis=(1, 2))
    ys = np.where(m, y, 0).sum(axis=(1, 2))
    g = pd.DataFrame({"d": origin_days, "ae": ae, "y": ys}).groupby("d").sum()
    return g["ae"].to_numpy(), g["y"].to_numpy()


def bootstrap_diff(ae_a, ae_b, ysum, n_boot=1000, seed=0):
    """WAPE(a) − WAPE(b) и 95% интервал; дни ресемплируются блоками (целые сутки)."""
    rng = np.random.default_rng(seed)
    k = len(ysum)
    idx = rng.integers(0, k, size=(n_boot, k))
    d = ae_a[idx].sum(1) / ysum[idx].sum(1) - ae_b[idx].sum(1) / ysum[idx].sum(1)
    return ae_a.sum() / ysum.sum() - ae_b.sum() / ysum.sum(), np.quantile(d, 0.025), np.quantile(d, 0.975)


# ---------------------------------------------------------------------------
def evaluate(split: str, models: list[str] | None = None, ref: str | None = None, data: Data | None = None):
    data = data or Data()
    files = sorted(PREDS.glob(f"*__{split}.npz"))
    if models:
        files = [f for f in files if f.name.split("__")[0] in models]
    if not files:
        raise SystemExit(f"Нет прогнозов в {PREDS} для выборки {split}")
    scale = mase_scale(data)
    all_origins = data.origins(split)
    common = data.origins(split, STRIDE_COMMON)
    y_all, m_all = data.targets(all_origins)
    slices = build_slices(data, data.target_index(all_origins))
    origin_days = data.index[all_origins].normalize()

    pos = pd.Series(np.arange(len(all_origins)), index=all_origins)
    cpos = pos.reindex(common).to_numpy(int)
    days = {FULL: origin_days, COMMON: origin_days[cpos]}
    rows, sums = [], {FULL: {}, COMMON: {}}

    def add(model, subset, y, p, m, idx, pr=None, q_idx=None):
        """Метрики на наборе точек idx (номера в all_origins) — целиком и по разрезам."""
        r = {"model": model, "subset": subset, "slice": "всё", **point_metrics(y, p, m, scale)}
        if pr is not None and q_idx is not None:
            r.update(quantile_metrics(y, pr["quantiles"][q_idx], pr["q_levels"], m))
        rows.append(r)
        for nm, sm in slices.items():
            sm_ = sm[idx] if sm.shape[0] == len(all_origins) else sm
            rows.append({"model": model, "subset": subset, "slice": nm, **point_metrics(y, p, m & sm_, scale)})
        sums[subset][model] = day_sums(y, p, m, days[subset])
        return r

    for f in files:
        model = f.name.split("__")[0]
        pr = load_preds(f, data)
        have = pos.reindex(pr["origins"])
        if have.isna().any():
            raise SystemExit(f"{f.name}: точки прогноза вне выборки {split}")
        sel = have.to_numpy(int)
        p = pr["yhat"]
        msg = []
        if len(sel) == len(all_origins):                       # все точки
            r = add(model, FULL, y_all[sel], p, m_all[sel], sel)
            msg.append(f"все точки {r['wape']:.4f}")
        # общий подсет — одинаковый для всех моделей (и для вероятностных)
        inm = pd.Series(np.arange(len(sel)), index=sel).reindex(cpos)
        if inm.notna().all():
            ii = inm.to_numpy(int)
            q_idx = None
            if "quantiles" in pr:
                qpos = pd.Series(np.arange(len(pr["q_origins"])), index=pr["q_origins"]).reindex(common)
                q_idx = qpos.to_numpy(int) if qpos.notna().all() else None
            r = add(model, COMMON, y_all[cpos], p[ii], m_all[cpos], cpos, pr, q_idx)
            msg.append(f"общий подсет {r['wape']:.4f}")
        else:
            msg.append("общий подсет не покрыт")
        print(f"  {model}: WAPE " + ", ".join(msg), flush=True)

    res = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS / f"metrics_{split}.csv", index=False)

    # Сравнение с эталоном (по умолчанию — лучший из бейзлайнов): бутстреп по дням, на обоих наборах точек
    if ref is None:
        main_ = res[(res.slice == "всё") & (res.subset == COMMON) & res.model.isin(BASELINES)]
        ref = main_.set_index("model")["wape"].idxmin() if len(main_) else None
    comp = []
    for subset in (FULL, COMMON):
        if ref not in sums[subset]:
            continue
        ae_ref, ysum = sums[subset][ref]
        for model, (ae, _) in sums[subset].items():
            d, lo, hi = bootstrap_diff(ae, ae_ref, ysum)
            comp.append({"subset": subset, "model": model, "ref": ref, "wape_diff": d, "ci_low": lo,
                         "ci_high": hi, "significant": bool(hi < 0 or lo > 0)})
    comp = pd.DataFrame(comp)
    comp.to_csv(RESULTS / f"compare_{split}.csv", index=False)
    return res, comp


def summary(res: pd.DataFrame, comp: pd.DataFrame, subset: str = COMMON) -> pd.DataFrame:
    r = res[res.subset == subset]
    out = r[r.slice == "всё"].set_index("model")[["wape", "mae", "rmse", "bias", "mase"]]
    hz = r[r.slice.str.startswith("горизонт")].pivot_table(index="model", columns="slice", values="wape")
    hz.columns = [c.replace("горизонт ", "WAPE ") for c in hz.columns]
    out = out.join(hz[[f"WAPE {k}" for k in HORIZON_GROUPS if f"WAPE {k}" in hz]])
    for c in ["wql", "cov80"]:
        if c in r and r[c].notna().any():
            out[c] = r[r.slice == "всё"].set_index("model")[c]
    c = comp[comp.subset == subset].set_index("model") if len(comp) else pd.DataFrame()
    if len(c):
        out[f"Δ к {c['ref'].iloc[0]} [95% ДИ]"] = [
            f"{c.loc[m, 'wape_diff']:+.4f} [{c.loc[m, 'ci_low']:+.4f}; {c.loc[m, 'ci_high']:+.4f}]" if m in c.index else ""
            for m in out.index]
    return out.sort_values("wape")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--ref")
    a = ap.parse_args()
    res, comp = evaluate(a.split, a.models, a.ref)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    for subset in (FULL, COMMON):
        t = summary(res, comp, subset)
        if len(t):
            print(f"\n=== {subset} ===\n" + t.round(4).to_string())
    print(f"\nПодробно: {RESULTS / f'metrics_{a.split}.csv'}")


if __name__ == "__main__":
    main()
