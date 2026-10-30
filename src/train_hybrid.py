"""
Гибридная модель: объединяет прогнозы LightGBM, графовой сети и Chronos-2 с учётом того, как они ошибались
в последние часы — на самой станции, у её соседей по графу и по городу в целом.

  python src/train_hybrid.py                                # все варианты, val и test
  python src/train_hybrid.py --variants full nonbr --splits val
  python src/train_hybrid.py --leak-test

Зачем (гипотеза H4). Базовые модели сильны в разном: LightGBM — на праздниках и событиях, графовая сеть —
на коротком горизонте за счёт соседей, Chronos-2 — на h1 без всякого обучения. Мета-модель решает, кому
и насколько верить в зависимости от горизонта, часа, типа дня и недавних ошибок: если все соседи станции
последние два часа идут выше прогноза (сбой на линии, событие), поправка переносится на станцию.
Это и есть схема «уровень города + поправка станции с учётом соседей».

Обучение без утечки (config.PERIODS):
  базовые модели учатся на pre_stack (до 30.09.2025) и прогнозируют stack (10.2025–03.2026);
  на этих прогнозах учится мета-модель для val. Для test она учится на stack + val —
  так же, как базовые модели для теста переобучаются на train + val.
  Ранняя остановка и поправка смещения — последние ES_WEEKS недель обучающего периода.
Нужны results/preds/<база>__{stack,val,test}.npz.

Варианты (сохраняются как hyb_<вариант>__<выборка>.npz):
  blend       линейные веса трёх прогнозов по горизонтам — простейший гибрид
  full        мета-LightGBM: прогнозы + контекст + недавние ошибки станции, соседей и города
  nonbr       full без ошибок соседей                       (вклад пространства)
  no_resid    full без недавних ошибок вообще               (только «кому верить»)
  no_chronos  full без Chronos-2
  no_gnn      full без графовой сети
  lgbm_only   только LightGBM + его недавние ошибки          (выигрыш от объединения или от поправки?)

--on-blend: деревья стартуют не с нуля, а с линейного смешивания тех же моделей и учат только поправку к нему.
Первый запуск (с нуля) показал, что мета-LightGBM хуже простого смешивания: при ранней остановке деревьям
не хватает итераций даже на то, чтобы точно повторить сам прогноз базовой модели (lgbm_only хуже lgbm_final на 4%).
  python src/train_hybrid.py --on-blend --prefix hybr --variants full nonbr no_resid no_chronos no_gnn lgbm_only
"""

import argparse
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from baselines import EPS
from config import PERIODS
from features import STATIC_COLS, FeatureBuilder
from forecasting import H, PREDS, RESULTS, Data, Timer, load_preds, log_run, save_preds

BASES = {"lgbm": "lgbm_final", "gnn": "gnn_od_in", "chronos": "chronos2"}
LAG = 2                    # ошибки считаются по прогнозу из точки T−2 на часы T−2 и T−1 (в точке T они уже известны)
ES_WEEKS = 6
CTX_COLS = ["hour", "dow", "is_holiday", "is_day_before_holiday", "is_day_after_holiday", "is_school_day",
            "xmas_week", "is_city_event"]
VARIANTS = ["blend", "full", "nonbr", "no_resid", "no_chronos", "no_gnn", "lgbm_only"]


def nanmean(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


class Base:
    """Прогнозы одной базовой модели на одной выборке с доступом по номеру часа точки прогноза."""

    def __init__(self, data: Data, model: str, split: str):
        pr = load_preds(PREDS / f"{model}__{split}.npz", data)
        self.yhat = pr["yhat"]
        self.row = pd.Series(np.arange(len(pr["origins"])), index=pr["origins"])
        self.origins = pr["origins"]

    def get(self, origins):
        """(O, 24, N); NaN, если прогноза из этой точки нет."""
        idx = self.row.reindex(origins).to_numpy()
        out = np.full((len(origins),) + self.yhat.shape[1:], np.nan, "float32")
        ok = ~np.isnan(idx)
        out[ok] = self.yhat[idx[ok].astype(int)]
        return out


def load_bases(data: Data, split: str, names=BASES):
    return {k: Base(data, m, split) for k, m in names.items()}


class MetaFeatures:
    def __init__(self, data: Data, fb: FeatureBuilder, graphs):
        self.d, self.fb = data, fb
        self.W = {g: fb.graph(g) for g in graphs}
        self.cal = data.cov[CTX_COLS].to_numpy("float32")
        self.log_scale = fb.static[:, STATIC_COLS.index("log_scale")]

    def build(self, bases: dict, origins: np.ndarray):
        """Строки (точка, горизонт, станция). Возвращает (X, цель-лог, вес, норма, маска для обучения)."""
        d, fb = self.d, self.fb
        O, N = len(origins), len(d.stations)
        t = d.target_index(origins)
        nm = fb.norm[t]
        den = np.nan_to_num(nm) + EPS
        full = lambda a: np.broadcast_to(a, (O, H, N)).reshape(-1)
        f = {}
        # прогнозы базовых моделей в тех же единицах, что и цель: лог отклонения от нормы
        for k, b in bases.items():
            f[f"l_{k}"] = np.log((np.clip(b.get(origins), 0, None) + EPS) / den).reshape(-1)
        # недавние ошибки: факт часов T−2, T−1 минус прогноз, сделанный в T−2
        prev = origins[:, None] - LAG + np.arange(LAG)[None, :]
        r_prev = fb.r[prev]
        den_prev = np.nan_to_num(fb.norm[prev]) + EPS
        for k, b in bases.items():
            p = b.get(origins - LAG)[:, :LAG]
            e = nanmean(r_prev - np.log((np.clip(p, 0, None) + EPS) / den_prev), 1)          # (O, N)
            f[f"e_{k}"] = full(e[:, None, :])
            f[f"e_{k}_city"] = full(nanmean(e, 1)[:, None, None])
            for g, W in self.W.items():
                f[f"e_{k}_nbr_{g}"] = full(fb._nbr(W, e)[:, None, :])
        # контекст
        for w in (1, 3, 24):
            f[f"r{w}"] = full(fb.window_ratio(origins, w)[:, None, :])
        f["city_r1"] = full(fb.r_city[origins - 1][:, None, None])
        f["horizon"] = full(np.arange(1, H + 1, dtype="float32")[None, :, None])
        for j, c in enumerate(CTX_COLS):
            f[c] = full(self.cal[t, j][:, :, None])
        f["ev_active"] = fb.ev_active[t].reshape(-1)
        f["log_norm"] = np.log1p(np.nan_to_num(nm)).reshape(-1)
        f["log_scale"] = full(self.log_scale[None, None, :])
        X = pd.DataFrame({k: np.ascontiguousarray(v, dtype="float32") for k, v in f.items()})
        y = fb.r[t].reshape(-1)
        ok = ~np.isnan(y) & np.isfinite(f["l_lgbm"])
        return X, y, den.reshape(-1), nm.reshape(-1), ok


def columns(cols, variant):
    drop = {"full": [], "nonbr": ["_nbr_"], "no_resid": ["e_"], "no_chronos": ["chronos"], "no_gnn": ["gnn"],
            "lgbm_only": ["chronos", "gnn"]}[variant]
    return [c for c in cols if not any(c.startswith(x) if x == "e_" else x in c for x in drop)]


# ---------------------------------------------------------------------------
def collect(meta: MetaFeatures, parts, max_rows, rng, chunk=48):
    """parts — список (базы, точки прогноза). Случайная доля строк, чтобы уложиться в max_rows."""
    total = sum(len(o) for _, o in parts) * H * len(meta.d.stations)
    frac = min(1.0, max_rows / max(total, 1))
    Xs, ys, ws = [], [], []
    for bases, origins in parts:
        for i in range(0, len(origins), chunk):
            X, y, w, _, ok = meta.build(bases, origins[i:i + chunk])
            keep = ok & (rng.random(len(ok)) < frac)
            Xs.append(X[keep])
            ys.append(y[keep])
            ws.append(w[keep])
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(ws)


def bias_factor(pred_log, y_log, w, horizon):
    """Множитель по горизонтам: Σ факта / Σ прогноза во входах (как у базовых моделей)."""
    yc = w * np.exp(y_log) - EPS
    pc = np.clip(w * np.exp(pred_log) - EPS, 0, None)
    f = np.ones(H)
    for k in range(1, H + 1):
        m = horizon == k
        f[k - 1] = np.clip(yc[m].sum() / max(pc[m].sum(), 1e-9), 0.9, 1.1)
    return f


def fit_blend(Xtr, ytr, wtr, Xes, yes, wes, keys):
    """Взвешенный МНК по каждому горизонту: y ≈ Σ β_k · l_k + β_0."""
    cols = [f"l_{k}" for k in keys]
    beta = np.zeros((H, len(cols) + 1))
    htr = Xtr["horizon"].to_numpy().astype(int)
    for h in range(1, H + 1):
        m = htr == h
        A = np.column_stack([Xtr.loc[m, cols].to_numpy("float64"), np.ones(m.sum())])
        sw = wtr[m]
        beta[h - 1] = np.linalg.solve(A.T @ (A * sw[:, None]) + 1e-6 * np.eye(A.shape[1]), A.T @ (sw * ytr[m]))
    pred = lambda X: (np.column_stack([X[cols].to_numpy("float64"), np.ones(len(X))])
                      * beta[X["horizon"].to_numpy().astype(int) - 1]).sum(1)
    factor = bias_factor(pred(Xes), yes, wes, Xes["horizon"].to_numpy().astype(int))
    return {"kind": "blend", "beta": beta, "cols": cols, "predict": pred, "factor": factor}


def fit_lgbm(Xtr, ytr, wtr, Xes, yes, wes, cols, a):
    """Мета-LightGBM. С --on-blend деревья учат поправку к линейному смешиванию тех же моделей (init_score),
    без него — прогноз с нуля."""
    keys = [c[2:] for c in cols if c.startswith("l_")]
    base = fit_blend(Xtr, ytr, wtr, Xes, yes, wes, keys)["predict"] if a.on_blend else (lambda X: np.zeros(len(X)))
    params = {"objective": "regression", "learning_rate": a.lr, "num_leaves": a.leaves, "min_data_in_leaf": a.min_leaf,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "seed": a.seed, "verbose": -1, "num_threads": 0}
    dtr = lgb.Dataset(Xtr[cols], ytr, weight=wtr, init_score=base(Xtr))
    des = lgb.Dataset(Xes[cols], yes, weight=wes, init_score=base(Xes), reference=dtr)
    model = lgb.train(params, dtr, a.rounds, valid_sets=[des], valid_names=["es"],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
    pred = lambda X: base(X) + model.predict(X[cols], num_iteration=model.best_iteration, num_threads=0)
    factor = bias_factor(pred(Xes), yes, wes, Xes["horizon"].to_numpy().astype(int))
    return {"kind": "lgbm", "model": model, "cols": cols, "predict": pred, "factor": factor}


def es_split(data: Data, origins, end):
    """Точки, у которых весь прогноз до окна ранней остановки, и точки внутри окна."""
    es_start = int(data.pos[end.normalize() - pd.Timedelta(weeks=ES_WEEKS) + pd.Timedelta(days=1)])
    return origins[origins + H - 1 < es_start], origins[origins >= es_start]


def fit_all(meta, data, split, variants, a, bases_by_split):
    """Мета-модели для прогноза на split: для val — обучение на stack, для test — на stack + val."""
    periods = ["stack"] if split == "val" else ["stack", "val"]
    parts_tr, parts_es = [], []
    for i, p in enumerate(periods):
        b = bases_by_split[p]
        o = b["lgbm"].origins
        for k in b:
            o = np.intersect1d(o, b[k].origins)
        if i == len(periods) - 1:
            tr, es = es_split(data, o, PERIODS[p][1])
            parts_tr.append((b, tr))
            parts_es.append((b, es))
        else:
            parts_tr.append((b, o))
    rng = np.random.default_rng(a.seed)
    with Timer() as t:
        Xtr, ytr, wtr = collect(meta, parts_tr, a.max_rows, rng)
        Xes, yes, wes = collect(meta, parts_es, a.max_rows // 4, rng)
    print(f"[мета для {split}: {' + '.join(periods)}] строк: обучение {len(ytr):,}, ранняя остановка {len(yes):,}; "
          f"признаков {Xtr.shape[1]}; сборка {t.seconds / 60:.1f} мин", flush=True)
    keys = [k for k in BASES if f"l_{k}" in Xtr]
    out = {}
    for v in variants:
        with Timer() as t:
            if v == "blend":
                m = fit_blend(Xtr, ytr, wtr, Xes, yes, wes, keys)
                w = ", ".join(f"{k} {m['beta'][:, j].mean():.2f}" for j, k in enumerate(keys))
                info = f"средние веса: {w}"
            else:
                m = fit_lgbm(Xtr, ytr, wtr, Xes, yes, wes, columns(list(Xtr.columns), v), a)
                info = f"деревьев {m['model'].best_iteration}"
                fi = pd.DataFrame({"feature": m["cols"], "gain": m["model"].feature_importance("gain")})
                fi["share"] = fi["gain"] / fi["gain"].sum()
                tag = "" if split == "val" else "_trainval"
                fi.sort_values("share", ascending=False).to_csv(RESULTS / f"fi_{a.prefix}_{v}{tag}.csv", index=False)
        m["fit_s"] = t.seconds
        out[v] = m
        print(f"  {v}: {t.seconds / 60:.1f} мин, {info}; поправка h1 ×{m['factor'][0]:.3f}, h24 ×{m['factor'][23]:.3f}",
              flush=True)
    return out


def predict_split(meta, data, bases, models, split, chunk=48):
    origins = data.origins(split)
    out = {v: np.empty((len(origins), H, len(data.stations)), "float32") for v in models}
    with Timer() as t:
        for i in range(0, len(origins), chunk):
            o = origins[i:i + chunk]
            X, _, _, nm, _ = meta.build(bases, o)
            fallback = np.nan_to_num(bases["lgbm"].get(o)).reshape(-1)
            for v, m in models.items():
                p = m["predict"](X).reshape(len(o), H, -1) * 1.0
                cnt = (np.nan_to_num(nm).reshape(len(o), H, -1) + EPS) * np.exp(p) * m["factor"][None, :, None] - EPS
                cnt = np.where(np.isnan(nm).reshape(cnt.shape), fallback.reshape(cnt.shape), cnt)
                out[v][i:i + len(o)] = np.clip(np.nan_to_num(cnt), 0, None)
    return origins, out, t.seconds


def leak_test(data: Data, a) -> bool:
    """Признаки мета-модели для точки T не меняются, если испортить все входы начиная с T."""
    bases = load_bases(data, "val")
    T = data.origins("val")[[100, 1500]]
    X0 = MetaFeatures(data, FeatureBuilder(data), a.graphs).build(bases, T)[0]
    y_bak = data.y
    data.y = data.y.copy()
    rng = np.random.default_rng(0)
    data.y[T.min():] = rng.integers(0, 5000, data.y[T.min():].shape)
    X1 = MetaFeatures(data, FeatureBuilder(data), a.graphs).build(bases, T[:1])[0]
    data.y = y_bak
    n = len(X1)
    return bool(np.allclose(X0.iloc[:n].to_numpy(), X1.to_numpy(), equal_nan=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    ap.add_argument("--graphs", nargs="*", default=["od_in", "line"], choices=["line", "dist", "od_in"])
    ap.add_argument("--max-rows", type=int, default=4_000_000, help="строк для обучения мета-модели")
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--min-leaf", type=int, default=1000)
    ap.add_argument("--on-blend", action="store_true", help="мета-модель учит поправку к линейному смешиванию")
    ap.add_argument("--prefix", default="hyb", help="имена прогнозов: <prefix>_<вариант>")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--leak-test", action="store_true")
    a = ap.parse_args()

    data = Data()
    if a.leak_test:
        print("Тест утечки:", "пройден" if leak_test(data, a) else "УТЕЧКА")
        return
    fb = FeatureBuilder(data)
    meta = MetaFeatures(data, fb, a.graphs)
    RESULTS.mkdir(parents=True, exist_ok=True)
    need = {"val": ["stack", "val"], "test": ["stack", "val", "test"]}
    bases_by_split = {s: load_bases(data, s) for s in sorted({p for s in a.splits for p in need[s]})}
    for split in a.splits:
        models = fit_all(meta, data, split, a.variants, a, bases_by_split)
        origins, preds, pred_s = predict_split(meta, data, bases_by_split[split], models, split)
        y, m = data.targets(origins)
        ys = np.where(m, y, 0).sum()
        ref = np.where(m, np.abs(bases_by_split[split]["lgbm"].get(origins) - y), 0).sum() / ys
        print(f"{split}: WAPE {BASES['lgbm']} {ref:.4f}", flush=True)
        for v, p in preds.items():
            name = f"{a.prefix}_{v}"
            save_preds(name, split, data, origins, p)
            params = {k: val for k, val in vars(a).items() if k not in ("variants", "splits", "leak_test", "prefix")}
            params.update(bases=BASES, trained_on="stack" if split == "val" else "stack+val",
                          bias_factor_mean=round(float(models[v]["factor"].mean()), 4))
            log_run(name, split, params, models[v]["fit_s"], pred_s / len(preds), len(origins), None)
            wape = np.where(m, np.abs(p - y), 0).sum() / ys
            print(f"  {name}: WAPE {wape:.4f} ({100 * (wape / ref - 1):+.1f}% к {BASES['lgbm']})", flush=True)


if __name__ == "__main__":
    main()
