"""
Внешняя проверка на открытых наборах HZMetro (Ханчжоу, 80 станций, январь 2019) и SHMetro (Шанхай, 288 станций,
июль–сентябрь 2016) из работы PVCGN (Liu et al., IEEE T-ITS 2022, github.com/ivechan/PVCGN).

  python src/external.py --data PVCGN/data --city hangzhou --step 15
  python src/external.py --data PVCGN/data --city hangzhou --step 30 --inflow-only     # постановка HW-DMD
  python src/external.py ... --chronos                                                   # + Chronos-2 (нужен GPU)

Метод обучается заново на чужих данных — модель, обученная на Нью-Йорке, не переносится.
Наборы: входы и выходы каждые 15 мин, 05:30–23:30; разбиение по дням как у авторов (HZ: train 01–18.01, val 19–20.01,
test 21–25.01; SH: train 01.07–31.08, val 01–09.09, test 10–30.09).

Постановки:
  step 15 — как у PVCGN: прогноз на 4 шага (15–60 мин) для входов и выходов, точки прогноза 06:30–22:30,
            MAE / RMSE / MAPE их кодом (нули в факте исключаются), по каждому шагу;
  step 30 — как у HW-DMD (Ханчжоу): только входы, шаг 30 мин, 1–3 шага, WMAPE (= наш WAPE), тест 21–25.01.
Модели:
  ha       историческое среднее того же интервала по дням обучения (два варианта: все дни / дни того же типа)
  norm     медиана того же интервала за последние 4 дня того же типа (аналог нашей нормы — недель здесь мало)
  lgbm4    LightGBM по правилам PVCGN: видит только последние 4 интервала (+ время, тип дня, станция)
  lgbm     LightGBM по нашим правилам: вся история до точки прогноза, цель — отклонение от нормы
  chronos  Chronos-2 без обучения, контекст — вся история до точки прогноза (только интервалы работы метро)
  blend    смешивание lgbm и chronos с весами по шагам, подобранными на val (как наш гибрид, без графовой сети)
Результат: results/external_<city>_<step>.csv и прогнозы results/external_<city>_<step>_preds.npz.
"""

import argparse
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecasting import RESULTS

EPS = 10.0
SPLITS = {"hangzhou": {"train": ("2019-01-01", "2019-01-18"), "val": ("2019-01-19", "2019-01-20"), "test": ("2019-01-21", "2019-01-25")},
          "shanghai": {"train": ("2016-07-01", "2016-08-31"), "val": ("2016-09-01", "2016-09-09"), "test": ("2016-09-10", "2016-09-30")}}
HOLIDAYS = {"hangzhou": ["2019-01-01"], "shanghai": ["2016-09-15", "2016-09-16", "2016-09-17"]}
WORKDAYS = {"hangzhou": [], "shanghai": ["2016-09-18"]}          # рабочая суббота/воскресенье по переносу выходных
PUBLISHED = {   # для сравнения: проверено по таблицам статей
    ("hangzhou", 15): "PVCGN: MAE 22,68 / 23,30 / 23,93 / 24,93 на 15/30/45/60 мин; HA: MAE 36,37 на 15 мин; "
                      "Graph WaveNet (по STR-GODEs): MAE 29,21 на 60 мин",
    ("shanghai", 15): "PVCGN: MAE 23,29 (15 мин), 26,29 (60 мин); HA: MAE 48,26 (15 мин); Graph WaveNet (по STR-GODEs): 30,61 (60 мин)",
    ("hangzhou", 30): "HW-DMD (Ханчжоу, входы, 30 мин): WMAPE 7,38 / 8,05 / 8,12% на 1/2/3 шага; HA 11,09%",
}


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------
class _SafeUnpickler(pickle.Unpickler):
    """Наборы PVCGN — pickle со словарями numpy; разрешены только классы numpy."""
    OK = {("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "_reconstruct"),
          ("numpy._core.multiarray", "_reconstruct")}

    def find_class(self, module, name):
        if (module, name) not in self.OK:
            raise pickle.UnpicklingError(f"в файле недопустимый объект {module}.{name}")
        if module == "numpy.core.multiarray":
            module = "numpy._core.multiarray" if hasattr(np, "_core") else module
        return super().find_class(module, name)


def _load_pkl(path):
    with open(path, "rb") as f:
        return _SafeUnpickler(f, encoding="latin1").load()


def load_city(folder: Path, city: str):
    """Непрерывный ряд по дням: Y (дни, интервалы, станции, 2 [вход, выход]), даты, время интервалов."""
    vals = {}
    for sp in ("train", "val", "test"):
        d = _load_pkl(folder / city / f"{sp}.pkl")
        for k in ("x", "y"):
            for i in range(len(d[k])):
                for j in range(d[k].shape[1]):
                    vals[pd.Timestamp(d[f"{k}time"][i, j])] = d[k][i, j]
    ts = pd.DatetimeIndex(sorted(vals))
    days = ts.normalize().unique()
    slots = pd.Index(sorted(set(ts - ts.normalize())))
    Y = np.full((len(days), len(slots)) + next(iter(vals.values())).shape, np.nan, "float32")
    di = pd.Series(np.arange(len(days)), index=days)
    si = pd.Series(np.arange(len(slots)), index=slots)
    for t, v in vals.items():
        Y[di[t.normalize()], si[t - t.normalize()]] = v
    return Y, days, slots


def aggregate(Y, slots, step):
    if step == 15:
        return Y, slots
    k = step // 15
    n = Y.shape[1] // k
    return Y[:, :n * k].reshape(Y.shape[0], n, k, *Y.shape[2:]).sum(2), slots[:n * k:k]


def day_types(days, city):
    hol = pd.DatetimeIndex(HOLIDAYS[city])
    work = pd.DatetimeIndex(WORKDAYS[city])
    off = (days.dayofweek >= 5) | days.isin(hol)
    off &= ~days.isin(work)
    return off.astype(int)                         # 0 — рабочий день, 1 — выходной или праздник


def split_days(days, city):
    return {k: np.where((days >= a) & (days <= b))[0] for k, (a, b) in SPLITS[city].items()}


def norm_of(Y, dtype, k=4):
    """Медиана того же интервала за последние k дней того же типа (строго до дня); иначе — за любые прошлые дни."""
    N = np.full_like(Y, np.nan)
    for d in range(Y.shape[0]):
        same = [j for j in range(d - 1, -1, -1) if dtype[j] == dtype[d]][:k]
        prev = same if same else list(range(max(0, d - k), d))
        if prev:
            N[d] = np.median(Y[prev], axis=0)
    return N


# ---------------------------------------------------------------------------
# Точки прогноза и признаки
# ---------------------------------------------------------------------------
def origins(D, S, H, day_idx, first_slot):
    """(день, интервал) точек прогноза: все H шагов внутри того же дня."""
    return np.array([(d, s) for d in day_idx for s in range(first_slot, S - H + 1)])


def features(Y, N, dtype, org, H, rules):
    """Строки (точка, шаг, станция, направление). rules='pvcgn' — только последние 4 интервала, 'ours' — вся история."""
    D, S, St, C = Y.shape
    flat = Y.reshape(D * S, St, C)
    nflat = N.reshape(D * S, St, C)
    pos = org[:, 0] * S + org[:, 1]                                          # номер интервала точки прогноза
    O = len(org)
    f = {}
    full = lambda a: np.broadcast_to(a, (O, H, St, C)).reshape(-1)
    # последние 4 интервала (внутри того же дня; в начале дня — конец предыдущего)
    for k in range(1, 5):
        f[f"lag{k}"] = full(np.log1p(flat[pos - k])[:, None])
    f["slot"] = full((org[:, 1][:, None] + np.arange(H)[None, :])[:, :, None, None].astype("float32"))
    f["h"] = full(np.arange(1, H + 1, dtype="float32")[None, :, None, None])
    f["off"] = full(dtype[org[:, 0]].astype("float32")[:, None, None, None])
    f["station"] = full(np.arange(St, dtype="float32")[None, None, :, None])
    f["dir"] = full(np.arange(C, dtype="float32")[None, None, None, :])
    tgt_pos = pos[:, None] + np.arange(H)[None, :]
    if rules == "ours":
        nm_t = nflat[tgt_pos]                                                   # (O, H, St, C)
        f["log_norm"] = np.log1p(np.nan_to_num(nm_t)).reshape(-1)
        for k in range(1, 5):
            f[f"r{k}"] = full(np.log((flat[pos - k] + EPS) / (np.nan_to_num(nflat[pos - k]) + EPS))[:, None])
        # день до точки прогноза: сумма факта к сумме нормы с начала дня
        csum_y = np.zeros((O, St, C), "float32")
        csum_n = np.zeros((O, St, C), "float32")
        for i, (d, s) in enumerate(org):
            if s > 0:
                csum_y[i] = Y[d, :s].sum(0)
                csum_n[i] = np.nan_to_num(N[d, :s]).sum(0)
        f["r_day"] = full(np.log((csum_y + EPS) / (csum_n + EPS))[:, None])
        f["r_city1"] = full(np.log((flat[pos - 1].sum((1, 2)) + EPS) / (np.nan_to_num(nflat[pos - 1]).sum((1, 2)) + EPS))[:, None, None, None])
        prev_day = np.clip(org[:, 0] - 1, 0, None)
        same_yday = Y[prev_day[:, None], org[:, 1][:, None] + np.arange(H)[None, :]]       # (O, H, St, C)
        f["r_yday"] = np.log((same_yday + EPS) / (np.nan_to_num(N[prev_day[:, None], org[:, 1][:, None] + np.arange(H)[None, :]]) + EPS)).reshape(-1)
        base = np.nan_to_num(nm_t) + EPS
    else:
        base = None
    X = pd.DataFrame({k: np.ascontiguousarray(v, dtype="float32") for k, v in f.items()})
    y = flat[tgt_pos]                                                            # (O, H, St, C)
    return X, y, base


def fit_lgbm(Y, N, dtype, org_tr, org_va, H, rules, seed=42):
    Xtr, ytr, btr = features(Y, N, dtype, org_tr, H, rules)
    Xva, yva, bva = features(Y, N, dtype, org_va, H, rules)
    if rules == "ours":
        tr_t, va_t = np.log((ytr.reshape(-1) + EPS) / btr.reshape(-1)), np.log((yva.reshape(-1) + EPS) / bva.reshape(-1))
        wtr, wva = btr.reshape(-1), bva.reshape(-1)
        ok = np.isfinite(Xtr["log_norm"].to_numpy()) & np.isfinite(tr_t)
        Xtr, tr_t, wtr = Xtr[ok], tr_t[ok], wtr[ok]
    else:
        tr_t, va_t = np.log1p(ytr.reshape(-1)), np.log1p(yva.reshape(-1))
        wtr = wva = None
    params = {"objective": "regression", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 50,
              "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "seed": seed,
              "verbose": -1, "num_threads": 0}
    dtr = lgb.Dataset(Xtr, tr_t, weight=wtr, categorical_feature=["station"])
    dva = lgb.Dataset(Xva, va_t, weight=wva, reference=dtr)
    m = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def predict_lgbm(m, Y, N, dtype, org, H, rules):
    X, y, base = features(Y, N, dtype, org, H, rules)
    p = m.predict(X, num_iteration=m.best_iteration)
    p = base.reshape(-1) * np.exp(p) - EPS if rules == "ours" else np.expm1(p)
    return np.clip(p, 0, None).reshape(y.shape), y


def predict_chronos(Y, org, H, context=2048, batch=256):
    """Медиана Chronos-2; контекст — вся история до точки прогноза (не длиннее context интервалов)."""
    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda" if torch.cuda.is_available() else "cpu",
                                               torch_dtype=torch.float32)
    D, S, St, C = Y.shape
    flat = Y.reshape(D * S, St * C)
    out = np.zeros((len(org), H, St * C), "float32")
    for i, (d, s) in enumerate(org):
        pos = d * S + s
        ctx = flat[max(0, pos - context):pos].T[:, None, :].astype("float32")       # (ряды, 1, история)
        q, _ = pipe.predict_quantiles(ctx, prediction_length=H, quantile_levels=[0.5], batch_size=batch)
        out[i] = np.stack([(t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t))[0, :, 0] for t in q], 1)
        if i % 50 == 0:
            print(f"  Chronos-2: {i}/{len(org)}", flush=True)
    return np.clip(np.nan_to_num(out), 0, None).reshape(len(org), H, St, C)


# ---------------------------------------------------------------------------
# Метрики (как в PVCGN: нули в факте исключаются; WMAPE — как наш WAPE)
# ---------------------------------------------------------------------------
def metrics(p, y):
    """По шагам: MAE, RMSE, MAPE (нули исключены), WMAPE."""
    rows = []
    for h in range(y.shape[1]):
        yy, pp = y[:, h].reshape(-1), p[:, h].reshape(-1)
        m = yy != 0
        e = pp - yy
        rows.append({"шаг": h + 1, "MAE": np.abs(e[m]).mean(), "RMSE": np.sqrt((e[m] ** 2).mean()),
                     "MAPE, %": 100 * np.abs(e[m] / yy[m]).mean(), "WMAPE, %": 100 * np.abs(e).sum() / yy.sum()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="папка с hangzhou/ и shanghai/ (распакованный data.tar.gz PVCGN)")
    ap.add_argument("--city", default="hangzhou", choices=list(SPLITS))
    ap.add_argument("--step", type=int, default=15, choices=[15, 30])
    ap.add_argument("--inflow-only", action="store_true")
    ap.add_argument("--chronos", action="store_true")
    a = ap.parse_args()

    Y, days, slots = load_city(Path(a.data), a.city)
    Y, slots = aggregate(Y, slots, a.step)
    if a.inflow_only:
        Y = Y[..., :1]
    H = 4 if a.step == 15 else 3
    first = 4 if a.step == 15 else 1                     # PVCGN: точки с 06:30 (после 4 интервалов дня)
    dtype = day_types(days, a.city)
    sp = split_days(days, a.city)
    N = norm_of(Y, dtype)
    D, S = Y.shape[:2]
    org = {k: origins(D, S, H, v, first) for k, v in sp.items()}
    tr_ok = org["train"][[np.isfinite(N[d]).all() for d, _ in org["train"]]]
    print(f"{a.city}, шаг {a.step} мин: дней {D}, интервалов в дне {S}, станций {Y.shape[2]}, направлений {Y.shape[3]}; "
          f"точек прогноза train {len(tr_ok)}, val {len(org['val'])}, test {len(org['test'])}")

    preds, res = {}, []
    y_te = None
    for split in ("val", "test"):
        o = org[split]
        tidx = (o[:, 0][:, None], o[:, 1][:, None] + np.arange(H)[None, :])
        y = Y[tidx]
        if split == "test":
            y_te = y
        tr_days = sp["train"]
        ha_all = Y[tr_days].mean(0)
        ha_type = np.stack([Y[tr_days[dtype[tr_days] == t]].mean(0) if (dtype[tr_days] == t).any() else ha_all for t in (0, 1)])
        preds[(split, "ha (все дни)")] = ha_all[tidx[1]]
        preds[(split, "ha (тот же тип дня)")] = ha_type[dtype[o[:, 0]][:, None], tidx[1]]
        preds[(split, "norm")] = np.nan_to_num(N[tidx])
    for rules, name in (("pvcgn", "lgbm4"), ("ours", "lgbm")):
        m = fit_lgbm(Y, N, dtype, tr_ok, org["val"], H, rules)
        print(f"{name}: деревьев {m.best_iteration}", flush=True)
        for split in ("val", "test"):
            preds[(split, name)] = predict_lgbm(m, Y, N, dtype, org[split], H, rules)[0]
    if a.chronos:
        for split in ("val", "test"):
            preds[(split, "chronos")] = predict_chronos(Y, org[split], H)
        # смешивание по шагам в лог-пространстве, веса — по val (МНК)
        o = org["val"]
        y_va = Y[(o[:, 0][:, None], o[:, 1][:, None] + np.arange(H)[None, :])]
        beta = []
        for h in range(H):
            A = np.column_stack([np.log1p(preds[("val", "lgbm")][:, h].reshape(-1)), np.log1p(preds[("val", "chronos")][:, h].reshape(-1)),
                                 np.ones(y_va[:, h].size)])
            w = y_va[:, h].reshape(-1) + EPS
            beta.append(np.linalg.lstsq(A * np.sqrt(w)[:, None], np.log1p(y_va[:, h].reshape(-1)) * np.sqrt(w), rcond=None)[0])
        print("веса смешивания (lgbm, chronos, const) по шагам:", np.round(beta, 2).tolist())
        for split in ("val", "test"):
            pl, pc = np.log1p(preds[(split, "lgbm")]), np.log1p(preds[(split, "chronos")])
            preds[(split, "blend")] = np.clip(np.expm1(np.stack([beta[h][0] * pl[:, h] + beta[h][1] * pc[:, h] + beta[h][2]
                                                                for h in range(H)], 1)), 0, None)
    for (split, name), p in preds.items():
        o = org[split]
        y = Y[(o[:, 0][:, None], o[:, 1][:, None] + np.arange(H)[None, :])]
        for r in metrics(p, y):
            res.append({"выборка": split, "модель": name, **r})
    res = pd.DataFrame(res)
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = f"external_{a.city}_{a.step}" + ("_in" if a.inflow_only else "")
    res.to_csv(RESULTS / f"{tag}.csv", index=False)
    np.savez_compressed(RESULTS / f"{tag}_preds.npz", y_test=y_te, **{name: p for (s, name), p in preds.items() if s == "test"})
    pd.set_option("display.width", 200, "display.max_columns", 20)
    t = res[res["выборка"] == "test"].pivot_table(index="модель", columns="шаг", values=["MAE", "WMAPE, %"]).round(2)
    print(f"\n=== {a.city}, шаг {a.step} мин, test ===\n" + t.to_string())
    print("\nОпубликовано:", PUBLISHED.get((a.city, a.step), "—"))


if __name__ == "__main__":
    main()
