"""
LightGBM для скользящего прогноза на 24 ч: одна глобальная модель на все станции и горизонты.

  python src/train_lgbm.py --name lgbm                         # без соседей (H1)
  python src/train_lgbm.py --name lgbm_line --graphs line      # + соседи по линии (H2, H3)
  python src/train_lgbm.py --name lgbm_all --graphs line dist od_in od_dyn
  python src/train_lgbm.py --name lgbm_wactual --weather actual --splits val   # абляция «идеальный прогноз погоды» (H5)

Обучение: точки прогноза из train с шагом --stride часов, на каждую — --hpo случайных горизонтов из 24
(все станции). Последние 8 недель train — для ранней остановки, на них модель не учится.
Прогноз: все точки val и test (--pred-stride 7 — только общий подсет, для вариантов-абляций). Результаты — results/preds/<name>__<split>.npz, журнал results/runs.csv,
важность признаков results/fi_<name>.csv.
"""

import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from baselines import lagged
from config import TRAIN_END
from features import FeatureBuilder, to_counts
from forecasting import H, RESULTS, Data, Timer, log_run, save_preds

TRAIN_START = pd.Timestamp("2023-03-01")          # норме нужны 4 недели истории
ES_WEEKS = 8


def sample_rows(fb: FeatureBuilder, origins, hpo, graphs, weather, rng, chunk=200):
    """Собирает обучающие строки кусками; оставляет только открытые станции с нормой."""
    Xs, ys, ws = [], [], []
    for i in range(0, len(origins), chunk):
        o = origins[i:i + chunk]
        hs = np.sort(rng.choice(np.arange(1, H + 1), size=hpo, replace=False)) if hpo < H else None
        X, y, w, _, ok = fb.build(o, horizons=hs, graphs=graphs, weather=weather)
        Xs.append(X[ok])
        ys.append(y[ok])
        ws.append(w[ok])
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(ws)


def predict_split(model, fb: FeatureBuilder, data: Data, split, graphs, weather, stride=1, chunk=120):
    origins = data.origins(split, stride)
    week = lagged(data.y, data.open, 168)
    out = np.empty((len(origins), H, len(data.stations)), "float32")
    for i in range(0, len(origins), chunk):
        o = origins[i:i + chunk]
        X, _, _, norm_t, _ = fb.build(o, graphs=graphs, weather=weather, with_target=False)
        p = model.predict(X, num_threads=0)
        fallback = np.nan_to_num(week[data.target_index(o)]).reshape(-1)
        out[i:i + len(o)] = to_counts(p, norm_t, fallback).reshape(len(o), H, -1)
    return origins, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lgbm")
    ap.add_argument("--graphs", nargs="*", default=[], choices=["line", "dist", "od_in", "od_dyn"])
    ap.add_argument("--weather", default="forecast", choices=["forecast", "actual"],
                    help="погода на val/test; на обучении всегда фактическая")
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--hpo", type=int, default=8, help="горизонтов на точку прогноза в обучении")
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=255)
    ap.add_argument("--min-leaf", type=int, default=200)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--pred-stride", type=int, default=1,
                    help="1 — прогноз на все точки; 7 — на общий подсет (в 7 раз быстрее, для сравнения вариантов)")
    ap.add_argument("--max-origins", type=int, default=0, help="только для отладки: ограничить число точек")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    t0 = time.perf_counter()
    data = Data()
    fb = FeatureBuilder(data)
    print(f"Данные и признаки-основа: {time.perf_counter() - t0:.0f} с", flush=True)

    tr_all = data.origins("train", a.stride)
    tr_all = tr_all[tr_all >= int(data.pos[TRAIN_START])]
    es_start = int(data.pos[TRAIN_END.normalize() - pd.Timedelta(weeks=ES_WEEKS) + pd.Timedelta(days=1)])
    tr = tr_all[tr_all + H - 1 < es_start]                      # прогноз целиком до окна ранней остановки
    es = tr_all[tr_all >= es_start]
    if a.max_origins:
        tr, es = tr[:: max(1, len(tr) // a.max_origins)], es[:: max(1, len(es) // max(1, a.max_origins // 5))]
    print(f"Точек прогноза: обучение {len(tr)}, ранняя остановка {len(es)}; соседи: {a.graphs or 'нет'}", flush=True)

    with Timer() as t_fit:
        Xtr, ytr, wtr = sample_rows(fb, tr, a.hpo, a.graphs, "actual", rng)
        Xes, yes, wes = sample_rows(fb, es, H, a.graphs, "actual", rng)
        print(f"Строк: обучение {len(ytr):,}, ранняя остановка {len(yes):,}; признаков {Xtr.shape[1]}", flush=True)
        params = {"objective": "regression", "learning_rate": a.lr, "num_leaves": a.leaves,
                  "min_data_in_leaf": a.min_leaf, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                  "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 255, "seed": a.seed, "verbose": -1,
                  "num_threads": 0}
        dtr = lgb.Dataset(Xtr, ytr, weight=wtr, free_raw_data=True)
        des = lgb.Dataset(Xes, yes, weight=wes, reference=dtr)
        model = lgb.train(params, dtr, a.rounds, valid_sets=[des], valid_names=["es"],
                          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
        del Xtr, ytr, wtr, dtr, Xes, yes, wes, des
    print(f"Обучение: {t_fit.seconds / 60:.1f} мин, лучших итераций {model.best_iteration}", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    fi = pd.DataFrame({"feature": model.feature_name(), "gain": model.feature_importance("gain")})
    fi["share"] = fi["gain"] / fi["gain"].sum()
    fi.sort_values("share", ascending=False).to_csv(RESULTS / f"fi_{a.name}.csv", index=False)
    (RESULTS / "models").mkdir(exist_ok=True)
    model.save_model(str(RESULTS / "models" / f"{a.name}.txt"), num_iteration=model.best_iteration)

    params_log = {k: v for k, v in vars(a).items() if k not in ("splits", "name")}
    params_log["best_iter"] = model.best_iteration
    for split in a.splits:
        with Timer() as t:
            origins, p = predict_split(model, fb, data, split, a.graphs, a.weather, a.pred_stride)
        save_preds(a.name, split, data, origins, p)
        log_run(a.name, split, params_log, t_fit.seconds, t.seconds, len(origins), t.peak_mb)
        print(f"{split}: {len(origins)} точек, прогноз {t.seconds / 60:.1f} мин", flush=True)


if __name__ == "__main__":
    main()
