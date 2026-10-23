"""
Бейзлайны для скользящего прогноза на 24 ч (docs/PROTOCOL.md, раздел «Бейзлайны»).

  snaive168   — тот же час неделю назад;
  median4w    — «норма»: медиана того же часа той же станции за 4 прошлые недели (закрытые дни не берутся);
  norm_level  — норма × поправка уровня: насколько станция последние 3 и 24 часа шла выше или ниже нормы.
                Поправка своя для каждого горизонта h, коэффициенты — линейная регрессия на train
                (3 числа на горизонт, общие для всех станций). Без соседей и без внешних признаков —
                это «потолок» того, что даёт собственная недавняя история станции.

Все три используют только часы < T (точка прогноза), минимальный лаг нормы — 168 ч > 24 ч.

  python src/baselines.py                   # val и test
  python src/baselines.py --splits val
"""

import argparse

import numpy as np
import pandas as pd

from forecasting import H, Data, Timer, log_run, save_preds

EPS = 10.0                       # сглаживание в логарифмах отношений (входов в час)
WEEKS = (1, 2, 3, 4)


def lagged(y: np.ndarray, op: np.ndarray, k: int) -> np.ndarray:
    """y со сдвигом на k часов; часы, где станция была закрыта, — NaN."""
    out = np.full_like(y, np.nan)
    out[k:] = np.where(op[:-k], y[:-k], np.nan)
    return out


def weekly_norm(data: Data, chunk: int = 2048) -> np.ndarray:
    """(часы, станции) медиана того же часа недели за 4 прошлые недели. NaN, если все 4 закрыты.

    Считается кусками по времени и сортировкой 4 значений (np.nanmedian на всём массиве требует ~1,5 ГБ)."""
    T, N = data.y.shape
    out = np.full((T, N), np.nan, dtype="float32")
    for a in range(0, T, chunk):
        b = min(T, a + chunk)
        st = np.full((len(WEEKS), b - a, N), np.nan, dtype="float32")
        for k, w in enumerate(WEEKS):
            lag = 168 * w
            lo = max(a, lag)
            if lo < b:
                src = slice(lo - lag, b - lag)
                st[k, lo - a:] = np.where(data.open[src], data.y[src], np.nan)
        st.sort(axis=0)                                   # NaN уходят в конец
        cnt = (~np.isnan(st)).sum(axis=0)
        lo_i = np.clip((cnt - 1) // 2, 0, None)
        hi_i = np.clip(cnt // 2, 0, None)
        med = 0.5 * (np.take_along_axis(st, lo_i[None], 0)[0] + np.take_along_axis(st, hi_i[None], 0)[0])
        out[a:b] = np.where(cnt > 0, med, np.nan)
    return out


def level_features(data: Data, norm: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """(O, N, 2): log-отношение факта к норме за последние 3 и 24 часа до точки прогноза.

    Если в окне станция была закрыта или нормы нет — признак 0 (поправки нет)."""
    feats = []
    for w in (3, 24):
        idx = origins[:, None] - np.arange(1, w + 1)[None, :]          # часы T-1 ... T-w
        y, nm, op = data.y[idx], norm[idx], data.open[idx]
        ok = op.all(axis=1) & ~np.isnan(nm).any(axis=1)
        r = np.log((y.sum(1) + EPS * w) / (np.nan_to_num(nm).sum(1) + EPS * w))
        feats.append(np.where(ok, r, 0.0))
    return np.stack(feats, axis=-1).astype("float32")


def fit_level(data: Data, norm: np.ndarray, stride: int = 7, start: str = "2023-03-01") -> np.ndarray:
    """Коэффициенты (24, 3): [свободный член, при 3 ч, при 24 ч] для каждого горизонта. Только train."""
    a = int(data.pos[pd.Timestamp(start)])
    origins = data.origins("train", stride)
    origins = origins[origins >= a]
    X = level_features(data, norm, origins)                            # (O, N, 2)
    coef = np.zeros((H, 3), dtype="float64")
    for h in range(H):                                                 # по одному горизонту — меньше памяти
        rows = origins + h
        y, op, nm = data.y[rows], data.open[rows], norm[rows]          # (O, N)
        ok = op & ~np.isnan(nm) & (nm >= 5)
        t = np.log((y[ok] + EPS) / (nm[ok] + EPS))
        A = np.c_[np.ones(ok.sum()), X[ok]]
        coef[h] = np.linalg.lstsq(A, t, rcond=None)[0]
    return coef


def predict(name: str, data: Data, norm: np.ndarray, origins: np.ndarray, coef=None) -> np.ndarray:
    tidx = data.target_index(origins)
    nm = norm[tidx]
    week = lagged(data.y, data.open, 168)[tidx]
    if name == "median4w":
        p = np.where(np.isnan(nm), week, nm)
    elif name == "snaive168":
        p = np.where(np.isnan(week), nm, week)
    elif name == "norm_level":
        X = level_features(data, norm, origins)
        p = np.empty_like(nm)
        for h in range(H):
            adj = coef[h, 0] + X @ coef[h, 1:]
            p[:, h] = (np.nan_to_num(nm[:, h]) + EPS) * np.exp(adj) - EPS
        p = np.where(np.isnan(nm), week, p)
    else:
        raise ValueError(name)
    return np.clip(np.nan_to_num(p, nan=0.0), 0, None).astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    a = ap.parse_args()
    data = Data()
    with Timer() as t_norm:
        norm = weekly_norm(data)
    with Timer() as t_fit:
        coef = fit_level(data, norm)
    print("Коэффициенты поправки уровня (h: свободный, 3 ч, 24 ч):")
    for h in (0, 2, 5, 11, 23):
        print(f"  h={h + 1:>2}: {coef[h, 0]:+.3f}  {coef[h, 1]:+.3f}  {coef[h, 2]:+.3f}")
    for split in a.splits:
        origins = data.origins(split)
        for name in ("snaive168", "median4w", "norm_level"):
            with Timer() as t:
                p = predict(name, data, norm, origins, coef)
            path = save_preds(name, split, data, origins, p)
            fit_s = t_fit.seconds if name == "norm_level" else 0.0
            log_run(name, split, {"eps": EPS, "weeks": len(WEEKS)} if name != "snaive168" else {},
                    fit_s, t.seconds + (t_norm.seconds if name != "snaive168" else 0), len(origins), t.peak_mb)
            print(f"{split} {name}: {len(origins)} точек прогноза -> {path.name}")


if __name__ == "__main__":
    main()
