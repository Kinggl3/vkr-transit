"""
Данные для прототипа (app/): компактные прогнозы тестового периода и сценарии «что если».

  python src/app_data.py --model results/models/lgbm_final_trainval.txt

Пишет в app/data/:
  stations.parquet  станции: название, район, линии, координаты, площадка, масштаб
  edges.npy         пары соседних станций на линиях — для схемы сети на карте
  series.npz        по часам теста: факт, норма; прогноз на 1 час вперёд (h1) и на сутки вперёд
                    (из точки 00:00) — гибрид, LightGBM, Chronos-2; границы 80%-интервала по ошибкам на val
  scenarios.npz     для каждого дня теста: во сколько раз меняется прогноз LightGBM на сутки при сценарии
                    (дождь, ливень, жара, праздник, матч на площадке) — относительно того же прогноза без изменений
  events.csv        матчи у площадок в тестовом периоде
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from baselines import EPS
from evaluate import _find
from features import FeatureBuilder, to_counts
from forecasting import H, PREDS, ROOT, Data, load_preds

MODELS = {"hyb": "hyb_blend", "lgbm": "lgbm_final", "chronos": "chronos2"}
SCENARIOS = {
    "base": "Без изменений",
    "rain": "Дождь весь день (2 мм/ч с 7 до 22 ч)",
    "storm": "Ливень днём (10 мм/ч с 12 до 18 ч)",
    "heat": "Жара: 35 °C с 11 до 19 ч",
    "holiday": "Праздничный день",
    "yankees": "Матч на Yankee Stadium в 19:00",
    "mets": "Матч на Citi Field в 19:00",
    "barclays": "Матч в Barclays Center в 19:30",
}
VENUE = {"yankees": "Yankee Stadium", "mets": "Citi Field / USTA", "barclays": "Barclays Center"}


def by_hour(pr, data: Data, split: str, h: int):
    """Прогноз на h часов вперёд, разложенный по прогнозируемым часам выборки: (часы, N), NaN — нет прогноза."""
    a, b = data.split_bounds(split)
    out = np.full((b - a + 1, len(data.stations)), np.nan, "float32")
    t = pr["origins"] + h - 1
    ok = (t >= a) & (t <= b)
    out[t[ok] - a] = pr["yhat"][ok, h - 1]
    return out


def day_ahead(pr, data: Data, split: str):
    """Прогноз из точки 00:00 на 24 часа, разложенный по часам выборки."""
    a, b = data.split_bounds(split)
    out = np.full((b - a + 1, len(data.stations)), np.nan, "float32")
    mid = data.index[pr["origins"]].hour == 0
    for i in np.where(mid)[0]:
        t = pr["origins"][i] - a
        n = min(H, out.shape[0] - t)
        out[t:t + n] = pr["yhat"][i, :n]
    return out


def bands(y, p, open_):
    """Квантили 10% и 90% отношения (факт + EPS) / (прогноз + EPS) по станциям на val."""
    r = np.where(open_ & ~np.isnan(p), np.log((y + EPS) / (np.nan_to_num(p) + EPS)), np.nan)
    q = np.nanquantile(r, [0.1, 0.9], axis=0)
    sd = np.nanstd(r, axis=0)
    return np.exp(q).astype("float32"), sd.astype("float32")


def scenarios(data: Data, fb: FeatureBuilder, model_path: Path):
    model = lgb.Booster(model_file=str(model_path))
    a, b = data.split_bounds("test")
    days = [i for i in range(a, b - H + 2) if data.index[i].hour == 0]
    venue = data.static["venue"].fillna("").to_numpy()
    N = len(data.stations)
    ratio = np.ones((len(days), len(SCENARIOS), H, N), "float32")
    hour = np.repeat(np.arange(H), N)
    for di, o in enumerate(days):
        X, _, _, norm_t, _ = fb.build(np.array([o]), with_target=False)
        week = np.nan_to_num(data.y[o - 168:o - 168 + H]).reshape(-1)
        preds = {}
        for k in SCENARIOS:
            Z = X.copy()
            if k == "rain":
                m = (hour >= 7) & (hour <= 22)
                Z.loc[m, "w_precipitation"] = np.maximum(Z.loc[m, "w_precipitation"], 2.0)
                Z.loc[m, "w_cloud_cover"] = 100.0
                Z.loc[m, "w_relative_humidity_2m"] = np.maximum(Z.loc[m, "w_relative_humidity_2m"], 90.0)
            elif k == "storm":
                m = (hour >= 7) & (hour <= 22)
                Z.loc[m, "w_precipitation"] = np.maximum(Z.loc[m, "w_precipitation"], 2.0)
                m2 = (hour >= 12) & (hour <= 18)
                Z.loc[m2, "w_precipitation"] = 10.0
                Z.loc[m, "w_cloud_cover"] = 100.0
                Z.loc[m, "w_relative_humidity_2m"] = np.maximum(Z.loc[m, "w_relative_humidity_2m"], 95.0)
            elif k == "heat":
                m = (hour >= 11) & (hour <= 19)
                Z.loc[m, "w_temperature_2m"] = 35.0
                Z.loc[m, "w_apparent_temperature"] = 39.0
            elif k == "holiday":
                Z["is_holiday"] = 1.0
                Z["is_school_day"] = 0.0
            elif k in VENUE:
                start = 19
                st = np.tile(venue == VENUE[k], H)
                m = st & (hour >= start - 2)
                Z.loc[m, "ev_active"] = Z.loc[m, "ev_active"] + 1
                Z.loc[m, "ev_hours_from_start"] = (hour[m] - start).astype("float32")
            p = model.predict(Z[model.feature_name()], num_threads=0)
            preds[k] = to_counts(p, norm_t, week).reshape(H, N)
        for j, k in enumerate(SCENARIOS):
            ratio[di, j] = (preds[k] + EPS) / (preds["base"] + EPS)
        if di % 10 == 0:
            print(f"  сценарии: {di + 1}/{len(days)} дней", flush=True)
    return np.array(data.index[days].values), ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="файл модели LightGBM (обученной на train + val) для сценариев")
    ap.add_argument("--out", default=str(ROOT / "app" / "data"))
    a = ap.parse_args()
    data = Data()
    OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)

    s = data.static.reset_index()
    st = pd.DataFrame({"id": data.stations, "name": s["station_complex"].values, "borough": s["borough"].values,
                       "lines": s["lines"].values, "lat": s["latitude"].values, "lon": s["longitude"].values,
                       "venue": s["venue"].fillna("").values, "weekday_entries": s["scale_weekday"].round().values})
    st.to_parquet(OUT / "stations.parquet", index=False)

    fb = FeatureBuilder(data)
    adj = np.load(fb.graph_dir / "adj_line.npy")                  # рёбра схемы метро для карты
    i, j = np.nonzero(np.triu((adj + adj.T) > 0, 1))
    np.save(OUT / "edges.npy", np.stack([i, j], 1).astype("int16"))
    out = {}
    a_, b_ = data.split_bounds("test")
    out["hours"] = data.index[a_:b_ + 1].values.astype("datetime64[ns]")
    y = data.y[a_:b_ + 1]
    out["actual"] = np.where(data.open[a_:b_ + 1], y, np.nan).astype("float32")
    out["norm"] = np.nan_to_num(fb.norm[a_:b_ + 1]).astype("float16")
    av, bv = data.split_bounds("val")
    yv, ov = data.y[av:bv + 1], data.open[av:bv + 1]
    for key, name in MODELS.items():
        for split in ("val", "test"):
            f = PREDS / f"{name}__{split}.npz"
            if not f.exists():
                print(f"нет {f.name} — пропуск")
                continue
            pr = load_preds(f, data)
            h1, da = by_hour(pr, data, split, 1), day_ahead(pr, data, split)
            if split == "test":
                out[f"h1_{key}"] = h1.astype("float16")
                out[f"da_{key}"] = da.astype("float16")
            elif key == "hyb":
                out["band_h1"], out["sd_h1"] = bands(yv, h1, ov)
                out["band_da"], _ = bands(yv, da, ov)
    np.savez_compressed(OUT / "series.npz", **out)
    print("series.npz:", {k: v.shape for k, v in out.items()})

    if a.model:
        days, ratio = scenarios(data, fb, Path(a.model))
        np.savez_compressed(OUT / "scenarios.npz", days=days, keys=np.array(list(SCENARIOS)),
                            names=np.array(list(SCENARIOS.values())), ratio=ratio.astype("float16"))
        print("scenarios.npz:", ratio.shape)

    ev = _find("events.csv", data)
    if ev is not None:
        e = pd.read_csv(ev, parse_dates=["start"])
        t0, t1 = data.index[a_], data.index[b_]
        e[(e.start >= t0) & (e.start <= t1)].to_csv(OUT / "events.csv", index=False)
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 2**20:.1f} МБ")


if __name__ == "__main__":
    main()
