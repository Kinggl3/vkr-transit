"""
Chronos-2 (amazon/chronos-2) без обучения на наших данных — предобученная модель для временных рядов.

  python src/run_chronos.py --name chronos2                                 # только история ряда, все точки
  python src/run_chronos.py --name chronos2_cov --covariates --pred-stride 7   # + ковариаты, общий подсет

Каждая станция в каждой точке прогноза T — отдельный ряд: входы за последние --context часов до T
(часы, когда станция закрыта, — пропуски). Модель выдаёт квантили на 24 ч вперёд; точечный прогноз — медиана.

Ковариаты (--covariates), известные заранее и для прошлого, и для 24 ч прогноза:
  special_day  — праздник / день до или после / рождественская неделя / городское событие (0/1)
  event        — число событий у станции в этот час (по расписанию)
  precip, snow — осадки и снег: в прошлом — факт, на 24 ч вперёд — прогноз погоды (как у LightGBM)
Каждая ковариата для Chronos-2 — ещё один ряд, поэтому расчёт с ними в ~5 раз дороже.

Нужен GPU и пакет chronos-forecasting (на Kaggle: pip install chronos-forecasting==2.3.2).
"""

import argparse

import numpy as np

from features import FeatureBuilder
from forecasting import H, QUANTILES, STRIDE_COMMON, Data, Timer, log_run, save_preds

COV_NAMES = ("special_day", "event", "precip", "snow")


def build_covariates(fb: FeatureBuilder) -> dict:
    """Ковариаты в виде массивов (часы,) или (часы, станции): прошлое (факт) и будущее (прогноз погоды)."""
    cov = fb.d.cov
    special = np.zeros(len(cov), "float32")
    for c in ["is_holiday", "is_day_before_holiday", "is_day_after_holiday", "xmas_week", "is_city_event"]:
        special = np.maximum(special, cov[c].to_numpy("float32"))
    j = {c: i for i, c in enumerate(["temperature_2m", "apparent_temperature", "relative_humidity_2m",
                                     "precipitation", "snowfall", "cloud_cover", "wind_speed_10m"])}
    fut = fb.weather_fc if fb.weather_fc is not None else fb.weather_actual
    return {"special_day": special, "event": fb.ev_active,
            "precip_past": fb.weather_actual[:, j["precipitation"]], "precip_fut": fut[:, j["precipitation"]],
            "snow_past": fb.weather_actual[:, j["snowfall"]], "snow_fut": fut[:, j["snowfall"]]}


def make_inputs(data: Data, origins: np.ndarray, context: int, cov: dict | None):
    """Список входов для Chronos-2: по одному на (точка, станция), в порядке точка → станция."""
    N = len(data.stations)
    hist = origins[:, None] - context + np.arange(context)[None, :]          # (O, L)
    ctx = np.where(data.open[hist], data.y[hist], np.nan).astype("float32")  # (O, L, N)
    if cov is None:
        return ctx.transpose(0, 2, 1).reshape(-1, 1, context)                # (O·N, 1, L)
    fut = origins[:, None] + np.arange(H)[None, :]                           # (O, 24)
    out = []
    for o in range(len(origins)):
        sd_p, sd_f = cov["special_day"][hist[o]], cov["special_day"][fut[o]]
        pr_p, pr_f = cov["precip_past"][hist[o]], cov["precip_fut"][fut[o]]
        sn_p, sn_f = cov["snow_past"][hist[o]], cov["snow_fut"][fut[o]]
        ev_p, ev_f = cov["event"][hist[o]], cov["event"][fut[o]]
        for s in range(N):
            out.append({"target": ctx[o, :, s],
                        "past_covariates": {"special_day": sd_p, "event": ev_p[:, s], "precip": pr_p, "snow": sn_p},
                        "future_covariates": {"special_day": sd_f, "event": ev_f[:, s], "precip": pr_f, "snow": sn_f}})
    return out


def to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def predict(pipe, data: Data, origins, context, cov, batch_size, chunk):
    """(O, 24, N) медиана и (O, 24, N, Q) квантили."""
    N = len(data.stations)
    med = np.empty((len(origins), H, N), "float32")
    qs = np.empty((len(origins), H, N, len(QUANTILES)), "float32")
    for i in range(0, len(origins), chunk):
        o = origins[i:i + chunk]
        q, _ = pipe.predict_quantiles(make_inputs(data, o, context, cov), prediction_length=H,
                                      quantile_levels=list(QUANTILES), batch_size=batch_size)
        q = np.stack([to_numpy(t)[0] for t in q])                             # (len(o)·N, 24, Q)
        q = q.reshape(len(o), N, H, len(QUANTILES)).transpose(0, 2, 1, 3)
        q = np.clip(np.nan_to_num(q), 0, None)
        qs[i:i + len(o)] = q
        med[i:i + len(o)] = q[..., QUANTILES.index(0.5)]
        if (i // chunk) % 20 == 0:
            print(f"  {i + len(o)}/{len(origins)} точек", flush=True)
    return med, qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="chronos2")
    ap.add_argument("--model", default="amazon/chronos-2")
    ap.add_argument("--context", type=int, default=672, help="часов истории (672 = 4 недели)")
    ap.add_argument("--covariates", action="store_true")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--pred-stride", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=8, help="точек прогноза за один вызов модели")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(a.model, device_map=a.device, torch_dtype=torch.float32)
    data = Data()
    fb = FeatureBuilder(data) if a.covariates else None
    cov = build_covariates(fb) if a.covariates else None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    for split in a.splits:
        origins = data.origins(split, a.pred_stride)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with Timer() as t:
            med, qs = predict(pipe, data, origins, a.context, cov, a.batch_size, a.chunk)
        # квантили сохраняем только для общего подсета (иначе файл огромный)
        common = data.origins(split, STRIDE_COMMON)
        keep = np.isin(origins, common)
        q_ok = keep.sum() == len(common)
        save_preds(a.name, split, data, origins, med, quantiles=qs[keep] if q_ok else None,
                   q_origins=origins[keep] if q_ok else None)
        gpu_mb = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None
        log_run(a.name, split, {"model": a.model, "context": a.context, "covariates": a.covariates}, 0.0,
                t.seconds, len(origins), gpu_mb, gpu=gpu, notes="zero-shot; peak_mb = память GPU")
        print(f"{split}: {len(origins)} точек, {t.seconds / 60:.1f} мин", flush=True)


if __name__ == "__main__":
    main()
