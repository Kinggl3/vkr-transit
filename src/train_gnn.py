"""
Пространственно-временная графовая сеть (в духе Graph WaveNet, Wu et al., IJCAI 2019) для скользящего
прогноза на 24 ч. Проверяет гипотезы H2/H3 на модели, которая может брать от соседей больше, чем средние
отклонения (как в LightGBM): свёртка по графу идёт на каждом временном слое.

  python src/train_gnn.py --name gnn_none                          # без графа: та же сеть, станции независимы
  python src/train_gnn.py --name gnn_line --graphs line
  python src/train_gnn.py --name gnn_adapt --adaptive              # граф выучивается из данных (эмбеддинги станций)
  python src/train_gnn.py --name gnn_all --graphs line dist od_in --adaptive
  python src/train_gnn.py --name gnn_od_dyn --graphs od_dyn        # динамический граф: срез по часу точки прогноза

Вход для точки прогноза T: последние --hist часов по всем станциям — отклонение от нормы (как у LightGBM),
маска «станция открыта», время суток и день недели. Известное о 24 часах прогноза: календарь, события у станции,
норма, погода (обучение — факт, val/test — прогноз на сутки). Цель — log((y + 10) / (норма + 10)),
лосс — MSE с весом (норма + 10), как у LightGBM; поправка смещения по горизонтам — на неделях ранней остановки.
Всё до T — только часы < T (проверка — --leak-test).

Нужен GPU (Kaggle T4); на CPU — только для проверки на маленьком размере.
"""

import argparse
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines import EPS, lagged
from config import SPLITS
from features import FeatureBuilder
from forecasting import H, RESULTS, Data, Timer, log_run, save_preds

TRAIN_START = pd.Timestamp("2023-03-01")
ES_WEEKS = 8
GCOLS = ["is_holiday", "is_day_before_holiday", "is_day_after_holiday", "is_school_day", "xmas_week",
         "is_city_event", "holiday_shopping_weekend", "regime_student_omny", "cp_active", "doy_sin", "doy_cos"]
WCOLS = ["temperature_2m", "precipitation", "snowfall"]


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------
class Tensors:
    """Массивы час × станция и сборка батчей по точкам прогноза."""

    def __init__(self, data: Data, fb: FeatureBuilder, hist: int):
        self.d, self.fb, self.hist = data, fb, hist
        r = fb.r
        self.R = np.clip(np.nan_to_num(r), -3, 3).astype("float32")
        self.M = (~np.isnan(r)).astype("float32")
        ts = data.index
        self.T4 = np.stack([np.sin(2 * np.pi * ts.hour / 24), np.cos(2 * np.pi * ts.hour / 24),
                            np.sin(2 * np.pi * ts.dayofweek / 7), np.cos(2 * np.pi * ts.dayofweek / 7)], 1).astype("float32")
        g = data.cov[GCOLS].to_numpy("float32")
        tr_end = int(data.pos[SPLITS["train"][1]]) + 1
        wa = data.cov[WCOLS].to_numpy("float32")
        wf = (pd.read_parquet(data.lab / "weather_forecast_city.parquet").reindex(data.index)[WCOLS].to_numpy("float32")
              if (data.lab / "weather_forecast_city.parquet").exists() else wa)
        wf = np.where(np.isnan(wf), wa, wf)
        mu, sd = np.nanmean(wa[:tr_end], 0), np.nanstd(wa[:tr_end], 0) + 1e-6
        self.G_actual = np.concatenate([g, self.T4, (wa - mu) / sd], 1).astype("float32")      # (T, Cg)
        self.G_fc = np.concatenate([g, self.T4, (wf - mu) / sd], 1).astype("float32")
        self.EV = np.log1p(fb.ev_active).astype("float32")                                        # (T, N)
        ln = np.log1p(np.nan_to_num(fb.norm))
        self.LN = ((ln - ln[:tr_end].mean()) / (ln[:tr_end].std() + 1e-6)).astype("float32")
        st = fb.static
        self.S = ((st - np.nanmean(st, 0)) / (np.nanstd(st, 0) + 1e-6))
        self.S = np.nan_to_num(self.S).astype("float32")                                         # (N, S)
        self.week = lagged(data.y, data.open, 168)
        idx = data.index
        band = np.select([idx.hour <= 5, idx.hour <= 9, idx.hour <= 15, idx.hour <= 19], [0, 1, 2, 3], 4)
        self.slice = ((idx.dayofweek >= 5).astype(int) * 5 + band).astype("int64")

    @property
    def c_in(self):
        return 2 + self.T4.shape[1]

    @property
    def c_fut(self):
        return self.G_actual.shape[1] + 2

    def batch(self, origins: np.ndarray, weather: str, with_target=True):
        hidx = origins[:, None] - self.hist + np.arange(self.hist)[None, :]          # (B, L)
        fidx = origins[:, None] + np.arange(H)[None, :]                               # (B, 24)
        B, L, N = len(origins), self.hist, self.R.shape[1]
        x = np.concatenate([self.R[hidx][..., None], self.M[hidx][..., None],
                            np.broadcast_to(self.T4[hidx][:, :, None, :], (B, L, N, 4))], -1)   # (B, L, N, C)
        G = self.G_actual if weather == "actual" else self.G_fc
        fut = np.concatenate([np.broadcast_to(G[fidx][:, :, None, :], (B, H, N, G.shape[1])),
                              self.EV[fidx][..., None], self.LN[fidx][..., None]], -1)           # (B, 24, N, Cf)
        out = {"x": torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 2, 1))),          # (B, C, N, L)
               "fut": torch.from_numpy(np.ascontiguousarray(fut)),
               "slice": torch.from_numpy(self.slice[origins - 1])}
        if with_target:
            d = self.d
            nm = self.fb.norm[fidx]
            ok = d.open[fidx] & ~np.isnan(nm)
            y = np.log((d.y[fidx] + EPS) / (np.nan_to_num(nm) + EPS))
            out.update(y=torch.from_numpy(np.where(ok, y, 0).astype("float32")),
                       w=torch.from_numpy(np.where(ok, np.nan_to_num(nm) + EPS, 0).astype("float32")))
        return out


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------
class GraphConv(nn.Module):
    """Свёртка по графу 1-го порядка: [x, A1·x, A2·x, ...] → 1×1 свёртка. A — строки нормированы."""

    def __init__(self, c, n_supports):
        super().__init__()
        self.mlp = nn.Conv2d(c * (1 + n_supports), c, 1)

    def forward(self, x, supports):
        out = [x]
        for A in supports:
            if A.dim() == 2:
                out.append(torch.einsum("bcnl,mn->bcml", x, A))
            else:                                                       # своя матрица у каждого примера (B, N, N)
                out.append(torch.einsum("bcnl,bmn->bcml", x, A))
        return self.mlp(torch.cat(out, 1))


class GWNet(nn.Module):
    def __init__(self, n_nodes, c_in, c_fut, s_dim, n_supports, adaptive, C=32, skip=64, emb=10,
                 dilations=(1, 2, 4, 8, 16, 32, 64, 64), dropout=0.1):
        super().__init__()
        self.adaptive = adaptive
        n_sup = n_supports + int(adaptive)
        self.start = nn.Conv2d(c_in, C, 1)
        self.filt = nn.ModuleList([nn.Conv2d(C, C, (1, 2), dilation=(1, d)) for d in dilations])
        self.gate = nn.ModuleList([nn.Conv2d(C, C, (1, 2), dilation=(1, d)) for d in dilations])
        self.skip = nn.ModuleList([nn.Conv2d(C, skip, 1) for _ in dilations])
        self.gconv = nn.ModuleList([GraphConv(C, n_sup) for _ in dilations]) if n_sup else None
        self.res = nn.ModuleList([nn.Conv2d(C, C, 1) for _ in dilations])
        self.bn = nn.ModuleList([nn.BatchNorm2d(C) for _ in dilations])
        self.receptive = 1 + sum(dilations)
        self.e1 = nn.Parameter(torch.randn(n_nodes, emb) * 0.1)
        self.e2 = nn.Parameter(torch.randn(n_nodes, emb) * 0.1)
        self.static = nn.Linear(s_dim, 16)
        self.hz = nn.Embedding(H, 8)
        self.head = nn.Sequential(nn.Linear(skip + 16 + emb + c_fut + 8, 128), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, fut, static, supports):
        if self.adaptive:
            supports = supports + [F.softmax(F.relu(self.e1 @ self.e2.T), dim=1)]
        h = self.start(x)                                              # (B, C, N, L)
        skip = 0
        for i in range(len(self.filt)):
            r = h
            z = torch.tanh(self.filt[i](h)) * torch.sigmoid(self.gate[i](h))
            s = self.skip[i](z[..., -1:])                              # нужен только последний момент — T−1
            skip = s + (skip[..., -1:] if torch.is_tensor(skip) else 0)
            if self.gconv is not None:
                z = self.gconv[i](self.drop(z), supports)
            else:
                z = self.res[i](z)
            h = self.bn[i](z + r[..., -z.size(3):])
        node = F.relu(skip[..., -1]).transpose(1, 2)                   # (B, N, skip)
        B, N = node.shape[:2]
        st = self.static(static)                                       # (N, 16)
        hz = self.hz.weight                                            # (24, 8)
        parts = [node[:, None].expand(B, H, N, -1), st[None, None].expand(B, H, N, -1),
                 self.e1[None, None].expand(B, H, N, -1), fut, hz[None, :, None].expand(B, H, N, -1)]
        return self.head(torch.cat(parts, -1)).squeeze(-1)             # (B, 24, N)


def load_supports(fb: FeatureBuilder, graphs, device):
    """Фиксированные матрицы (N, N) и динамическая (10, N, N) — строки нормированы, как у LightGBM."""
    fixed, dyn = [], None
    for g in graphs:
        W = fb.graph(g)
        if g == "od_dyn":
            dyn = torch.tensor(W, device=device)
        else:
            fixed.append(torch.tensor(W, device=device))
    return fixed, dyn


# ---------------------------------------------------------------------------
# Обучение и прогноз
# ---------------------------------------------------------------------------
def wmse(pred, y, w):
    return (w * (pred - y) ** 2).sum() / w.sum().clamp_min(1e-6)


def run_model(model, tb: Tensors, origins, weather, static, fixed, dyn, device, bs, with_target):
    """Прогноз (логарифм отклонения) по точкам; при with_target — ещё факт и веса."""
    model.eval()
    P, Y, W = [], [], []
    with torch.no_grad():
        for i in range(0, len(origins), bs):
            b = tb.batch(origins[i:i + bs], weather, with_target)
            sup = fixed + ([dyn[b["slice"].to(device)]] if dyn is not None else [])
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                p = model(b["x"].to(device), b["fut"].to(device), static, sup)
            P.append(p.float().cpu().numpy())
            if with_target:
                Y.append(b["y"].numpy())
                W.append(b["w"].numpy())
    P = np.concatenate(P)
    return (P, np.concatenate(Y), np.concatenate(W)) if with_target else P


def fit(tb: Tensors, until, a, fixed, dyn, static, device):
    d = tb.d
    end = SPLITS[until][1]
    start = max(int(d.pos[TRAIN_START]), a.hist + 1)
    origins = np.arange(start, int(d.pos[end]) - H + 2)
    es_start = int(d.pos[end.normalize() - pd.Timedelta(weeks=ES_WEEKS) + pd.Timedelta(days=1)])
    tr = origins[origins + H - 1 < es_start]
    es = origins[origins >= es_start][::7]
    if a.max_origins:
        tr, es = tr[:a.max_origins], es[:max(4, a.max_origins // 10)]
    torch.manual_seed(a.seed)
    model = GWNet(len(d.stations), tb.c_in, tb.c_fut, tb.S.shape[1], len(fixed) + int(dyn is not None), a.adaptive,
                  C=a.channels, skip=2 * a.channels).to(device)
    assert model.receptive <= a.hist, f"история {a.hist} ч короче поля зрения сети {model.receptive} ч"
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    rng = np.random.default_rng(a.seed)
    best, best_state, bad = math.inf, None, 0
    print(f"[до конца {until}] точек: обучение {len(tr)}, ранняя остановка {len(es)}; "
          f"параметров {sum(p.numel() for p in model.parameters()):,}", flush=True)
    t0 = time.perf_counter()
    for ep in range(a.epochs):
        model.train()
        sel = rng.choice(tr, size=min(a.per_epoch, len(tr)), replace=False)
        losses = []
        for i in range(0, len(sel), a.batch):
            b = tb.batch(sel[i:i + a.batch], "actual")
            sup = fixed + ([dyn[b["slice"].to(device)]] if dyn is not None else [])
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                p = model(b["x"].to(device), b["fut"].to(device), static, sup)
            loss = wmse(p.float(), b["y"].to(device), b["w"].to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        P, Y, W = run_model(model, tb, es, "actual", static, fixed, dyn, device, a.batch, True)
        es_loss = float((W * (P - Y) ** 2).sum() / W.sum())
        print(f"  эпоха {ep + 1}: обучение {np.mean(losses):.5f}, ранняя остановка {es_loss:.5f} "
              f"({(time.perf_counter() - t0) / 60:.1f} мин)", flush=True)
        if es_loss < best - 1e-5:
            best, bad = es_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    # поправка смещения по горизонтам (как у LightGBM): Σ факта / Σ прогноза во входах на неделях ранней остановки
    P, Y, W = run_model(model, tb, es, "actual", static, fixed, dyn, device, a.batch, True)
    yc = np.where(W > 0, W * np.exp(Y) - EPS, 0)
    pc = np.where(W > 0, np.clip(W * np.exp(P) - EPS, 0, None), 0)
    factor = np.clip(yc.sum((0, 2)) / np.maximum(pc.sum((0, 2)), 1e-9), 0.9, 1.1) if a.bias_correct else np.ones(H)
    return model, factor, time.perf_counter() - t0, best


def predict_split(model, tb: Tensors, split, a, fixed, dyn, static, device, factor):
    d = tb.d
    origins = d.origins(split, a.pred_stride)
    P = run_model(model, tb, origins, a.weather, static, fixed, dyn, device, a.batch, False)
    tidx = d.target_index(origins)
    nm = tb.fb.norm[tidx]
    p = (np.nan_to_num(nm) + EPS) * np.exp(P) * factor[None, :, None] - EPS
    p = np.where(np.isnan(nm), np.nan_to_num(tb.week[tidx]), p)
    return origins, np.clip(np.nan_to_num(p), 0, None).astype("float32")


def leak_test(tb: Tensors, a, device) -> bool:
    """Прогноз сети для точки T не меняется, если испортить все данные начиная с T."""
    d = tb.d
    fixed, dyn = load_supports(tb.fb, a.graphs, device)
    static = torch.tensor(tb.S, device=device)
    torch.manual_seed(0)
    model = GWNet(len(d.stations), tb.c_in, tb.c_fut, tb.S.shape[1], len(fixed) + int(dyn is not None), a.adaptive,
                  C=8, skip=16).to(device)
    T = d.origins("test")[[0, 500]]
    p0 = run_model(model, tb, T, a.weather, static, fixed, dyn, device, 2, False)
    bak = (tb.R.copy(), tb.M.copy())
    rng = np.random.default_rng(1)
    tb.R[T.min():] = rng.normal(0, 1, tb.R[T.min():].shape)
    tb.M[T.min():] = rng.random(tb.M[T.min():].shape) < 0.5
    p1 = run_model(model, tb, T[:1], a.weather, static, fixed, dyn, device, 2, False)
    tb.R, tb.M = bak
    return bool(np.allclose(p0[:1], p1, atol=1e-5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="gnn")
    ap.add_argument("--graphs", nargs="*", default=[], choices=["line", "dist", "od_in", "od_dyn"])
    ap.add_argument("--adaptive", action="store_true", help="добавить выучиваемый граф")
    ap.add_argument("--weather", default="forecast", choices=["forecast", "actual"])
    ap.add_argument("--hist", type=int, default=192, help="часов истории (≥ поля зрения сети 192)")
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--per-epoch", type=int, default=2400, help="точек прогноза за эпоху (случайных)")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--pred-stride", type=int, default=1)
    ap.add_argument("--bias-correct", action="store_true")
    ap.add_argument("--refit-for-test", action="store_true")
    ap.add_argument("--max-origins", type=int, default=0, help="только для отладки")
    ap.add_argument("--leak-test", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Data()
    fb = FeatureBuilder(data)
    tb = Tensors(data, fb, a.hist)
    if a.leak_test:
        print("Тест утечки:", "пройден" if leak_test(tb, a, device) else "УТЕЧКА")
        return
    fixed, dyn = load_supports(fb, a.graphs, device)
    static = torch.tensor(tb.S, device=device)
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "models").mkdir(exist_ok=True)
    fitted = {}
    for split in a.splits:
        until = "val" if (split == "test" and a.refit_for_test) else "train"
        if until not in fitted:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            fitted[until] = fit(tb, until, a, fixed, dyn, static, device)
            torch.save(fitted[until][0].state_dict(), RESULTS / "models" / f"{a.name}{'' if until == 'train' else '_trainval'}.pt")
        model, factor, fit_s, best = fitted[until]
        with Timer() as t:
            origins, p = predict_split(model, tb, split, a, fixed, dyn, static, device, factor)
        save_preds(a.name, split, data, origins, p)
        gpu_mb = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else t.peak_mb
        params = {k: v for k, v in vars(a).items() if k not in ("splits", "name", "leak_test")}
        params.update(fit_until=until, es_loss=round(best, 5), bias_factor_mean=round(float(factor.mean()), 4))
        log_run(a.name, split, params, fit_s, t.seconds, len(origins), gpu_mb, gpu=gpu,
                notes="peak_mb = память GPU" if device.type == "cuda" else "")
        print(f"{split}: модель до конца {until}, {len(origins)} точек, прогноз {t.seconds / 60:.1f} мин", flush=True)


if __name__ == "__main__":
    main()
