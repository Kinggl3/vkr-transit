"""
Признаки для табличных моделей скользящего прогноза (LightGBM и др.).

Одна строка = (точка прогноза T, горизонт h, станция s); прогнозируемый час t = T + h − 1.
Всё, что зависит от входов, берётся только из часов < T (docs/PROTOCOL.md, раздел 3). Проверка — test_no_leak().

Целевая переменная — log((y_t + EPS) / (norm_t + EPS)): отклонение от нормы (медиана того же часа за 4 недели).
Прогноз восстанавливается как (norm_t + EPS) · exp(ŷ) − EPS.

Группы признаков:
  recency   — отклонение станции от нормы за последние 1, 3, 24, 168 ч; тот же час вчера и неделю назад
  level     — log нормы, нет ли нормы
  city      — те же отклонения в среднем по городу (общий для всех фактор)
  calendar  — календарь прогнозируемого часа, горизонт
  events    — события у станции в прогнозируемый час, сколько из 4 недель нормы тоже были с событием
  weather   — погода в прогнозируемый час: факт (обучение) или прогноз на сутки (val/test)
  closure   — закрыта ли станция в последний час, сколько часов подряд, был ли закрыт тот же час неделю назад
  static    — признаки станции
  nbr_<g>   — средние отклонения соседей по графу g: line (линии), dist (расстояние),
              od_in (10 главных источников поездок на станцию), od_dyn (динамический OD, срез по часу)
"""

import numpy as np
import pandas as pd

from baselines import EPS, weekly_norm
from evaluate import EVENT_WINDOW, KIND_VENUE, _find
from forecasting import H, Data

CAL_COLS = ["hour", "dow", "is_holiday", "is_day_before_holiday", "is_day_after_holiday", "is_school_day",
            "xmas_week", "holiday_shopping_weekend", "is_city_event", "regime_student_omny", "cp_active",
            "doy_sin", "doy_cos", "is_dst_double_hour"]
WEATHER_COLS = ["temperature_2m", "apparent_temperature", "relative_humidity_2m", "precipitation",
                "snowfall", "cloud_cover", "wind_speed_10m"]
STATIC_COLS = ["log_scale", "commute_index", "cluster", "n_lines", "in_cbd", "borough_code", "is_venue",
               "log_od_out", "latitude", "longitude"]
GRAPHS = ("line", "dist", "od_in", "od_dyn")


def _row_norm(A: np.ndarray) -> np.ndarray:
    W = (A > 0).astype("float32")
    return W / np.maximum(W.sum(1, keepdims=True), 1)


class FeatureBuilder:
    def __init__(self, data: Data, graph_dir=None):
        self.d = data
        self.norm = weekly_norm(data)
        y, nm, op = data.y, self.norm, data.open
        # Почасовое отклонение от нормы; NaN, если станция закрыта или нормы нет
        self.r = np.where(op & ~np.isnan(nm), np.log((y + EPS) / (np.nan_to_num(nm) + EPS)), np.nan).astype("float32")
        self.r_city = np.nanmean(self.r, axis=1).astype("float32")             # (часы,)
        self.cum = self._cumsums()
        self.cal = data.cov[CAL_COLS].to_numpy("float32")
        self.weather_actual = data.cov[WEATHER_COLS].to_numpy("float32")
        fc_path = data.lab / "weather_forecast_city.parquet"
        fc = pd.read_parquet(fc_path).reindex(data.index)[WEATHER_COLS] if fc_path.exists() else None
        # Прогноз погоды; где прогноза нет (до весны 2024) — факт. В val/test прогноз полный.
        self.weather_fc = None if fc is None else np.where(fc.isna(), self.weather_actual, fc.to_numpy("float32"))
        self.ev_active, self.ev_hours = self._events()
        self.static = self._static()
        self.graph_dir = graph_dir or (data.lab.parent / "graph")
        self.W = {}

    # ------------------------------------------------------------------
    def _cumsums(self):
        """Кумулятивные суммы факта и нормы по открытым часам — для окон любой длины за O(1)."""
        ok = self.d.open & ~np.isnan(self.norm)
        y = np.where(ok, self.d.y, 0).astype("float64")
        n = np.where(ok, self.norm, 0).astype("float64")
        c = ok.astype("int32")
        z = lambda a: np.vstack([np.zeros((1, a.shape[1]), a.dtype), np.cumsum(a, axis=0)])
        return z(y), z(n), z(c)

    def window_ratio(self, origins: np.ndarray, w: int) -> np.ndarray:
        """(O, N) log отношения факта к норме за часы [T−w, T−1]; NaN, если в окне меньше половины открытых часов."""
        cy, cn, cc = self.cum
        a, b = origins - w, origins
        sy, sn, sc = cy[b] - cy[a], cn[b] - cn[a], cc[b] - cc[a]
        r = np.log((sy + EPS * sc) / (sn + EPS * sc + 1e-9))
        return np.where(sc >= max(1, w // 2), r, np.nan).astype("float32")

    def _events(self):
        """(часы, станции): число активных событий у станции и часов от начала ближайшего."""
        T, N = self.d.y.shape
        active = np.zeros((T, N), "float32")
        hours = np.full((T, N), np.nan, "float32")
        path = _find("events.csv", self.d)
        if path is None:
            print("ВНИМАНИЕ: events.csv не найден — признаки событий нулевые")
            return active, hours
        ev = pd.read_csv(path, parse_dates=["start"])
        venue = self.d.static["venue"].fillna("").to_numpy()
        lo, hi = EVENT_WINDOW
        for _, e in ev.iterrows():
            cols = np.where(venue == KIND_VENUE.get(e["kind"], "?"))[0]
            if not len(cols):
                continue
            t0 = e["start"].floor("h")
            for k in range(lo, hi + 1):
                t = t0 + pd.Timedelta(hours=k)
                if t in self.d.pos.index:
                    i = self.d.pos[t]
                    active[i, cols] += 1
                    cur = hours[i, cols]
                    hours[i, cols] = np.where(np.isnan(cur) | (abs(k) < np.abs(cur)), k, cur)
        return active, hours

    def _static(self) -> np.ndarray:
        s = self.d.static
        out = pd.DataFrame(index=s.index)
        out["log_scale"] = np.log1p(s["scale_weekday"])
        out["commute_index"] = s["commute_index"]
        out["cluster"] = s["cluster"]
        out["n_lines"] = s["n_lines"]
        out["in_cbd"] = s["in_cbd"]
        out["borough_code"] = s["borough"].astype("category").cat.codes
        out["is_venue"] = (s["venue"].fillna("") != "").astype(int)
        out["log_od_out"] = np.log1p(s["od_out_flow"])
        out["latitude"], out["longitude"] = s["latitude"], s["longitude"]
        return out[STATIC_COLS].to_numpy("float32")

    def graph(self, name: str):
        if name not in self.W:
            g = self.graph_dir
            if name == "line":
                self.W[name] = _row_norm(np.load(g / "adj_line.npy"))
            elif name == "dist":
                self.W[name] = _row_norm(np.load(g / "adj_dist.npy"))
            elif name == "od_in":
                # 10 станций, откуда больше всего едут НА эту станцию (по анализу 05 входящие лучше исходящих).
                # Берём из полной матрицы: транспонированный top-10 исходящих оставил бы 28% станций без соседей.
                A = np.load(g / "adj_od.npy").T.copy()               # A[i, j] = поток j -> i
                np.fill_diagonal(A, 0)
                top = np.argsort(-A, axis=1)[:, :10]
                M = np.zeros_like(A)
                M[np.arange(len(A))[:, None], top] = 1
                self.W[name] = _row_norm(M * (A > 0))
            elif name == "od_dyn":
                # срезы динамического графа хранят 10 главных исходящих направлений — у каждой станции есть соседи
                A = np.load(g / "adj_od_dyn_top10.npy")             # (2, 5, N, N)
                self.W[name] = np.stack([_row_norm(A[k // 5, k % 5]) for k in range(10)])
            else:
                raise ValueError(name)
        return self.W[name]

    @staticmethod
    def _nbr(W: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Среднее по соседям с пропусками: R (..., N) -> (..., N)."""
        X, V = np.nan_to_num(R), (~np.isnan(R)).astype("float32")
        num, den = X @ W.T, V @ W.T
        return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan).astype("float32")

    def _dyn_slice(self, rows: np.ndarray) -> np.ndarray:
        ts = self.d.index[rows]
        band = np.select([ts.hour <= 5, ts.hour <= 9, ts.hour <= 15, ts.hour <= 19], [0, 1, 2, 3], 4)
        return (ts.dayofweek >= 5).astype(int) * 5 + band

    # ------------------------------------------------------------------
    def build(self, origins: np.ndarray, horizons=None, graphs=(), weather: str = "forecast",
              with_target: bool = True):
        """Признаки для точек origins и горизонтов horizons (номера 1..24; по умолчанию все).

        Возвращает (X: DataFrame (O·len(h)·N строк), y_log или None, вес, norm_t, маска открытых).
        Порядок строк: точка, горизонт, станция (C-порядок массива (O, h, N))."""
        d = self.d
        hs = np.arange(1, H + 1) if horizons is None else np.asarray(horizons)
        O, K, N = len(origins), len(hs), len(d.stations)
        t = origins[:, None] + hs[None, :] - 1                                   # (O, K)
        f: dict[str, np.ndarray] = {}
        full = lambda a: np.broadcast_to(a, (O, K, N)).reshape(-1)

        # recency станции (до T)
        r_last = {w: self.window_ratio(origins, w) for w in (1, 3, 24, 168)}
        for w, v in r_last.items():
            f[f"r{w}"] = full(v[:, None, :])
        f["r_same_hour_yday"] = self.r[t - 24].reshape(-1)                       # t−24 < T при h ≤ 24
        f["r_same_hour_lweek"] = self.r[t - 168].reshape(-1)
        nm = self.norm[t]
        f["log_norm"] = np.log1p(np.nan_to_num(nm, nan=0)).reshape(-1)
        f["norm_missing"] = np.isnan(nm).reshape(-1).astype("float32")
        # город
        for w in (1, 3, 24):
            cy = np.nanmean(r_last[w], axis=1)
            f[f"city_r{w}"] = full(cy[:, None, None])
        f["city_r_same_hour_yday"] = full(self.r_city[t - 24][:, :, None])
        # календарь прогнозируемого часа и горизонт
        for j, c in enumerate(CAL_COLS):
            f[c] = full(self.cal[t, j][:, :, None])
        f["horizon"] = full(hs[None, :, None].astype("float32"))
        # события
        f["ev_active"] = self.ev_active[t].reshape(-1)
        f["ev_hours_from_start"] = self.ev_hours[t].reshape(-1)
        f["ev_in_norm_weeks"] = sum((self.ev_active[t - 168 * w] > 0) for w in (1, 2, 3, 4)).reshape(-1).astype("float32")
        # погода
        src = self.weather_actual if (weather == "actual" or self.weather_fc is None) else self.weather_fc
        for j, c in enumerate(WEATHER_COLS):
            f[f"w_{c}"] = full(src[t, j][:, :, None])
        # закрытия (известны до T)
        cl = d.closed
        f["closed_last_hour"] = full(cl[origins - 1][:, None, :].astype("float32"))
        run = np.zeros((O, N), "float32")
        still = np.ones((O, N), bool)
        for k in range(1, 169):                                                  # часов подряд закрыта, до недели
            still &= cl[origins - k]
            run += still
            if not still.any():
                break
        f["closed_run_hours"] = full(run[:, None, :])
        f["closed_same_hour_lweek"] = cl[t - 168].reshape(-1).astype("float32")
        # статические признаки станции
        for j, c in enumerate(STATIC_COLS):
            f[c] = full(self.static[None, None, :, j])
        # соседи
        for g in graphs:
            if g == "od_dyn":
                Ws = self.graph(g)
                sl = self._dyn_slice(origins - 1)                                # граф среза последнего известного часа
                for w in (1, 3, 24):
                    v = np.full((O, N), np.nan, "float32")
                    for k in np.unique(sl):
                        m = sl == k
                        v[m] = self._nbr(Ws[k], r_last[w][m])
                    f[f"nbr_{g}_r{w}"] = full(v[:, None, :])
                yd = np.full((O, K, N), np.nan, "float32")
                for k in np.unique(sl):
                    m = sl == k
                    yd[m] = self._nbr(Ws[k], self.r[t[m] - 24])
                f[f"nbr_{g}_same_hour_yday"] = yd.reshape(-1)
            else:
                W = self.graph(g)
                for w in (1, 3, 24):
                    f[f"nbr_{g}_r{w}"] = full(self._nbr(W, r_last[w])[:, None, :])
                f[f"nbr_{g}_same_hour_yday"] = self._nbr(W, self.r[t - 24]).reshape(-1)

        X = pd.DataFrame({k: np.ascontiguousarray(v, dtype="float32") for k, v in f.items()})
        norm_t = nm.reshape(-1)
        if not with_target:
            return X, None, None, norm_t, None
        y = d.y[t].reshape(-1)
        ok = (d.open[t].reshape(-1)) & ~np.isnan(norm_t)
        ylog = np.log((y + EPS) / (np.nan_to_num(norm_t) + EPS)).astype("float32")
        weight = (np.nan_to_num(norm_t) + EPS).astype("float32")                # крупные станции весят больше (как в WAPE)
        return X, ylog, weight, norm_t, ok


def to_counts(pred_log: np.ndarray, norm_t: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Из лог-отклонения обратно во входы; где нормы нет — fallback (тот же час неделю назад)."""
    p = (np.nan_to_num(norm_t) + EPS) * np.exp(pred_log) - EPS
    p = np.where(np.isnan(norm_t), fallback, p)
    return np.clip(np.nan_to_num(p), 0, None).astype("float32")


def test_no_leak(data: Data, origins=None, graphs=GRAPHS) -> bool:
    """Признаки для точки T не меняются, если испортить все данные начиная с T."""
    data.y, data.closed = data.y.copy(), data.closed.copy()
    origins = data.origins("test")[[0, 700, 1500]] if origins is None else origins
    ok = True
    for T in origins:
        X0 = FeatureBuilder(data).build(np.array([T]), graphs=graphs, with_target=False)[0]
        yb, cb = data.y.copy(), data.closed.copy()
        rng = np.random.default_rng(0)
        data.y[T:] = rng.integers(0, 5000, data.y[T:].shape)
        data.closed[T:] = rng.random(data.closed[T:].shape) < 0.3
        X1 = FeatureBuilder(data).build(np.array([T]), graphs=graphs, with_target=False)[0]
        data.y, data.closed = yb, cb
        same = X0.equals(X1)
        if not same:
            bad = [c for c in X0 if not X0[c].equals(X1[c])]
            print(f"УТЕЧКА в точке {data.index[T]}: {bad}")
        ok &= same
    return ok
