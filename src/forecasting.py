"""
Общая обвязка экспериментов: данные в широком формате, точки прогноза, формат прогнозов, журнал запусков.

Постановка (docs/PROTOCOL.md): скользящий прогноз. В точке прогноза T (граница часа) известны все входы
за часы < T. Модель прогнозирует входы за часы T, T+1, ..., T+23 по всем станциям (горизонт h = 1..24,
h = 1 — час, который начинается в T). Точки прогноза идут каждый час внутри выборки.

Все модели пишут прогнозы в один формат (save_preds), а src/evaluate.py считает метрики одинаково для всех.

Пути: по умолчанию data/processed/lab; на Kaggle или в другом месте — переменная окружения VKR_LAB
(папка с ridership_wide.parquet и т. д.), VKR_RESULTS — куда писать результаты (по умолчанию results/).
"""

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import PERIODS, PROCESSED, ROOT

H = 24                                    # горизонт: 24 часа
STRIDE_COMMON = 7                         # общий подсет точек прогноза: каждая 7-я (7 и 24 взаимно просты —
                                          # точки равномерно проходят все часы суток и дни недели)
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)   # квантили для вероятностных моделей

LAB = Path(os.environ.get("VKR_LAB", PROCESSED / "lab"))
RESULTS = Path(os.environ.get("VKR_RESULTS", ROOT / "results"))
PREDS = RESULTS / "preds"


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------
class Data:
    """Широкие таблицы час × станция и всё, что нужно для оценки. Загружается один раз."""

    def __init__(self, lab: Path = LAB):
        self.lab = Path(lab)
        y = pd.read_parquet(self.lab / "ridership_wide.parquet")
        self.index = y.index                                   # часы (местное время, без пояса)
        self.stations = np.array([str(c) for c in y.columns], dtype="U")
        self.y = y.to_numpy(dtype="float32")                   # (часы, станции)
        self.closed = pd.read_parquet(self.lab / "closed_wide.parquet").to_numpy(dtype=bool)
        self.cov = pd.read_parquet(self.lab / "covariates_time.parquet")
        self.static = pd.read_parquet(self.lab / "station_static.parquet")
        self.static["station_complex_id"] = self.static["station_complex_id"].astype(str)
        self.static = self.static.set_index("station_complex_id").loc[self.stations]
        self.pos = pd.Series(np.arange(len(self.index)), index=self.index)

    @property
    def open(self) -> np.ndarray:
        return ~self.closed

    def split_bounds(self, split: str) -> tuple[int, int]:
        """Номера первого и последнего часа выборки (включительно)."""
        a, b = PERIODS[split]
        return int(self.pos[a]), int(self.pos[b])

    def origins(self, split: str, stride: int = 1) -> np.ndarray:
        """Номера часов-точек прогноза T: все 24 часа прогноза лежат внутри выборки."""
        a, b = self.split_bounds(split)
        return np.arange(a, b - H + 2, stride)

    def target_index(self, origins: np.ndarray) -> np.ndarray:
        """(O, 24) номера часов, которые прогнозируются из каждой точки."""
        return origins[:, None] + np.arange(H)[None, :]

    def targets(self, origins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Факт (O, 24, N) и маска оценки (O, 24, N): открытые станции."""
        idx = self.target_index(origins)
        return self.y[idx], self.open[idx]


# ---------------------------------------------------------------------------
# Формат прогнозов
# ---------------------------------------------------------------------------
def save_preds(model: str, split: str, data: Data, origins: np.ndarray, yhat: np.ndarray,
               quantiles: np.ndarray | None = None, q_origins: np.ndarray | None = None,
               q_levels=QUANTILES) -> Path:
    """Сохраняет прогноз: results/preds/<model>__<split>.npz.

    yhat:       (O, 24, N) точечный прогноз (медиана или среднее) для точек origins.
    quantiles:  (Oq, 24, N, Q) квантили — только для общего подсета точек q_origins (их меньше,
                иначе файл слишком большой). Необязательно.
    """
    O, h, n = yhat.shape
    assert h == H and n == len(data.stations) and O == len(origins), "форма прогноза не (O, 24, N)"
    assert np.isfinite(yhat).all(), "в прогнозе есть NaN/inf"
    PREDS.mkdir(parents=True, exist_ok=True)
    out = {"origins": data.index[origins].values.astype("datetime64[ns]"), "stations": data.stations,
           "yhat": yhat.astype("float32")}
    if quantiles is not None:
        assert q_origins is not None and quantiles.shape[:3] == (len(q_origins), H, n)
        out.update(q_origins=data.index[q_origins].values.astype("datetime64[ns]"),
                   quantiles=quantiles.astype("float16"), q_levels=np.array(q_levels))
    path = PREDS / f"{model}__{split}.npz"
    np.savez_compressed(path, **out)
    return path


def load_preds(path: Path, data: Data) -> dict:
    z = np.load(path, allow_pickle=False)
    assert (z["stations"].astype(str) == data.stations).all(), "порядок станций в прогнозе не совпадает"
    d = {"origins": data.pos[pd.DatetimeIndex(z["origins"])].to_numpy(), "yhat": z["yhat"]}
    if "quantiles" in z:
        d.update(q_origins=data.pos[pd.DatetimeIndex(z["q_origins"])].to_numpy(),
                 quantiles=z["quantiles"].astype("float32"), q_levels=z["q_levels"])
    return d


# ---------------------------------------------------------------------------
# Журнал запусков
# ---------------------------------------------------------------------------
class Timer:
    """with Timer() as t: ...; t.seconds. Заодно пиковая память процесса (если доступна)."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        self.peak_mb = peak_memory_mb()


def peak_memory_mb() -> float | None:
    try:
        import resource                                         # Linux / Kaggle
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        try:
            import psutil                                       # Windows
            return psutil.Process().memory_info().peak_wset / 2 ** 20
        except Exception:
            return None


def log_run(model: str, split: str, params: dict, fit_seconds: float, predict_seconds: float,
            n_origins: int, peak_mb: float | None = None, gpu: str | None = None, notes: str = ""):
    """Строка в results/runs.csv: что запускали, с какими параметрами, сколько стоило."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    row = {"time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "model": model, "split": split,
           "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
           "fit_s": round(fit_seconds, 1), "predict_s": round(predict_seconds, 1),
           "ms_per_origin": round(1000 * predict_seconds / max(n_origins, 1), 1),
           "peak_mb": None if peak_mb is None else round(peak_mb), "gpu": gpu or "",
           "host": platform.node(), "notes": notes}
    path = RESULTS / "runs.csv"
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
