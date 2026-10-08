"""
Единые константы проекта: границы выборок, даты режимов, пути.

Все скрипты и ноутбуки берут даты отсюда, чтобы разбиение нигде не разъехалось.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# --- Период данных -----------------------------------------------------------
DATA_START = pd.Timestamp("2023-01-01 00:00")
DATA_END = pd.Timestamp("2026-09-24 23:00")      # последний час, опубликованный MTA при загрузке

# --- Разбиение по времени (без перемешивания) ----------------------------------
# train: 2023-01-01 – 2026-03-31, val: 2026-04-01 – 2026-06-30, test: 2026-07-01 – 2026-09-24
TRAIN_END = pd.Timestamp("2026-03-31 23:00")
VAL_END = pd.Timestamp("2026-06-30 23:00")
TEST_END = DATA_END

SPLITS = {
    "train": (DATA_START, TRAIN_END),
    "val": (TRAIN_END + pd.Timedelta(hours=1), VAL_END),
    "test": (VAL_END + pd.Timedelta(hours=1), TEST_END),
}

# --- Структурные сдвиги ----------------------------------------------------------
# С 2024/25 учебного года школьники ездят по картам OMNY: 4 поездки в день, круглосуточно,
# в выходные и каникулы (раньше — только учебные дни 5:30–20:30). Первый учебный день — 05.09.2024.
STUDENT_OMNY_START = pd.Timestamp("2024-09-05")
# Плата за въезд в центр Манхэттена (congestion pricing).
CONGESTION_PRICING_START = pd.Timestamp("2025-01-05")


def split_of(ts: pd.Series) -> pd.Series:
    """Метка выборки ('train' / 'val' / 'test') для серии временных меток."""
    out = pd.Series("train", index=ts.index, dtype="object")
    out[ts > TRAIN_END] = "val"
    out[ts > VAL_END] = "test"
    return out.astype(pd.CategoricalDtype(["train", "val", "test"], ordered=True))
