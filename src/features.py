"""
Подготовка признаков и пост-калибровка для drift-aware прогнозирования.

Финальная конфигурация (по итогам экспериментов в notebooks/03_baseline.ipynb):
- модель: CatBoost (сырые категориальные через cat_features)
- фичи: 52 базовые колонки минус ID, оба таргета, MONTH_COUNT
- MONTH_COUNT — НЕ фича модели (детерминированная метка времени → leak),
  но используется для пост-калибровки dynamic_k
- FE-надстройки (FAMILIES_PER_SQ, RENT_REL_REGION и др.) проверены
  и отвергнуты — не дали переносимого прироста (см. notebooks/03_baseline.ipynb)

Используется в train_model.py, pipeline.py и ноутбуках.
"""


import numpy as np
import pandas as pd


# ============ КОНФИГУРАЦИЯ ФИЧЕЙ ============
# Категориальные для CatBoost (Сырые cat_features, без частотного кодирования.)
CAT_COLS_CB = ["REGION", "BRANCH", "SUBJECT", "CITY"]


# ============ КОЛОНКИ, ИСКЛЮЧЁННЫЕ ИЗ ФИЧЕЙ ============
# - ID — идентификатор
# - TARGET_1/2 — таргеты (в holdout их нет)
# - MONTH_COUNT — детерминированная метка времени (leak как фича модели).
#       Диапазоны train (37-318), val (13-36), holdout (1-12) не пересекаются
#     → как фича модели создаёт leak и экстраполяцию вслепую.
#       ВАЖНО: используется отдельно для пост-калибровки dynamic_k, но НЕ в модели.
DROP_FROM_FEATURES = {"ID", "TARGET_1", "TARGET_2", "MONTH_COUNT"}


def get_feat_cb(df: pd.DataFrame) -> list:
    """
    Возвращает список фичей для CatBoost из колонок df.
    Применимо к train/val/holdout — у всех одинаковая структура колонок (минус таргеты).
    """
    return [c for c in df.columns if c not in DROP_FROM_FEATURES]


def prepare_for_catboost(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """
    Готовит DataFrame для CatBoost:
    - выбирает feat_cols
    - категориальные → str (требование CatBoost для cat_features)
    Возвращает копию (не мутирует исходный df).
    """
    out = df[feat_cols].copy()
    for c in CAT_COLS_CB:
        if c in out.columns:
            out[c] = out[c].astype(str)
    return out


# ============ ПОСТ-КАЛИБРОВКА: ДИНАМИЧЕСКИЙ МНОЖИТЕЛЬ ============
# Модель обучается только на старых магазинах (train: 37+ мес), holdout — 1-12 мес.
# Старые магазины имеют устоявшуюся клиентскую базу → модель завышает прогноз для новых.
# Компенсируем глобальным занижением, причём чем моложе магазин — тем сильнее.
#
# Параметры подобраны на val + проверены на old→young (см. notebooks/03_baseline.ipynb):
#   val MAPE:        15.00% → 14.69% (статический k=0.88 → динамический)
#   old→young MAPE:  15.49% → 14.61% (улучшение переносится)
#
# Границы 1-36: диапазон 1 покрывает молодую часть holdout (1-12 мес) без клипа.

K_PARAMS = {
    "k_young": 0.80,    # k для самых новых магазинов (m_min месяцев)
    "k_old":   0.93,    # k для "взрослых" (m_max месяцев и старше)
    "m_min":   1,
    "m_max":   36,
}


def dynamic_k(month_count, k_young: float, k_old: float,
              m_min: int, m_max: int) -> np.ndarray:
    """
    Линейная интерполяция множителя по возрасту магазина.

    Магазин возрастом m_min мес → k_young (сильное занижение).
    Магазин возрастом m_max+ мес → k_old (слабое занижение).
    За пределами [m_min, m_max] клипуется к границам.

    Параметры:
        month_count : array-like или scalar — возраст магазинов в месяцах
        k_young     : float — множитель для самых молодых
        k_old       : float — множитель для самых старых в диапазоне
        m_min, m_max: int   — границы диапазона интерполяции

    Возвращает:
        np.ndarray той же длины с множителями k для каждого магазина.
    """
    m = np.clip(np.asarray(month_count, dtype=float), m_min, m_max)
    frac = (m - m_min) / (m_max - m_min)
    return k_young + (k_old - k_young) * frac


# ============ КЛИП ПРЕДСКАЗАНИЙ/ГРАНИЦЫ ПРОГНОЗА ============
# Таргет искусственно ограничен в данных: min=100_000, max=1_000_000.
# Применяем те же границы к прогнозам (выбросы за пределы не имеют смысла).
CLIP_LOW = 100_000
CLIP_HIGH = 1_000_000


def postprocess(pred_raw: np.ndarray, month_count, k_params: dict = None) -> np.ndarray:
    """
    Полная пост-обработка прогноза модели:
      pred_raw → × dynamic_k(month_count) → clip [CLIP_LOW, CLIP_HIGH].

    Параметры:
        pred_raw    : np.ndarray — прогнозы в исходной шкале (после expm1)
        month_count : array-like — MONTH_COUNT каждого магазина (для калибровки)
        k_params    : dict       — параметры dynamic_k (по умолчанию K_PARAMS)

    Возвращает:
        np.ndarray финальных прогнозов с применёнными k и clip.
    """
    if k_params is None:
        k_params = K_PARAMS
    k_vec = dynamic_k(month_count, **k_params)
    return np.clip(pred_raw * k_vec, CLIP_LOW, CLIP_HIGH)
