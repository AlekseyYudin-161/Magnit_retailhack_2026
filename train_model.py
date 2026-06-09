"""
Обучение CatBoost-моделей для drift-aware прогнозирования продаж.

ЗАПУСК (один раз, локально):
    python train_model.py

ЧТО ДЕЛАЕТ:
    1. Читает train.csv, val.csv через src.read_data
    2. Обучает CatBoost на TARGET_2 (основная) и TARGET_1 (страховка)
       с зафиксированными параметрами (детерминированно, seed=42)
    3. Сохраняет модели в models/catboost_t2.cbm и models/catboost_t1.cbm
    4. Печатает val MAPE с динамической калибровкой для контроля

Эти .cbm затем подхватывает pipeline.py (инференс в Docker).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.read_data import read_data
from src.features import (
    CAT_COLS_CB, K_PARAMS,
    get_feat_cb, prepare_for_catboost, postprocess,
)


# ============ ПАРАМЕТРЫ ============
# Зафиксированы явно (не через Optuna study) — для воспроизводимости.
# Подобраны на val в notebooks/03_baseline.ipynb через Optuna (20 итер, seed=42).
BEST_PARAMS = dict(
    iterations=3000,
    learning_rate=0.04050837781329675,
    depth=4,
    l2_leaf_reg=1.5854643368675156,
    random_strength=1.8977710745066665,
    bagging_temperature=0.9656320330745594,
    loss_function="RMSE",
    random_seed=42,
    thread_count=4,
    early_stopping_rounds=100,
    verbose=0,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def mape_safe(y_true, y_pred, eps=1e-8):
    """MAPE с защитой от деления на ноль."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > eps

    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def train_one_target(target_col, train_df, val_df, feat_cols, model_path):
    """Обучает CatBoost на target_col, сохраняет .cbm, возвращает val MAPE."""
    print(f"\n--- Обучение {target_col} ---")

    X_tr = prepare_for_catboost(train_df, feat_cols)
    X_va = prepare_for_catboost(val_df, feat_cols)

    model = CatBoostRegressor(**BEST_PARAMS)
    model.fit(
        X_tr, np.log1p(train_df[target_col]),
        cat_features=CAT_COLS_CB,
        eval_set=(X_va, np.log1p(val_df[target_col])),
        use_best_model=True,
    )

    # Валидация с динамической калибровкой по MONTH_COUNT
    pred_raw = np.expm1(model.predict(X_va))
    pred_final = postprocess(pred_raw, val_df["MONTH_COUNT"], K_PARAMS)
    val_mape = mape_safe(val_df[target_col].to_numpy(), pred_final)

    model.save_model(str(model_path))
    print(f"  best_iteration: {model.get_best_iteration()}")
    print(f"  val MAPE (dynamic k): {val_mape:.2f}%")
    print(f"  сохранено: {model_path}")
    return val_mape


def main():
    print(f"Чтение данных из {DATA_DIR}...")
    train = read_data(DATA_DIR / "train.csv")
    val = read_data(DATA_DIR / "val.csv")
    print(f"  train: {train.shape}, val: {val.shape}")

    feat_cols = get_feat_cb(train)
    print(f"  фичей: {len(feat_cols)} (ID, TARGET_1/2, MONTH_COUNT исключены)")

    # Sanity-check: train и val имеют одинаковый набор фичей
    assert set(feat_cols).issubset(val.columns), "val не содержит всех train-фичей!"

    # Обучаем обе модели (T2 — основная, T1 — страховка)
    mape_t2 = train_one_target("TARGET_2", train, val, feat_cols,
                                MODELS_DIR / "catboost_t2.cbm")
    mape_t1 = train_one_target("TARGET_1", train, val, feat_cols,
                                MODELS_DIR / "catboost_t1.cbm")

    print(f"\n{'='*50}")
    print(f"Финальный val MAPE (с динамическим k):")
    print(f"  TARGET_2: {mape_t2:.2f}%  ← основная (в predictions.csv)")
    print(f"  TARGET_1: {mape_t1:.2f}%  ← страховка")
    print(f"\nМодели сохранены в {MODELS_DIR}")


if __name__ == "__main__":
    main()