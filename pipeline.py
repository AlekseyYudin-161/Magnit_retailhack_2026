"""
Инференс для drift-aware прогнозирования продаж новых магазинов.

Запуск:
    локально:      python3 pipeline.py
    в Docker:      python3 pipeline.py
                   (пути конфигурируются через ENV переменные)

ЧТО ДЕЛАЕТ:
    1. Читает holdout.csv через src.read_data
    2. Загружает models/catboost_t2.cbm
    3. Прогноз → expm1 → dynamic_k(MONTH_COUNT) → clip → predictions.csv
    4. Сохраняет в OUTPUT_PATH

Инференс детерминированный (нет обучения, только load + predict).
Время выполнения: ~3-5 сек.

Конфигурация через ENV (с дефолтами для локального запуска):
    INPUT_PATH   — путь к holdout.csv             (default: ./data/holdout.csv)
    OUTPUT_PATH  — путь для predictions.csv       (default: ./output/predictions.csv)
    MODEL_PATH   — путь к .cbm модели             (default: ./models/catboost_t2.cbm)

В docker-compose.yml эти ENV переопределяются на /data/input, /data/output.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.read_data import read_data
from src.features import (
    K_PARAMS, get_feat_cb, prepare_for_catboost, postprocess,
)


# ============ ПУТИ (ENV → дефолт) ============
INPUT_PATH  = Path(os.environ.get("INPUT_PATH",  ROOT / "data" / "holdout.csv"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", ROOT / "output" / "predictions.csv"))
MODEL_PATH  = Path(os.environ.get("MODEL_PATH",  ROOT / "models" / "catboost_t2.cbm"))


def main():
    t0 = time.time()
    print("=== Inference pipeline ===")
    print(f"Input:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Model:  {MODEL_PATH}")

    # 1. Проверки наличия файлов
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Не найден входной файл: {INPUT_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Не найдена модель: {MODEL_PATH}\n"
            f"Запустите train_model.py для генерации .cbm файлов."
        )

    # 2. Чтение holdout (та же read_data, что использовалась при обучении)
    holdout = read_data(INPUT_PATH)
    print(f"\nholdout: {holdout.shape}\n")

    # 3. Загрузка модели
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))
    print(f"\nМодель загружена ({model.tree_count_} деревьев)\n")

    # 4. Подготовка фичей (те же, что при обучении: 52 базовых, без MONTH_COUNT)
    feat_cols = get_feat_cb(holdout)
    X = prepare_for_catboost(holdout, feat_cols)
    print(f"\nФичей: {len(feat_cols)}\n")

    # 5. Прогноз + динамическая калибровка по возрасту магазина
    pred_raw = np.expm1(model.predict(X))
    pred_final = postprocess(pred_raw, holdout["MONTH_COUNT"], K_PARAMS)

    # 6. Сборка submit (ID + PREDICT, порядок holdout сохранён)
    submission = pd.DataFrame({
        "ID": holdout["ID"].to_numpy(),
        "PREDICT": pred_final,
    })

    # 7. Валидация формата
    assert len(submission) == len(holdout), "длина не совпадает с holdout!"
    assert submission["PREDICT"].isna().sum() == 0, "есть NaN в PREDICT!"
    assert (submission["PREDICT"] < 0).sum() == 0, "есть отрицательные!"
    assert submission["ID"].equals(holdout["ID"]), "порядок ID нарушен!"

    # 8. Сохранение
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\nСохранено: {OUTPUT_PATH}")
    print(f"Строк: {len(submission)}")
    print(f"Диапазон PREDICT: [{pred_final.min():.0f}, {pred_final.max():.0f}]")
    print(f"Время инференса: {elapsed:.2f} сек")


if __name__ == "__main__":
    main()