# Drift-aware прогнозирование продаж новых магазинов

> Хакатон **MAGNIT TECH × НИУ ВШЭ Машинное обучение для ритейла 2026**.
> Cold-start регрессия продаж новых торговых точек в условиях намеренного
> covariate + concept drift между обучением и инференсом.

## TL;DR

| | |
|---|---|
| **Задача** | предсказать выручку магазинов с 1–12 мес работы по данным магазинов с 37+ мес |
| **Главный вызов** | adversarial AUC train↔holdout = 0.963 (без `MONTH_COUNT`) — сильный многомерный дрейф |
| **Финальная модель** | оттюненный CatBoost + drift-aware калибровка по возрасту |
| **Val MAPE** | 14.69% (TARGET_2) |
| **Командное место** | 6-е из 28 |

---

## Контекст задачи

Magnit использует регрессионную модель продаж для оценки новых локаций. Со временем
свойства новых магазинов перестают совпадать с обучающими: меняются форматы,
конкуренция, насыщенность зон. Модель деградирует.

Организаторы намеренно ввели дрейф в разбиение:
- **train**: 15 109 магазинов, `MONTH_COUNT` 37–318 (старые)
- **val**: 2 068 магазинов, `MONTH_COUNT` 13–36 (средние)
- **holdout**: 1 322 магазина, `MONTH_COUNT` 1–12 (**новейшие, < 1 года**)

Метрика автооценки — MAPE. Два таргета: `TARGET_1` (фактические продажи) и
`TARGET_2` (скорректированный на внешние факторы). В зачёт — лучший из двух.


---

## Главное: анализ дрейфа

### 1. Adversarial validation vs univariate PSI

| Конфигурация | AUC |
|---|---|
| Все 53 фичи | 1.000 (leak от `MONTH_COUNT`) |
| Без `MONTH_COUNT` | **0.963** (реальный shift) |

`MONTH_COUNT` — детерминированная метка времени, диапазоны срезов не пересекаются → дроп как фичи.

### 2. Univariate vs multivariate

PSI: 44 из 49 фичей «стабильны» (< 0.1). Adversarial: AUC 0.963.
Дрейф **многомерный** — каждая фича стабильна, но совокупно train↔holdout различимы.

### 3. Concept drift в наблюдаемой зоне слабый

`corr(MONTH_COUNT, TARGET_2)` = −0.065 на train. Зона holdout (1–12 мес) — слепое пятно.

### 4. old→young как защита от val-артефактов

Обучить на `MONTH_COUNT >= 60`, проверить на `< 60`. Метод принимается только если улучшает обе метрики.
Тест трижды выявил val-артефакты (`MONTH_COUNT` как фича, фильтр <100, бинная калибровка).

---

## Финальное решение

```
data/{train,val,holdout}.csv
        ↓ src/read_data.py        (BOM, пробелы в числах "1 000 000")
        ↓ src/features.py         (feat_cb: 52 фичи без MONTH_COUNT)
train_model.py  →  models/catboost_t2.cbm     (Optuna, thread_count=1)
        ↓
pipeline.py     →  output/predictions.csv     (~0.07 сек)
```

### Конфигурация

- **Модель**: CatBoost, оттюненный Optuna (20 итер, val + old→young проверка)
- **Target**: `log1p(TARGET_2)`, обратно `expm1`
- **Drift-aware калибровка**: динамический множитель по `MONTH_COUNT`
- **Clip**: [100_000, 1_000_000] (искусственные границы таргета)

### Динамическая калибровка по возрасту магазинов

Модель учится только на старых магазинах → завышает прогноз для новых
(у старых уже устоявшаяся клиентская база). Компенсируем занижением,
причём чем моложе магазин — тем сильнее:

```
k(MONTH_COUNT) = 0.80 + (0.93 - 0.80) × clip(m, 1, 36) / (36 - 1)
```

| Возраст | k | Эффект |
|---|---|---|
| 1 мес  | 0.80 | сильное занижение (магазин ещё без базы) |
| 12 мес | 0.84 | умеренное |
| 36+ мес | 0.93 | минимальное (близко к устоявшимся) |

**Параметры подобраны на val + проверены на old→young** (улучшение переносится:
val 15.00→14.69%, old→young 15.49→14.61%).

---

## Что проверили и отвергли

Решения принимались по двойному тесту (val + old→young). Принимался только тот метод,
который улучшал **обе** метрики или был нейтрален к переносу.

| Гипотеза | Эффект | Решение |
|---|---|---|
| LightGBM objective: `regression` vs `mape` vs `tweedie` | regression на log1p лучше | взят |
| TARGET_1 vs TARGET_2 | T2 устойчивее к concept drift | T2 основной |
| `MONTH_COUNT` как фича модели | val лучше, old→young хуже | артефакт → дроп |
| Importance weighting (density-ratio) | val −0.09 п.п., схлопывается при AUC 0.963 | отвергнут |
| FE-фичи (FAMILIES_PER_SQ, COMP_PRESSURE, INFRA_SCORE и др.) | ~0 эффект | отвергнут |
| Drift-robust (RENT_REL_REGION) | шум | отвергнут |
| Фильтр старых магазинов (< 100 мес) | val лучше, old→young хуже | артефакт → отвергнут |
| Сегмент по площади (флаг и раздельные модели) | ~0 или хуже | отвергнут |
| LightGBM + Target Encoding | бесполезен (CatBoost сам делает TE) | отвергнут |
| Блендинг LGBM+CB | после тюнинга CB одиночный = блендингу | одиночный CB |
| Глобальный множитель k=0.88 (статический) | val 18.98 → 15.00 (главный рывок) | заменён на динамический |
| Сегментная калибровка (CITY_TYPE, 5 бинов) | drift равномерен → нет прироста | отвергнут |
| Optuna тюнинг (20 итер) | val −0.01, old→young −0.11 (переносится) | взят |
| **Динамическая калибровка по возрасту** | **val 15.00→14.69, old→young 15.49→14.61** | **финал** |

---

## Результаты

| Метрика | Значение |
|---|---|
| Val MAPE TARGET_2 (динамический k) | 14.69% |
| Val MAPE TARGET_1 | 14.76% |
| old→young MAPE TARGET_2 | 14.61% |
| Public leaderboard - (`мой submit` со статическим k=0.88) | `0.1702` |
| Командное место (submit сокомандника) | 6 из 28 |
| Время инференса (Docker) | 0.07 сек |
| Размер Docker-образа | 899 MB |

---

## Воспроизводимость

### Deploy

```bash
docker compose up --build
```

Что произойдёт:
1. Соберётся образ (~2 мин, python:3.12-slim + catboost + numpy/pandas).
2. Запустится `pipeline.py` с контрактными путями (см. ниже).
3. В `./output/predictions.csv` появится результат (~0.5 сек инференса).

### Контракт инференса

| | Хост (твоя машина) | Контейнер |
|---|---|---|
| Вход | `./data/holdout.csv` | `/data/input/holdout.csv` (ro) |
| Выход | `./output/predictions.csv` | `/data/output/predictions.csv` |
| Модель | `./models/catboost_t2.cbm` | `/app/models/catboost_t2.cbm` |

### Локально (без Docker)

```bash
pip install -r requirements_inference.txt
python3 train_model.py                          # один раз — обучить модель и сохранить .cbm
python3 pipeline.py                             # инференс — predictions.csv в ./output/
```

### Детерминизм

- `train_model.py` использует `thread_count=1` + `seed=42` → один и тот же `.cbm` на одной машине.
- `pipeline.py` детерминирован абсолютно (load + predict, без обучения).
- `PYTHONHASHSEED=42` в Dockerfile фиксирует hash-функцию Python.

---

## Структура репозитория

```
├── README.md                                         # этот файл
├── Dockerfile                                        # инференс-образ
├── docker-compose.yml                                # запуск одной командой
├── .dockerignore
├── .gitignore
├── requirements.txt                                  # полный (разработка + ноутбуки)
├── requirements-inference.txt                        # минимум для Docker
├── train_model.py                                    # обучение → .cbm (один раз)
├── pipeline.py                                       # инференс по контракту
├── predictions.csv                                   # текущий submit
├── src/
│   ├── read_data.py                                  # чистка CSV (BOM, "1 000 000" → 1000000)
│   ├── features.py                                   # feat_cb, dynamic_k, postprocess
│   └── drift.py                                      # PSI, adversarial AUC, importance weights
├── models/
│   ├── catboost_t2.cbm                               # основная модель
│   └── catboost_t1.cbm                               # страховка (TARGET_1)
├── notebooks/
│   ├── 01_eda_cold_start.ipynb                       # EDA, артефакты данных
│   ├── 02_drift_diagnosis.ipynb                      # adversarial + PSI + leak hunting
│   └── 03_baseline.ipynb                             # baseline → ансамбли → калибровка
├── data/                                             # gitignored (NDA)
└── output/                                           # gitignored (результат pipeline)
```

---

## Стек

- **Python** 3.12
- **CatBoost**
- **LightGBM** — рассматривался, не вошёл в финал
- **scikit-learn** — adversarial validation, RidgeCV (эксперимент)
- **Optuna** — тюнинг гиперпараметров
- **pandas**, **numpy**
- **Docker + docker compose**

---

## Ограничения и честные оговорки

- **Динамический k подобран на val** (MONTH_COUNT 13–36) и экстраполируется на holdout
  (1–12). Это осознанный риск, проверен через old→young.
- **Concept drift в зоне < 12 мес ненаблюдаем** (нет таргета) — параметры калибровки
  могут быть субоптимальными для самых молодых магазинов.
- **Сокомандник сдал основное решение** (CatBoost (TARGET_2) / CatBoost + LightGBM бленд (TARGET_1) + глобальный correction для компенсации дрейфа, ~14.92% на лидерборде). 
- Моя ветка — независимая альтернатива с акцентом на drift-диагностику и динамическую калибровку по возрасту.