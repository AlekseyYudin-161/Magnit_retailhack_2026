"""
Диагностика и компенсация дрейфа для Drift-aware прогнозирования продаж.

Функции:
- psi: Population Stability Index (сила сдвига одной фичи)
- adversarial_auc: насколько различимы две выборки (covariate shift)
- compute_importance_weights: density-ratio веса для train под распределение holdout

Используется в notebooks/02_drift_diagnosis и notebooks/03_baseline.
"""


import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index между двумя распределениями одной фичи.

    Интерпретация: <0.1 стабильно, 0.1-0.25 лёгкий drift, >0.25 значимый.

    reference, current — массивы значений (NaN отфильтровываются).
    Биннинг по квантилям reference.
    """
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return np.nan
    breaks = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(breaks) < 3:  # фича почти константа / бинарная — PSI неинформативен
        return np.nan
    pr = np.histogram(reference, breaks)[0] / len(reference) + 1e-6
    pc = np.histogram(current, breaks)[0] / len(current) + 1e-6
    return float(((pc - pr) * np.log(pc / pr)).sum())


def psi_ks_table(df_ref: pd.DataFrame, df_cur: pd.DataFrame,
                 numeric_feats: list, bins: int = 10) -> pd.DataFrame:
    """
    Таблица PSI + KS по списку числовых фичей между df_ref и df_cur.
    Возвращает DataFrame, отсортированный по PSI убыванию.
    """
    rows = []
    for col in numeric_feats:
        psi_val = psi(df_ref[col].to_numpy(), df_cur[col].to_numpy(), bins=bins)
        ks_stat, ks_p = ks_2samp(df_ref[col].dropna(), df_cur[col].dropna())
        rows.append({
            "feature": col,
            "psi": psi_val,
            "ks_stat": ks_stat,
            "ks_pval": ks_p,
            "ks_drift": ks_p < 0.05,
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def _encode_categoricals(Xa: pd.DataFrame, Xb: pd.DataFrame, cat_text: list):
    """Label encoding категориальных по объединённым категориям двух выборок."""
    Xa, Xb = Xa.copy(), Xb.copy()
    for col in cat_text:
        if col in Xa.columns:
            combined = pd.concat([Xa[col], Xb[col]]).astype("category")
            Xa[col] = combined.iloc[:len(Xa)].cat.codes
            Xb[col] = combined.iloc[len(Xa):].cat.codes
    return Xa, Xb


def adversarial_auc(df_a: pd.DataFrame, df_b: pd.DataFrame,
                    feat_cols: list, cat_text: list,
                    n_splits: int = 5, seed: int = 42):
    """
    Adversarial validation: AUC классификатора, отличающего df_a (0) от df_b (1).

    AUC ~0.5 — выборки неразличимы (нет covariate shift).
    AUC →1.0 — сильный сдвиг (или leak-метка среди фичей).

    Возвращает (auc, importance_df), где importance_df отсортирован по убыванию.
    """
    Xa, Xb = _encode_categoricals(df_a[feat_cols], df_b[feat_cols], cat_text)
    X = pd.concat([Xa, Xb], axis=0).reset_index(drop=True)
    y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]

    oof = np.zeros(len(X))
    importances = np.zeros(len(feat_cols))
    for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                               num_leaves=31, max_depth=6, verbose=-1)
        m.fit(X.iloc[tr], y[tr],
              eval_set=[(X.iloc[va], y[va])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        importances += m.feature_importances_ / n_splits

    auc = roc_auc_score(y, oof)
    imp = (pd.DataFrame({"feature": feat_cols, "importance": importances})
             .sort_values("importance", ascending=False).reset_index(drop=True))
    return auc, imp


def compute_importance_weights(train: pd.DataFrame, holdout: pd.DataFrame,
                               feat_cols: list, cat_text: list,
                               power: float = 0.25, clip_range=(0.5, 3.0),
                               n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """
    Density-ratio importance weights для train под распределение holdout.

    w = (p/(1-p))^power, где p = P(магазин похож на holdout) из OOF-классификатора.
    Смягчение степенью power и клиппинг — против взрыва дисперсии при сильном дрейфе.
    Возвращает массив весов длины len(train), нормированный к среднему 1.

    ВАЖНО: feat_cols НЕ должны содержать MONTH_COUNT (детерминированная метка времени).
    """
    Xtr, Xho = _encode_categoricals(train[feat_cols], holdout[feat_cols], cat_text)
    X = pd.concat([Xtr, Xho], axis=0).reset_index(drop=True)
    y = np.r_[np.zeros(len(Xtr)), np.ones(len(Xho))]

    oof_p = np.zeros(len(X))
    for tr_idx, va_idx in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                               num_leaves=31, max_depth=6, verbose=-1)
        m.fit(X.iloc[tr_idx], y[tr_idx],
              eval_set=[(X.iloc[va_idx], y[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_p[va_idx] = m.predict_proba(X.iloc[va_idx])[:, 1]

    p_train = np.clip(oof_p[:len(Xtr)], 1e-4, 1 - 1e-4)
    w = (p_train / (1 - p_train)) ** power   # density ratio + смягчение
    w = w / w.mean()                          # нормировка
    w = np.clip(w, *clip_range)               # клип после нормировки
    w = w / w.mean()                          # ре-нормировка
    return w