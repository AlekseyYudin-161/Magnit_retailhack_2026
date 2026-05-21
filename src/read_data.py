"""
Чтение данных из CSV-файла и их предварительная обработка
"""

import pandas as pd


def read_data(path): 
    df = (pd.read_csv(path, encoding="utf-8-sig", skipinitialspace=True)
            .rename(columns=lambda c: c.strip().upper()))

    for col in df.columns:
        if df[col].dtype != "object":
            continue  # уже числовой или явно нечисловой — не трогаем

        # пробуем починить: убрать пробелы-разделители тысяч + запятая→точка
        cleaned = (df[col].astype(str)
                          .str.replace(r"\s+", "", regex=True)
                          .str.replace(",", ".", regex=False))
        converted = pd.to_numeric(cleaned, errors="coerce")

        # критерий "это числовая колонка": после чистки <5% новых NaN
        new_nan = converted.isna().sum() - df[col].isna().sum()
        if new_nan / len(df) < 0.05:
            df[col] = converted  # принимаем как число

    return df