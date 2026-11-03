"""Обработка признаков и целевой переменной из исходного ноутбука."""

import numpy as np
import pandas as pd


TARGET_COLUMN = "Интракраниальная прогрессия"
OUTCOME_COLUMNS = ["Локальный рецидив", "Дистантные метастазы", TARGET_COLUMN]
DIAGNOSIS_DATE = "Дата постановки онкологического диагноза / начала первичного лечения"
DATE_COLUMNS = ["Дата 1-ой РХ", "Дата рождения", DIAGNOSIS_DATE, "Дата развития МГМ"]
NUMERIC_COLUMNS = [
    "Число РХ процедур на ГН", "Индекс Карновского", "Число очагов в ГМ",
    "Суммарный объем очагов",
]
RAW_COLUMNS = [
    "Пол", "Дата рождения", "Онкологический диагноз", DIAGNOSIS_DATE,
    "Дата удаления первичного очага", "Дата развития МГМ", "Активирующие мутации",
    "Дата проведения ОВГМ", "Дата операции на ГМ", "Число РХ процедур на ГН",
    "Дата 1-ой РХ", "Индекс Карновского", "Число очагов в ГМ",
    "Суммарный объем очагов", "Экстракраниальные метастазы", "Лекарственное лечение",
]
MEDIAN_COLUMNS = ["Возраст", "Время реагирования", "Время метастазирования"]
CATEGORICAL_COLUMNS = [
    "Онкологический диагноз", "Экстракраниальные метастазы", "Лекарственное лечение",
    "Первичный очаг до 1-ой РХ", "ОВГМ до 1-ой РХ", "Операция на ГМ до 1-ой РХ",
]
TREATMENT_COLUMNS = [
    ("Дата удаления первичного очага", "Первичный очаг до 1-ой РХ",
     "Время от удаления первичного очага до 1-ой РХ", "не удален"),
    ("Дата проведения ОВГМ", "ОВГМ до 1-ой РХ", "Время от ОВГМ до 1-ой РХ", "нет"),
    ("Дата операции на ГМ", "Операция на ГМ до 1-ой РХ",
     "Время от операции на ГМ до 1-ой РХ", "нет"),
]


def require_columns(df, columns):
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise ValueError("Отсутствуют столбцы: " + ", ".join(missing))
    if df.empty:
        raise ValueError("Входная таблица не содержит строк.")


def map_binary(value, negative_word):
    if pd.isna(value):
        return np.nan
    return int(str(value).strip().lower() != negative_word)


def determine_expected_ip(local, distant):
    has_local = isinstance(local, str) and local != "нет"
    has_distant = isinstance(distant, str) and distant != "нет"
    if has_local and has_distant:
        return "ЛР+ДМ"
    if has_local:
        return "ЛР"
    if has_distant:
        return "ДМ"
    return "нет"


def prepare_features(df, medians=None):
    require_columns(df, RAW_COLUMNS)
    # Белый список исключает ID, исходы и случайные дополнительные столбцы из модели.
    features = df[RAW_COLUMNS].copy()
    for name in DATE_COLUMNS:
        clean = (features[name].astype(str).str.replace("-", ".", regex=False)
                 .str.replace("/", ".", regex=False)
                 .str.replace(r"[^\d.]", "", regex=True))
        features[name] = pd.to_datetime(clean, format="%d.%m.%Y", errors="coerce")

    first_rx = features["Дата 1-ой РХ"]
    features["Возраст"] = ((first_rx - features["Дата рождения"]).dt.days / 365.25).round()
    reaction_time = (first_rx - features["Дата развития МГМ"]).dt.days
    features["Время реагирования"] = reaction_time.clip(lower=0)
    features["Время метастазирования"] = (
        features["Дата развития МГМ"] - features[DIAGNOSIS_DATE]
    ).dt.days
    features["Временные аномалии"] = (reaction_time < 0).astype(int)

    for date_name, flag_name, interval_name, negative_word in TREATMENT_COLUMNS:
        treatment_flag = features[date_name].apply(lambda value: map_binary(value, negative_word))
        treatment_date = pd.to_datetime(features[date_name], format="%d.%m.%Y", errors="coerce")
        features[flag_name] = treatment_flag
        # Сохраняем формулу автора, включая отрицательные интервалы после первой РХ.
        features[interval_name] = treatment_flag * (first_rx - treatment_date).dt.days
        features[flag_name] = (treatment_flag == 1) & (treatment_date <= first_rx)
        features = features.drop(columns=date_name)

    for name in NUMERIC_COLUMNS:
        features[name] = pd.to_numeric(features[name].astype(str).str.replace(",", ".", regex=False)
                                       .replace({"nan": np.nan, "None": np.nan}))
    features["Активирующие мутации"] = features["Активирующие мутации"].map({"есть": 1, "нет": 0})
    # Название и кодировка сохранены: в ноутбуке М=0, Ж=1, несмотря на название столбца.
    features["Пол-Мужской"] = features["Пол"].str.upper().map({"М": 0, "Ж": 1})
    if features["Пол-Мужской"].isna().any():
        raise ValueError("Столбец 'Пол' должен содержать М или Ж.")
    features["Пол-Мужской"] = features["Пол-Мужской"].astype(int)
    features = features.drop(columns=DATE_COLUMNS + ["Пол"])

    if medians is not None:
        for name in MEDIAN_COLUMNS:
            if medians[name] is not None:
                features[name] = features[name].fillna(medians[name])
    for name in CATEGORICAL_COLUMNS:
        features[name] = features[name].fillna("неизвестно").astype(str)
    return features


def prepare_training_data(df):
    require_columns(df, RAW_COLUMNS + OUTCOME_COLUMNS)
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    if df.empty:
        raise ValueError("Нет строк с известной целевой переменной.")
    expected = df.apply(
        lambda row: determine_expected_ip(row["Локальный рецидив"], row["Дистантные метастазы"]),
        axis=1,
    )
    # Правки относятся к позициям строк исходного train.csv, а не к ID пациента.
    if 301 in expected.index:
        expected.loc[301] = "ДМ"
    y = (expected != "нет").astype(int).rename("Прогрессия")
    features = prepare_features(df)
    suspicious_idx_reaction = [573, 759, 1, 5, 414, 155, 275, 540, 811]
    suspicious_idx_metastasis = [272, 702, 582, 475, 196, 420, 770, 1, 462, 410, 43, 803, 834, 275, 177, 307]
    # Обе группы в исходной ячейке 31 меняют именно «Время метастазирования».
    indices = features.index.intersection(suspicious_idx_reaction + suspicious_idx_metastasis)
    features.loc[indices, "Время метастазирования"] = 0
    features.loc[indices, "Временные аномалии"] = 1
    medians = {}
    for name in MEDIAN_COLUMNS:
        value = features[name].median()
        medians[name] = None if pd.isna(value) else float(value)
        if medians[name] is not None:
            features[name] = features[name].fillna(medians[name])
    return features, y, medians
