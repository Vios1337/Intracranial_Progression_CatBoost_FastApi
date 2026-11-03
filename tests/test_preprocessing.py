import hashlib
import json
import re

import numpy as np
import pandas as pd
import pytest

from src.model import PROJECT_ROOT
from src.preprocessing import (
    OUTCOME_COLUMNS, RAW_COLUMNS, TARGET_COLUMN, prepare_features, prepare_training_data,
)


def test_original_notebook_unchanged():
    path = PROJECT_ROOT / "notebooks/score_0_84.ipynb"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "481d004526bc93976be674e384a312bbf8dfa2477285ce967bb2bb296deee465"
    )


def test_matches_original_notebook_preprocessing(raw_data):
    notebook = json.loads((PROJECT_ROOT / "notebooks/score_0_84.ipynb").read_text(encoding="utf-8"))
    scope = {"pd": pd, "np": np, "re": re, "df": raw_data.copy(), "display": lambda *args: None}
    # Выполняются только ячейки обработки. Загрузка, Optuna и обучение не запускаются.
    for index in [7, 8, 11, 15, 18, 21, 22, 23, 24, 26, 30, 31, 32, 37, 39, 42, 48]:
        exec("".join(notebook["cells"][index]["source"]), scope)
    features, y, medians = prepare_training_data(raw_data)
    pd.testing.assert_frame_equal(features, scope["X"], check_dtype=False)
    pd.testing.assert_series_equal(y, scope["y"])
    assert set(medians) == {"Возраст", "Время реагирования", "Время метастазирования"}


def test_inference_ignores_ids_and_outcomes(raw_data):
    _, _, medians = prepare_training_data(raw_data)
    source = raw_data.iloc[:10].copy()
    expected = prepare_features(source[RAW_COLUMNS], medians)
    source["ID"] = range(100, 110)
    for name in OUTCOME_COLUMNS:
        source[name] = "ЭТО НЕ ПРИЗНАК"
    pd.testing.assert_frame_equal(expected, prepare_features(source, medians))
    assert "ID" not in expected.columns
    assert not set(OUTCOME_COLUMNS).intersection(expected.columns)


def test_inference_reuses_training_medians(raw_data):
    _, _, medians = prepare_training_data(raw_data)
    row = raw_data.iloc[[0]].copy()
    row["Дата рождения"] = np.nan
    row["Дата развития МГМ"] = np.nan
    features = prepare_features(row, medians)
    for name, value in medians.items():
        assert features.iloc[0][name] == value


def test_missing_target_is_dropped_and_manual_corrections_preserved(raw_data):
    features, y, _ = prepare_training_data(raw_data)
    assert 2 not in y.index
    assert y.loc[301] == 1
    assert features.loc[573, "Время метастазирования"] == 0
    assert features.loc[573, "Временные аномалии"] == 1
    assert features.loc[6, "Время реагирования"] == 0
    assert features.loc[7, "Время метастазирования"] < 0


def test_missing_column_has_clear_error(raw_data):
    with pytest.raises(ValueError, match="Индекс Карновского"):
        prepare_features(raw_data.drop(columns="Индекс Карновского"))


def test_invalid_numeric_value_fails(raw_data):
    raw_data.loc[0, "Суммарный объем очагов"] = "ошибка ввода"
    with pytest.raises(ValueError):
        prepare_features(raw_data)


def test_empty_table_fails(raw_data):
    with pytest.raises(ValueError, match="не содержит строк"):
        prepare_features(raw_data.iloc[:0])


def test_unknown_gender_fails(raw_data):
    raw_data.loc[0, "Пол"] = "неизвестно"
    with pytest.raises(ValueError, match="М или Ж"):
        prepare_features(raw_data)
