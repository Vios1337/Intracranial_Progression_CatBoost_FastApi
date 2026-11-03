import json

import numpy as np
import pandas as pd
import pytest

from api.models import PatientInput
from src.model import PROJECT_ROOT
from src.preprocessing import TARGET_COLUMN


@pytest.fixture
def patient():
    return json.loads((PROJECT_ROOT / "examples/patient.json").read_text(encoding="utf-8"))


@pytest.fixture
def raw_data(patient):
    # Достаточно строк для всех ручных исправлений исходного ноутбука.
    base = PatientInput(**patient).model_dump(by_alias=True)
    rows = []
    for i in range(872):
        row = dict(base)
        row["Пол"] = "М" if i % 2 else "Ж"
        row["Онкологический диагноз"] = ["НМРЛ", "Меланома", "РМЖ"][i % 3]
        row["Число РХ процедур на ГН"] = 1 + i % 3
        row["Индекс Карновского"] = 50 + 10 * (i % 5)
        row["Число очагов в ГМ"] = 1 + i % 8
        row["Суммарный объем очагов"] = str(1 + i % 20 / 3).replace(".", ",")
        row["Объем максимального очага"] = "0,5"
        row["Активирующие мутации"] = ["есть", "нет", np.nan][i % 3]
        row["Экстракраниальные метастазы"] = ["есть", "нет", np.nan][i % 3]
        row["Дата проведения ОВГМ"] = ["нет", "01.05.2020", "01.08.2020", np.nan][i % 4]
        row["Дата удаления первичного очага"] = ["не удален", "01.03.2019", np.nan][i % 3]
        row["Дата операции на ГМ"] = ["нет", "15.06.2020", np.nan][i % 3]
        row["Локальный рецидив"] = "01.01.2021" if i % 4 == 1 else "нет"
        row["Дистантные метастазы"] = "02.02.2021" if i % 4 == 2 else "нет"
        row[TARGET_COLUMN] = "ЛР" if i % 4 == 1 else "ДМ" if i % 4 == 2 else "нет"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.loc[2, TARGET_COLUMN] = np.nan
    df.loc[3, "Дата рождения"] = np.nan
    df.loc[4, "Дата развития МГМ"] = np.nan
    df.loc[6, "Дата развития МГМ"] = "01.08.2020"
    df.loc[7, "Дата развития МГМ"] = "01.01.2018"
    df.loc[8, "Дата рождения"] = "15/06/1965"
    df.loc[9, "Дата развития МГМ"] = "01-06-2020 г."
    df.loc[301, "Локальный рецидив"] = "#REF!"
    return df
