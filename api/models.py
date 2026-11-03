from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PatientInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", allow_inf_nan=False)

    gender: Literal["М", "Ж"] = Field(alias="Пол")
    birth_date: str = Field(alias="Дата рождения")
    diagnosis: str = Field(alias="Онкологический диагноз", min_length=1)
    diagnosis_date: str | None = Field(
        default=None, alias="Дата постановки онкологического диагноза / начала первичного лечения"
    )
    primary_removal_date: str | None = Field(default=None, alias="Дата удаления первичного очага")
    brain_metastases_date: str | None = Field(default=None, alias="Дата развития МГМ")
    activating_mutations: Literal["есть", "нет"] | None = Field(default=None, alias="Активирующие мутации")
    wbrt_date: str | None = Field(default=None, alias="Дата проведения ОВГМ")
    brain_surgery_date: str | None = Field(default=None, alias="Дата операции на ГМ")
    radiosurgery_count: int = Field(alias="Число РХ процедур на ГН", ge=0, strict=True)
    first_radiosurgery_date: str = Field(alias="Дата 1-ой РХ")
    karnofsky_index: int = Field(alias="Индекс Карновского", ge=0, le=100, strict=True)
    lesion_count: int = Field(alias="Число очагов в ГМ", ge=0, strict=True)
    total_lesion_volume: float = Field(alias="Суммарный объем очагов", ge=0)
    extracranial_metastases: str | None = Field(default=None, alias="Экстракраниальные метастазы")
    drug_treatment: str | None = Field(default=None, alias="Лекарственное лечение")

    @field_validator(
        "birth_date", "diagnosis_date", "primary_removal_date", "brain_metastases_date",
        "wbrt_date", "brain_surgery_date", "first_radiosurgery_date",
    )
    @classmethod
    def validate_date(cls, value, info):
        if value is None:
            return value
        absence = {
            "primary_removal_date": "не удален", "wbrt_date": "нет", "brain_surgery_date": "нет"
        }
        if value == absence.get(info.field_name):
            return value
        try:
            datetime.strptime(value, "%d.%m.%Y")
        except ValueError as error:
            raise ValueError("Ожидается существующая дата в формате ДД.ММ.ГГГГ.") from error
        return value


class PredictionOutput(BaseModel):
    probability: float
    prediction: Literal[0, 1]
    threshold: float
    status: str
