import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.model_service as service
import train as training
from api.main import app
from api.models import PatientInput
from src.metrics import evaluate_predictions
from src.model import PROJECT_ROOT, load_ensemble, predict_dataframe
from src.preprocessing import RAW_COLUMNS, TARGET_COLUMN


@pytest.fixture
def trained_project(tmp_path, raw_data, monkeypatch):
    train_path = tmp_path / "train.csv"
    raw_data.iloc[:80].to_csv(train_path, index=False)
    # Только тест ускоряет реальные модели. Рабочие параметры остаются исходными.
    monkeypatch.setitem(training.BEST_PARAMS, "iterations", 12)
    monkeypatch.setitem(training.BEST_PARAMS, "verbose", False)
    model_dir, report_dir = tmp_path / "models", tmp_path / "reports"
    metrics = training.train(train_path, model_dir, report_dir)
    return model_dir, report_dir, metrics


def test_train_predict_cli_and_api_agree(trained_project, raw_data, patient, tmp_path, monkeypatch):
    model_dir, report_dir, metrics = trained_project
    assert len(list(model_dir.glob("fold_*.cbm"))) == 5
    assert metrics["rows_used"] == 79
    assert metrics["rows_without_target"] == 1
    for name in ["metrics.json", "oof_predictions.csv", "roc_curve.png", "confusion_matrix.png", "feature_importance.png"]:
        assert (report_dir / name).stat().st_size > 0

    models, metadata = load_ensemble(model_dir)
    sample = pd.DataFrame([PatientInput(**patient).model_dump(by_alias=True)])
    sample["ID"] = 42
    input_path, output_path = tmp_path / "test.csv", tmp_path / "prediction.csv"
    sample.to_csv(input_path, index=False)
    subprocess.run(
        [sys.executable, "predict.py", "--input", str(input_path), "--output", str(output_path),
         "--model-dir", str(model_dir)], cwd=PROJECT_ROOT, check=True,
    )
    csv_result = pd.read_csv(output_path)
    probabilities, predictions = predict_dataframe(sample, models, metadata)
    assert csv_result.loc[0, "ID"] == 42
    assert csv_result.loc[0, TARGET_COLUMN] == predictions[0]
    assert csv_result.loc[0, "Вероятность прогрессии"] == pytest.approx(probabilities[0])

    monkeypatch.setattr(service, "DEFAULT_MODEL_DIR", model_dir)
    service.get_ensemble.cache_clear()
    try:
        with TestClient(app) as client:
            response = client.post("/predict", json=patient)
            assert response.status_code == 200
            assert response.json()["prediction"] == int(predictions[0])
            assert response.json()["probability"] == pytest.approx(probabilities[0])
            russian = PatientInput(**patient).model_dump(by_alias=True)
            assert client.post("/predict", json=russian).json() == response.json()
    finally:
        service.get_ensemble.cache_clear()

    # Повторная загрузка native CatBoost-файлов сохраняет результат.
    restored, restored_meta = load_ensemble(model_dir)
    np.testing.assert_array_equal(predict_dataframe(sample, restored, restored_meta)[0], probabilities)
    raw_sample = raw_data.iloc[:3][RAW_COLUMNS]
    assert len(predict_dataframe(raw_sample, restored, restored_meta)[0]) == 3


def test_model_is_not_silently_overwritten(trained_project, tmp_path):
    model_dir, _, _ = trained_project
    with pytest.raises(FileExistsError):
        training.train(tmp_path / "missing.csv", model_dir, tmp_path / "reports")


def test_api_without_models_and_static_files(patient, tmp_path, monkeypatch):
    monkeypatch.setattr(service, "DEFAULT_MODEL_DIR", tmp_path / "missing")
    service.get_ensemble.cache_clear()
    with TestClient(app) as client:
        response = client.post("/predict", json=patient)
        assert response.status_code == 503
        assert "train.py" in response.json()["detail"]
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/style.css").status_code == 200
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/docs").status_code == 200
    service.get_ensemble.cache_clear()


@pytest.mark.parametrize("field,value", [
    ("birth_date", "31.02.2020"), ("first_radiosurgery_date", "2020-07-01"),
    ("total_lesion_volume", -1), ("karnofsky_index", 101),
    ("lesion_count", 1.5), ("gender", "?"),
])
def test_api_rejects_invalid_input(patient, field, value):
    patient[field] = value
    with TestClient(app) as client:
        assert client.post("/predict", json=patient).status_code == 422


def test_corrupt_model_is_not_replaced_with_prediction(trained_project, patient, monkeypatch):
    model_dir, _, _ = trained_project
    (model_dir / "fold_1.cbm").write_bytes(b"corrupted model")
    monkeypatch.setattr(service, "DEFAULT_MODEL_DIR", model_dir)
    service.get_ensemble.cache_clear()
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.post("/predict", json=patient).status_code == 500
    service.get_ensemble.cache_clear()


def test_metrics_use_strict_threshold_comparison():
    metrics = evaluate_predictions(np.array([0, 0, 1, 1]), np.array([0.1, 0.5, 0.6, 0.2]), 0.5)
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]
    assert metrics["balanced_accuracy"] == 0.75
