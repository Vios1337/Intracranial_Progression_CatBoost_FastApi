from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.model_service import predict_patient
from api.models import PatientInput, PredictionOutput


API_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Прогноз интракраниальной прогрессии", version="1.0.0")
app.mount("/static", StaticFiles(directory=API_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(API_DIR / "templates/index.html")


@app.post("/predict", response_model=PredictionOutput)
def predict(patient: PatientInput):
    try:
        return predict_patient(patient)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Файлы обученной модели не найдены. Запустите python train.py и перезапустите сервер.",
        ) from error
