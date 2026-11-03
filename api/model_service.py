from functools import lru_cache

import pandas as pd

from src.model import DEFAULT_MODEL_DIR, load_ensemble, predict_dataframe


@lru_cache(maxsize=1)
def get_ensemble():
    return load_ensemble(DEFAULT_MODEL_DIR)


def predict_patient(patient):
    models, metadata = get_ensemble()
    df = pd.DataFrame([patient.model_dump(by_alias=True)])
    probabilities, predictions = predict_dataframe(df, models, metadata)
    prediction = int(predictions[0])
    return {
        "probability": float(probabilities[0]), "prediction": prediction,
        "threshold": metadata["threshold"],
        "status": "Прогноз: прогрессия" if prediction else "Прогноз: нет прогрессии",
    }
