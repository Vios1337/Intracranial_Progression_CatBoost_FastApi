import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

from src.preprocessing import prepare_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
BEST_PARAMS = {
    "learning_rate": 0.07985748292995348,
    "depth": 4,
    "l2_leaf_reg": 4.173565289433228,
    "bootstrap_type": "Bayesian",
    "random_strength": 4.372026144360919e-05,
    "eval_metric": "AUC",
    "random_seed": 42,
    "iterations": 2000,
    "auto_class_weights": "Balanced",
    "verbose": 200,
    "allow_writing_files": False,
    "thread_count": 4,
}


def save_ensemble(models, metadata, model_dir):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for i, model in enumerate(models, start=1):
        model.save_model(str(model_dir / f"fold_{i}.cbm"))
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def load_ensemble(model_dir=DEFAULT_MODEL_DIR):
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["n_models"] != 5:
        raise ValueError("Ожидается ансамбль из пяти моделей.")
    models = []
    for i in range(1, metadata["n_models"] + 1):
        model = CatBoostClassifier()
        model_path = model_dir / f"fold_{i}.cbm"
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        model.load_model(str(model_path))
        if model.feature_names_ != metadata["feature_names"]:
            raise ValueError(f"Признаки {model_path.name} не совпадают с metadata.json.")
        models.append(model)
    return models, metadata


def predict_dataframe(df, models, metadata):
    features = prepare_features(df, metadata["medians"])
    features = features[metadata["feature_names"]]
    probabilities = np.mean([model.predict_proba(features)[:, 1] for model in models], axis=0)
    predictions = (probabilities > metadata["threshold"]).astype(int)
    return probabilities, predictions
