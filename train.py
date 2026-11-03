import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.metrics import evaluate_predictions, save_plots
from src.model import BEST_PARAMS, DEFAULT_MODEL_DIR, PROJECT_ROOT, save_ensemble
from src.preprocessing import CATEGORICAL_COLUMNS, prepare_training_data


def train(train_path, model_dir, report_dir):
    model_dir, report_dir = Path(model_dir), Path(report_dir)
    if (model_dir / "metadata.json").exists() or list(model_dir.glob("fold_*.cbm")):
        raise FileExistsError("Модели уже существуют. Укажите другую папку через --model-dir.")
    df = pd.read_csv(train_path)
    features, y, medians = prepare_training_data(df)
    if len(y.value_counts()) != 2 or y.value_counts().min() < 5:
        raise ValueError("Для пяти фолдов нужно минимум пять строк каждого из двух классов.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models, fold_scores = [], []
    oof_probabilities = np.zeros(len(features))
    for fold, (train_idx, val_idx) in enumerate(cv.split(features, y), start=1):
        print(f"Обучение модели на фолде {fold}/5")
        model = CatBoostClassifier(**BEST_PARAMS)
        model.fit(
            features.iloc[train_idx], y.iloc[train_idx],
            eval_set=(features.iloc[val_idx], y.iloc[val_idx]), cat_features=CATEGORICAL_COLUMNS,
        )
        models.append(model)
        probabilities = model.predict_proba(features.iloc[val_idx])[:, 1]
        oof_probabilities[val_idx] = probabilities
        fold_scores.append(evaluate_predictions(y.iloc[val_idx], probabilities, 0.5))

    threshold, best_score = 0.5, 0
    for value in np.arange(0.1, 0.9, 0.01):
        score = balanced_accuracy_score(y, (oof_probabilities > value).astype(int))
        if score > best_score:
            best_score, threshold = float(score), float(value)

    metrics = {
        "evaluation": "OOF; best iteration and threshold selected using validation outcomes",
        "rows_raw": len(df), "rows_used": len(y), "rows_without_target": len(df) - len(y),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "threshold": threshold,
        "at_0_5": evaluate_predictions(y, oof_probabilities, 0.5),
        "at_selected_threshold": evaluate_predictions(y, oof_probabilities, threshold),
        "folds_at_0_5": fold_scores,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame({"source_row": features.index, "target": y.to_numpy(),
                  "probability": oof_probabilities}).to_csv(report_dir / "oof_predictions.csv", index=False)
    save_plots(y, oof_probabilities, threshold, models, list(features.columns), report_dir)
    metadata = {
        "n_models": len(models), "feature_names": list(features.columns),
        "categorical_features": CATEGORICAL_COLUMNS, "medians": medians,
        "threshold": threshold, "best_params": BEST_PARAMS,
    }
    save_ensemble(models, metadata, model_dir)
    print(f"Модели сохранены: {model_dir}")
    print(f"Порог: {threshold:.2f}; OOF Balanced Accuracy: {best_score:.4f}")
    print(f"Отчёт: {report_dir / 'metrics.json'}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Обучить ансамбль 5×CatBoost без Optuna.")
    parser.add_argument("--train", type=Path, default=PROJECT_ROOT / "data/raw/train.csv")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports")
    args = parser.parse_args()
    train(args.train, args.model_dir, args.report_dir)


if __name__ == "__main__":
    main()
