import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)


def evaluate_predictions(y, probabilities, threshold):
    predictions = (probabilities > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def save_plots(y, probabilities, threshold, models, feature_names, report_dir):
    ConfusionMatrixDisplay.from_predictions(
        y, (probabilities > threshold).astype(int), display_labels=["Нет прогрессии", "Прогрессия"],
        cmap="Blues", colorbar=False,
    )
    plt.title("Матрица ошибок (OOF)")
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    RocCurveDisplay.from_predictions(y, probabilities, name="CatBoost OOF")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.title("ROC-кривая (OOF)")
    plt.tight_layout()
    plt.savefig(report_dir / "roc_curve.png", dpi=160)
    plt.close()

    importance = np.mean([model.feature_importances_ for model in models], axis=0)
    order = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(np.array(feature_names)[order], importance[order], color="#24786d")
    ax.set_xlabel("Средняя важность CatBoost, %")
    fig.tight_layout()
    fig.savefig(report_dir / "feature_importance.png", dpi=160)
    plt.close(fig)
