import argparse
from pathlib import Path

import pandas as pd

from src.model import DEFAULT_MODEL_DIR, PROJECT_ROOT, load_ensemble, predict_dataframe
from src.preprocessing import TARGET_COLUMN


def predict(input_path, output_path, model_dir=DEFAULT_MODEL_DIR):
    df = pd.read_csv(input_path)
    models, metadata = load_ensemble(model_dir)
    probabilities, predictions = predict_dataframe(df, models, metadata)
    result = pd.DataFrame({
        "ID": df["ID"] if "ID" in df.columns else df.index,
        TARGET_COLUMN: predictions, "Вероятность прогрессии": probabilities,
    })
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Сохранено {len(result)} прогнозов: {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Прогноз прогрессии для CSV.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/raw/test.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/predictions.csv")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    predict(args.input, args.output, args.model_dir)


if __name__ == "__main__":
    main()
