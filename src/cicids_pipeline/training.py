"""Train and evaluate the CIC-IDS2017 binary classifiers."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRAIN_FILE = Path("dataset/processed/train.parquet")
TEST_FILE = Path("dataset/processed/test.parquet")
FEATURE_FILE = Path("artifacts/feature_selection/selected_features.json")
OUTPUT_FOLDER = Path("artifacts/models")
RANDOM_STATE = 42
KNN_TRAINING_ROWS = 25_000
PREDICTION_BATCH_SIZE = 20_000


def load_selected_features(path: Path = FEATURE_FILE) -> list[str]:
    """Read the feature names created during feature selection."""
    if not path.exists():
        raise FileNotFoundError("Run select_features.py before model training")
    report = json.loads(path.read_text(encoding="utf-8"))
    return report["selected_features"]


def load_model_data(
    features: list[str],
    train_file: Path = TRAIN_FILE,
    test_file: Path = TEST_FILE,
    train_sample_size: int | None = 500_000,
):
    """Load selected columns and optionally sample the training partition."""
    columns = features + ["is_attack"]
    train = pd.read_parquet(train_file, columns=columns)
    test = pd.read_parquet(test_file, columns=columns)

    if train_sample_size and train_sample_size < len(train):
        benign = train[train["is_attack"] == 0]
        attacks = train[train["is_attack"] == 1]
        attack_ratio = len(attacks) / len(train)
        attack_rows = round(train_sample_size * attack_ratio)
        benign_rows = train_sample_size - attack_rows

        train = pd.concat(
            [
                benign.sample(benign_rows, random_state=RANDOM_STATE),
                attacks.sample(attack_rows, random_state=RANDOM_STATE),
            ]
        ).sample(frac=1, random_state=RANDOM_STATE)

    x_train = train[features]
    y_train = train["is_attack"]
    x_test = test[features]
    y_test = test["is_attack"]
    return x_train, y_train, x_test, y_test


def create_models() -> dict[str, object]:
    """Create Random Forest and the two comparison models."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "K-Nearest Neighbors": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                n_jobs=-1,
            ),
        ),
    }


def sample_training_data(x_train, y_train, sample_size):
    """Create a reproducible sample with the original class ratio."""
    if sample_size is None or sample_size >= len(x_train):
        return x_train, y_train

    data = x_train.copy()
    data["is_attack"] = y_train.to_numpy()
    benign = data[data["is_attack"] == 0]
    attacks = data[data["is_attack"] == 1]
    attack_rows = round(sample_size * len(attacks) / len(data))
    benign_rows = sample_size - attack_rows
    sample = pd.concat(
        [
            benign.sample(benign_rows, random_state=RANDOM_STATE),
            attacks.sample(attack_rows, random_state=RANDOM_STATE),
        ]
    ).sample(frac=1, random_state=RANDOM_STATE)
    return sample.drop(columns="is_attack"), sample["is_attack"]


def predict_in_batches(model, x_test, batch_size=PREDICTION_BATCH_SIZE):
    """Predict in batches so KNN does not allocate excessive memory."""
    predictions = []
    probabilities = []
    positive_class = list(model.classes_).index(1)
    batches = range(0, len(x_test), batch_size)
    for start in tqdm(batches, desc="Predicting", unit="batch", leave=False):
        batch = x_test.iloc[start : start + batch_size]
        batch_probabilities = model.predict_proba(batch)
        predictions.append(model.classes_[np.argmax(batch_probabilities, axis=1)])
        probabilities.append(batch_probabilities[:, positive_class])
    return np.concatenate(predictions), np.concatenate(probabilities)


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series):
    """Calculate binary classification metrics using batched predictions."""
    start = time.perf_counter()
    predictions, probabilities = predict_in_batches(model, x_test)
    prediction_time = time.perf_counter() - start

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "prediction_seconds": prediction_time,
    }
    return metrics, predictions, probabilities


def save_evaluation_plots(model_name, y_test, predictions, probabilities, output_folder):
    """Save the confusion matrix and ROC curve for one model."""
    file_name = model_name.lower().replace(" ", "_")

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Benign", "Attack"],
        cmap="Blues",
    )
    plt.title(f"{model_name} confusion matrix")
    plt.tight_layout()
    plt.savefig(output_folder / f"{file_name}_confusion_matrix.png", dpi=150)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, probabilities)
    plt.title(f"{model_name} ROC curve")
    plt.tight_layout()
    plt.savefig(output_folder / f"{file_name}_roc_curve.png", dpi=150)
    plt.close()


def train_models(
    train_sample_size: int | None = 500_000,
    knn_training_rows: int = KNN_TRAINING_ROWS,
) -> pd.DataFrame:
    """Train, evaluate, and save all three classifiers."""
    features = load_selected_features()
    x_train, y_train, x_test, y_test = load_model_data(
        features,
        train_sample_size=train_sample_size,
    )
    models = create_models()
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    results = []
    for model_name, model in models.items():
        model_x_train = x_train
        model_y_train = y_train
        if model_name == "K-Nearest Neighbors":
            model_x_train, model_y_train = sample_training_data(
                x_train, y_train, knn_training_rows
            )

        print(f"Training {model_name}...")
        start = time.perf_counter()
        model.fit(model_x_train, model_y_train)
        training_time = time.perf_counter() - start

        metrics, predictions, probabilities = evaluate_model(model, x_test, y_test)
        metrics.update(
            {
                "model": model_name,
                "training_rows": len(model_x_train),
                "testing_rows": len(x_test),
                "training_seconds": training_time,
            }
        )
        results.append(metrics)

        file_name = model_name.lower().replace(" ", "_")
        joblib.dump(model, OUTPUT_FOLDER / f"{file_name}.joblib")
        save_evaluation_plots(model_name, y_test, predictions, probabilities, OUTPUT_FOLDER)

    results = pd.DataFrame(results).set_index("model")
    results.to_csv(OUTPUT_FOLDER / "model_results.csv")

    report = {
        "main_model": "Random Forest",
        "selected_features": features,
        "training_sample_size": train_sample_size,
        "knn_training_rows": knn_training_rows,
        "results": results.reset_index().to_dict(orient="records"),
    }
    (OUTPUT_FOLDER / "training_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return results
