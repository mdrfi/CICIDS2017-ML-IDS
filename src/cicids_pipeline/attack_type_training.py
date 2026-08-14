"""Train a model that identifies the type of attack."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from cicids_pipeline.training import (
    FEATURE_FILE,
    RANDOM_STATE,
    TEST_FILE,
    TRAIN_FILE,
    load_selected_features,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_FOLDER = Path("artifacts/attack_type_model")
BENIGN_LABEL = "BENIGN"


def load_attack_data(
    features: list[str],
    train_file: Path = TRAIN_FILE,
    test_file: Path = TEST_FILE,
):
    """Load attack rows only, because benign traffic has no attack type."""
    columns = features + ["attack_label"]
    train = pd.read_parquet(train_file, columns=columns)
    test = pd.read_parquet(test_file, columns=columns)

    train = train[train["attack_label"] != BENIGN_LABEL]
    test = test[test["attack_label"] != BENIGN_LABEL]

    return (
        train[features],
        train["attack_label"],
        test[features],
        test["attack_label"],
    )


def create_attack_type_model(n_jobs: int = -1) -> RandomForestClassifier:
    """Create a class-balanced Random Forest for multiclass prediction."""
    return RandomForestClassifier(
        n_estimators=150,
        max_depth=25,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        n_jobs=n_jobs,
        random_state=RANDOM_STATE,
    )


def evaluate_attack_type_model(model, x_test, y_test):
    """Return overall metrics, predictions, and per-class metrics."""
    start = time.perf_counter()
    predictions = model.predict(x_test)
    prediction_seconds = time.perf_counter() - start

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "prediction_seconds": prediction_seconds,
    }
    per_class = classification_report(
        y_test,
        predictions,
        labels=model.classes_,
        output_dict=True,
        zero_division=0,
    )
    return metrics, predictions, per_class


def save_confusion_matrix(model, y_test, predictions, output_folder: Path) -> None:
    """Save a normalized confusion matrix for all attack classes."""
    figure, axis = plt.subplots(figsize=(15, 12))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=model.classes_,
        display_labels=model.classes_,
        normalize="true",
        values_format=".2f",
        cmap="Blues",
        xticks_rotation=45,
        ax=axis,
    )
    axis.set_title("Attack-type confusion matrix (normalized by true class)")
    figure.tight_layout()
    figure.savefig(output_folder / "confusion_matrix.png", dpi=150)
    plt.close(figure)


def train_attack_type_model(
    feature_file: Path = FEATURE_FILE,
    output_folder: Path = OUTPUT_FOLDER,
) -> dict:
    """Train, evaluate, and save the attack-type Random Forest."""
    features = load_selected_features(feature_file)
    x_train, y_train, x_test, y_test = load_attack_data(features)
    model = create_attack_type_model()

    print(f"Training attack-type model with {len(x_train):,} attack rows...")
    start = time.perf_counter()
    model.fit(x_train, y_train)
    training_seconds = time.perf_counter() - start

    metrics, predictions, per_class = evaluate_attack_type_model(
        model, x_test, y_test
    )
    metrics["training_seconds"] = training_seconds

    output_folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_folder / "random_forest_attack_type.joblib")
    save_confusion_matrix(model, y_test, predictions, output_folder)

    report = {
        "model": "Random Forest",
        "task": "multiclass attack-type classification",
        "selected_features": features,
        "classes": model.classes_.tolist(),
        "training_rows": len(x_train),
        "testing_rows": len(x_test),
        "training_class_counts": y_train.value_counts().to_dict(),
        "testing_class_counts": y_test.value_counts().to_dict(),
        "metrics": metrics,
        "per_class_metrics": per_class,
    }
    (output_folder / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(per_class).transpose().to_csv(
        output_folder / "per_class_metrics.csv"
    )
    return report
