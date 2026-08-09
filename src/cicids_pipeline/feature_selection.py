"""Feature selection for the cleaned CIC-IDS2017 training data."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRAIN_FILE = Path("dataset/processed/train.parquet")
OUTPUT_FOLDER = Path("artifacts/feature_selection")
SAMPLE_SIZE = 200_000
TOP_FEATURES = 20
RANDOM_STATE = 42


def load_training_sample(
    path: Path = TRAIN_FILE,
    sample_size: int = SAMPLE_SIZE,
) -> pd.DataFrame:
    """Load an exact random sample without reading the whole Parquet file into memory."""
    parquet = pq.ParquetFile(path)
    total_rows = parquet.metadata.num_rows
    sample_size = min(sample_size, total_rows)

    random = np.random.default_rng(RANDOM_STATE)
    selected_rows = np.sort(random.choice(total_rows, sample_size, replace=False))

    samples = []
    start = 0
    batches = parquet.iter_batches(batch_size=100_000)
    total_batches = math.ceil(total_rows / 100_000)

    for batch in tqdm(batches, total=total_batches, desc="Sampling training data"):
        end = start + len(batch)
        wanted = selected_rows[(selected_rows >= start) & (selected_rows < end)] - start
        if len(wanted):
            samples.append(batch.take(pa.array(wanted)).to_pandas())
        start = end

    return pd.concat(samples, ignore_index=True)


def normalize(scores: pd.Series) -> pd.Series:
    """Scale a score to the range 0–1."""
    minimum = scores.min()
    maximum = scores.max()
    if maximum == minimum:
        return pd.Series(0.0, index=scores.index)
    return (scores - minimum) / (maximum - minimum)


def calculate_feature_scores(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Rank features using correlation, mutual information, and Random Forest."""
    feature_columns = [
        column for column in data.columns if column not in {"attack_label", "is_attack"}
    ]
    constant_columns = [
        column for column in feature_columns if data[column].nunique(dropna=False) <= 1
    ]
    feature_columns = [column for column in feature_columns if column not in constant_columns]

    features = data[feature_columns]
    labels = data["is_attack"]

    correlation = features.corrwith(labels).abs()

    mutual_information = pd.Series(
        mutual_info_classif(features, labels, random_state=RANDOM_STATE),
        index=feature_columns,
    )

    forest = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight="balanced_subsample",
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    forest.fit(features, labels)
    forest_importance = pd.Series(forest.feature_importances_, index=feature_columns)

    scores = pd.DataFrame(
        {
            "correlation": normalize(correlation),
            "mutual_information": normalize(mutual_information),
            "random_forest": normalize(forest_importance),
        }
    )
    scores["combined_score"] = scores.mean(axis=1)
    scores.sort_values("combined_score", ascending=False, inplace=True)
    scores.insert(0, "rank", range(1, len(scores) + 1))
    scores.index.name = "feature"
    return scores, constant_columns


def save_plot(scores: pd.DataFrame, output_folder: Path) -> None:
    """Save a chart of the twenty highest-ranked features."""
    top = scores.head(TOP_FEATURES).sort_values("combined_score")
    top["combined_score"].plot(kind="barh", figsize=(10, 8), color="steelblue")
    plt.title("Top CIC-IDS2017 features")
    plt.xlabel("Combined importance score")
    plt.tight_layout()
    plt.savefig(output_folder / "top_features.png", dpi=150)
    plt.close()


def select_features(
    train_file: Path = TRAIN_FILE,
    output_folder: Path = OUTPUT_FOLDER,
    sample_size: int = SAMPLE_SIZE,
    top_count: int = TOP_FEATURES,
) -> dict:
    """Run feature selection and save the results."""
    data = load_training_sample(train_file, sample_size)
    scores, constant_columns = calculate_feature_scores(data)
    selected = scores.head(top_count).index.tolist()

    output_folder.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_folder / "feature_scores.csv")
    save_plot(scores, output_folder)

    report = {
        "training_file": str(train_file),
        "sample_rows": len(data),
        "candidate_features": len(scores),
        "constant_features": constant_columns,
        "selected_feature_count": len(selected),
        "selected_features": selected,
    }
    (output_folder / "selected_features.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report
