"""Simple preprocessing for the CIC-IDS2017 CSV files."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


CSV_FOLDER = Path("dataset/CSVs/MachineLearningCSV")
OUTPUT_FOLDER = Path("dataset/processed")
LABEL_COLUMN = "Label"
RANDOM_STATE = 42


def load_csv_files(folder: Path = CSV_FOLDER) -> pd.DataFrame:
    """Read and combine all CIC-IDS2017 machine-learning CSV files."""
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    dataframes = []
    for file in tqdm(files, desc="Reading CSV files", unit="file"):
        dataframes.append(pd.read_csv(file, low_memory=False))

    return pd.concat(dataframes, ignore_index=True)


def clean_data(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove invalid and duplicate rows and create the binary label."""
    data = data.copy()
    input_rows = len(data)

    # The original CSV headers contain many unwanted spaces.
    data.columns = data.columns.str.strip()

    # Fix spaces and the damaged dash in web-attack labels.
    data[LABEL_COLUMN] = (
        data[LABEL_COLUMN]
        .astype("string")
        .str.strip()
        .str.replace("�", "-", regex=False)
        .str.replace(r"\s*-\s*", " - ", regex=True)
    )

    feature_columns = [column for column in data.columns if column != LABEL_COLUMN]
    for column in tqdm(feature_columns, desc="Converting features", unit="column"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    before_invalid = len(data)
    data.dropna(inplace=True)
    invalid_rows = before_invalid - len(data)

    # CIC-IDS2017 contains the same header-length column twice.
    duplicate_column = "Fwd Header Length.1"
    if duplicate_column in data.columns:
        if not data[duplicate_column].equals(data["Fwd Header Length"]):
            raise ValueError(f"{duplicate_column} is not an exact copy")
        data.drop(columns=duplicate_column, inplace=True)

    before_duplicates = len(data)
    data.drop_duplicates(inplace=True)
    duplicate_rows = before_duplicates - len(data)

    data.rename(columns={LABEL_COLUMN: "attack_label"}, inplace=True)
    data["is_attack"] = (data["attack_label"] != "BENIGN").astype("int8")

    # Float32 uses half the memory of float64 and is enough for ML models.
    feature_columns = [
        column for column in data.columns if column not in {"attack_label", "is_attack"}
    ]
    data[feature_columns] = data[feature_columns].astype("float32")

    report = {
        "input": input_rows,
        "invalid": invalid_rows,
        "duplicates": duplicate_rows,
        "retained": len(data),
    }
    return data.reset_index(drop=True), report


def split_and_scale(data: pd.DataFrame):
    """Create an 80/20 split and scale features using training data only."""
    feature_columns = [
        column for column in data.columns if column not in {"attack_label", "is_attack"}
    ]
    features = data[feature_columns]
    labels = data["is_attack"]
    attack_names = data["attack_label"]

    split = train_test_split(
        features,
        attack_names,
        labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    x_train, x_test, names_train, names_test, y_train, y_test = split

    scaler = StandardScaler()
    x_train = pd.DataFrame(
        scaler.fit_transform(x_train).astype("float32"),
        columns=feature_columns,
    )
    x_test = pd.DataFrame(
        scaler.transform(x_test).astype("float32"),
        columns=feature_columns,
    )

    train = x_train.assign(
        attack_label=names_train.reset_index(drop=True),
        is_attack=y_train.reset_index(drop=True),
    )
    test = x_test.assign(
        attack_label=names_test.reset_index(drop=True),
        is_attack=y_test.reset_index(drop=True),
    )
    return train, test, scaler, feature_columns


def prepare_dataset(
    csv_folder: Path = CSV_FOLDER,
    output_folder: Path = OUTPUT_FOLDER,
) -> dict[str, int]:
    """Run all preprocessing steps and save their results."""
    with tqdm(total=4, desc="Preparing data", unit="step") as progress:
        data = load_csv_files(csv_folder)
        progress.update()

        progress.set_description("Cleaning data")
        data, report = clean_data(data)
        progress.update()

        progress.set_description("Splitting and scaling")
        train, test, scaler, features = split_and_scale(data)
        progress.update()

        progress.set_description("Saving results")
        output_folder.mkdir(parents=True, exist_ok=True)
        train.to_parquet(output_folder / "train.parquet", index=False)
        test.to_parquet(output_folder / "test.parquet", index=False)
        joblib.dump(scaler, output_folder / "standard_scaler.joblib")
        progress.update()

    report["train"] = len(train)
    report["test"] = len(test)
    report["features"] = len(features)
    (output_folder / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report
