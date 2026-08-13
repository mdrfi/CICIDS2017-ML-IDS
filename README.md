# CICIDS2017-ML-IDS

Machine-learning network intrusion detection using CIC-IDS2017.

## Data folders

```text
dataset/
├── CSVs/
│   ├── MachineLearningCSV/       # Model-ready CSV inputs
│   └── GeneratedLabelledFlows/   # Flow identifiers and official labels
├── PCAPs/                        # Raw .pcap inputs (when available)
├── generated/                    # CSVs generated from PCAPs
└── processed/                    # Clean train/test datasets
```

The preprocessing script reads only the CSV files inside
`dataset/CSVs/MachineLearningCSV`. A filename ending in
`.pcap_ISCX.csv` is still a CSV and is never treated as a PCAP file.

## First step: prepare the CSV data

Create the environment and install the locked project dependencies with
[uv](https://docs.astral.sh/uv/):

```powershell
uv sync
```

To include the development or notebook dependencies, use `uv sync --extra dev`
or `uv sync --extra notebook`.

Run the readable entry script:

```powershell
uv run python prepare_data.py
```

It performs these operations in order:

1. Reads the eight CSV files in chunks.
2. Cleans spaces from column names.
3. Converts feature values to numbers.
4. Removes missing and infinite records.
5. Removes exact duplicate records.
6. Converts `BENIGN` to `0` and attacks to `1`.
7. Creates a stratified 80/20 train/test split.
8. Fits `StandardScaler` on training data only.
9. Saves Parquet data and a detailed JSON report in `dataset/processed`.

Progress bars show which stage is currently running.

## Explore the data

For a single notebook containing all preprocessing and feature-selection code,
open `notebooks/00_complete_pipeline.ipynb`.

Open `notebooks/01_explore_data.ipynb` in VS Code, or start Jupyter with:

```powershell
uv run --extra notebook jupyter lab
```

The notebook shows the raw columns and rows, attack distribution, cleaning
results, processed dataset sizes, quality checks, and example feature plots.

## Select useful features

Run correlation, mutual information, and Random Forest feature selection:

```powershell
uv run python select_features.py
```

The ranked scores, top-20 feature list, and plot are saved in
`artifacts/feature_selection`. Open `notebooks/02_feature_selection.ipynb` to
review and compare the three selection methods.

## Train the IDS models

Train Random Forest as the main model, with Decision Tree and K-Nearest
Neighbors (KNN) for comparison:

```powershell
uv run python train_models.py
```

Models, evaluation metrics, confusion matrices, and ROC curves are saved in
`artifacts/models`. The complete training code is also available in
`notebooks/03_train_models.ipynb`.

KNN uses a stratified 25,000-row sample of the training data and predicts in batches.
This keeps its memory use and running time reasonable on CICIDS2017.

## PCAP files

PCAP processing is separate from cleaning. CICFlowMeter first converts each
raw capture into a flow CSV; those flows must then be matched with official
labels before they can be used for supervised learning.
