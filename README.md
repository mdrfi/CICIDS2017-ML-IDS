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

Open `notebooks/01_explore_data.ipynb` in VS Code, or start Jupyter with:

```powershell
uv run --extra notebook jupyter lab
```

The notebook shows the raw columns and rows, attack distribution, cleaning
results, processed dataset sizes, quality checks, and example feature plots.

## PCAP files

PCAP processing is separate from cleaning. CICFlowMeter first converts each
raw capture into a flow CSV; those flows must then be matched with official
labels before they can be used for supervised learning.
