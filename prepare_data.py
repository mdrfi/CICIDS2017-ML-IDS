"""Prepare CIC-IDS2017 for machine learning.

Run from the project directory:

    python prepare_data.py
"""

from cicids_pipeline.preprocess import prepare_dataset


def main() -> None:
    rows = prepare_dataset()
    print("\nData preparation completed")
    print(f"Input rows:     {rows['input']:,}")
    print(f"Invalid rows:   {rows['invalid']:,}")
    print(f"Duplicate rows: {rows['duplicates']:,}")
    print(f"Training rows:  {rows['train']:,}")
    print(f"Testing rows:   {rows['test']:,}")
    print("Results: dataset/processed")


if __name__ == "__main__":
    main()
