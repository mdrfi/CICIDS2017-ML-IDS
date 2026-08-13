"""Train Random Forest, Decision Tree, and KNN intrusion detectors."""

from cicids_pipeline.training import train_models


def main() -> None:
    results = train_models()
    print("\nModel training completed")
    print(results.round(4).to_string())
    print("\nResults: artifacts/models")


if __name__ == "__main__":
    main()
