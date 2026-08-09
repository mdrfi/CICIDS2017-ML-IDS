"""Select the most useful CIC-IDS2017 features."""

from cicids_pipeline.feature_selection import select_features


def main() -> None:
    report = select_features()

    print("\nFeature selection completed")
    print(f"Sample rows: {report['sample_rows']:,}")
    print(f"Selected features: {report['selected_feature_count']}")
    for number, feature in enumerate(report["selected_features"], start=1):
        print(f"{number:2}. {feature}")
    print("Results: artifacts/feature_selection")


if __name__ == "__main__":
    main()

