"""Train the multiclass CIC-IDS2017 attack-type model."""

from cicids_pipeline.attack_type_training import train_attack_type_model


def main() -> None:
    report = train_attack_type_model()
    metrics = report["metrics"]

    print("\nAttack-type model completed")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print("Results: artifacts/attack_type_model")


if __name__ == "__main__":
    main()
