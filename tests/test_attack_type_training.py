import unittest

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.utils.validation import check_is_fitted

from cicids_pipeline.attack_type_training import (
    create_attack_type_model,
    evaluate_attack_type_model,
)


class AttackTypeTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        x, y = make_classification(
            n_samples=240,
            n_features=8,
            n_informative=6,
            n_redundant=0,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42,
        )
        names = pd.Series(y).map({0: "DDoS", 1: "PortScan", 2: "Bot"})
        cls.x_train = pd.DataFrame(x[:180])
        cls.y_train = names[:180]
        cls.x_test = pd.DataFrame(x[180:])
        cls.y_test = names[180:]

    def test_model_learns_all_attack_types(self) -> None:
        model = create_attack_type_model(n_jobs=1)
        model.fit(self.x_train, self.y_train)

        check_is_fitted(model)
        self.assertEqual(set(model.classes_), {"DDoS", "PortScan", "Bot"})
        self.assertEqual(len(model.predict(self.x_test)), len(self.x_test))

    def test_multiclass_metrics_are_valid(self) -> None:
        model = create_attack_type_model(n_jobs=1)
        model.fit(self.x_train, self.y_train)
        metrics, predictions, per_class = evaluate_attack_type_model(
            model, self.x_test, self.y_test
        )

        for name in [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "f1_weighted",
        ]:
            self.assertGreaterEqual(metrics[name], 0.0)
            self.assertLessEqual(metrics[name], 1.0)

        self.assertEqual(len(predictions), len(self.x_test))
        self.assertTrue({"DDoS", "PortScan", "Bot"}.issubset(per_class))


if __name__ == "__main__":
    unittest.main()
