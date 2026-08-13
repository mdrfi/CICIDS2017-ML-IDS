import unittest
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from cicids_pipeline.training import create_models, evaluate_model


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        x, y = make_classification(
            n_samples=200,
            n_features=8,
            n_informative=6,
            n_redundant=0,
            random_state=42,
        )
        columns = [f"feature_{number}" for number in range(x.shape[1])]
        cls.x_train = pd.DataFrame(x[:150], columns=columns)
        cls.y_train = pd.Series(y[:150])
        cls.x_test = pd.DataFrame(x[150:], columns=columns)
        cls.y_test = pd.Series(y[150:])

    @staticmethod
    def create_test_models():
        models = create_models()
        models["Random Forest"].set_params(n_jobs=1)
        models["K-Nearest Neighbors"].set_params(
            kneighborsclassifier__n_jobs=1
        )
        return models

    def test_random_forest_is_the_main_model(self) -> None:
        models = create_models()

        self.assertEqual(list(models)[0], "Random Forest")
        self.assertIsInstance(models["Random Forest"], RandomForestClassifier)
        self.assertIsInstance(models["Decision Tree"], DecisionTreeClassifier)
        knn = models["K-Nearest Neighbors"]
        self.assertIsInstance(knn, Pipeline)
        self.assertIsInstance(knn.named_steps["standardscaler"], StandardScaler)
        self.assertIsInstance(knn.named_steps["kneighborsclassifier"], KNeighborsClassifier)

    def test_all_models_can_be_trained_and_make_predictions(self) -> None:
        for name, model in self.create_test_models().items():
            with self.subTest(model=name):
                model.fit(self.x_train, self.y_train)
                check_is_fitted(model)

                predictions = model.predict(self.x_test)
                probabilities = model.predict_proba(self.x_test)

                self.assertEqual(len(predictions), len(self.x_test))
                self.assertEqual(probabilities.shape, (len(self.x_test), 2))
                self.assertTrue(set(predictions).issubset({0, 1}))
                np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_evaluation_returns_valid_metrics(self) -> None:
        model = self.create_test_models()["K-Nearest Neighbors"]
        model.fit(self.x_train, self.y_train)

        metrics, predictions, probabilities = evaluate_model(
            model, self.x_test, self.y_test
        )

        for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            self.assertGreaterEqual(metrics[metric], 0.0)
            self.assertLessEqual(metrics[metric], 1.0)
        self.assertGreaterEqual(metrics["prediction_seconds"], 0.0)
        self.assertEqual(len(predictions), len(self.x_test))
        self.assertEqual(len(probabilities), len(self.x_test))

    def test_saved_model_keeps_the_same_predictions(self) -> None:
        model = self.create_test_models()["K-Nearest Neighbors"]
        model.fit(self.x_train, self.y_train)
        expected = model.predict(self.x_test)

        model_file = BytesIO()
        joblib.dump(model, model_file)
        model_file.seek(0)
        loaded_model = joblib.load(model_file)

        np.testing.assert_array_equal(loaded_model.predict(self.x_test), expected)


if __name__ == "__main__":
    unittest.main()
