import unittest

import numpy as np
import pandas as pd

from cicids_pipeline.feature_selection import calculate_feature_scores


class FeatureSelectionTests(unittest.TestCase):
    def test_signal_feature_is_ranked_above_noise(self) -> None:
        random = np.random.default_rng(42)
        labels = random.integers(0, 2, size=2_000)
        data = pd.DataFrame(
            {
                "strong_signal": labels + random.normal(0, 0.05, size=2_000),
                "noise": random.normal(size=2_000),
                "constant": 0,
                "attack_label": np.where(labels == 1, "Attack", "BENIGN"),
                "is_attack": labels,
            }
        )

        scores, constants = calculate_feature_scores(data)

        self.assertEqual(scores.index[0], "strong_signal")
        self.assertIn("constant", constants)


if __name__ == "__main__":
    unittest.main()
