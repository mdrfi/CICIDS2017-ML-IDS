import unittest

import numpy as np
import pandas as pd

from cicids_pipeline.preprocess import clean_data


class CleanChunkTests(unittest.TestCase):
    def test_cleans_headers_values_and_labels(self) -> None:
        raw = pd.DataFrame(
            {
                " Feature A ": [1, 2, np.inf],
                " Fwd Header Length ": [20, 40, 60],
                " Fwd Header Length.1 ": [20, 40, 60],
                " Label ": [" BENIGN ", "Web Attack � XSS", "BENIGN"],
            }
        )

        cleaned, stats = clean_data(raw)

        self.assertEqual(len(cleaned), 2)
        self.assertNotIn("Fwd Header Length.1", cleaned.columns)
        self.assertEqual(cleaned["attack_label"].tolist(), ["BENIGN", "Web Attack - XSS"])
        self.assertEqual(cleaned["is_attack"].tolist(), [0, 1])
        self.assertEqual(stats["invalid"], 1)

    def test_rejects_a_nonidentical_redundant_column(self) -> None:
        raw = pd.DataFrame(
            {
                "Fwd Header Length": [20],
                "Fwd Header Length.1": [40],
                "Label": ["BENIGN"],
            }
        )

        with self.assertRaises(ValueError):
            clean_data(raw)


if __name__ == "__main__":
    unittest.main()
