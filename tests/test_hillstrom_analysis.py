import unittest

import pandas as pd

import src.hillstrom_analysis as hillstrom_analysis


class HillstromAnalysisTests(unittest.TestCase):
    def test_prepare_data_creates_binary_treatment_and_features(self):
        df = pd.DataFrame(
            {
                "recency": [1, 2, 3],
                "history": [100.0, 200.0, 300.0],
                "mens": [1, 0, 0],
                "womens": [0, 1, 0],
                "channel": ["Phone", "Web", "Multichannel"],
                "newbie": [1, 0, 1],
                "segment": ["Mens E-Mail", "Womens E-Mail", "No E-Mail"],
                "visit": [1, 0, 1],
                "conversion": [0, 1, 0],
                "spend": [0.0, 25.0, 0.0],
            }
        )

        prepared, feature_cols = hillstrom_analysis.prepare_data(df)

        self.assertEqual(prepared["treatment"].tolist(), [1, 1, 0])
        self.assertIn("recency_log", prepared.columns)
        self.assertIn("channel_multi", prepared.columns)
        self.assertIn("is_newbie", prepared.columns)
        self.assertEqual(
            feature_cols,
            [
                "recency",
                "recency_log",
                "history",
                "is_mens",
                "is_womens",
                "channel_phone",
                "channel_web",
                "channel_multi",
                "is_newbie",
            ],
        )


    def test_hillstrom_checksum_matches_pinned(self):
        """Issue #8: Local CSV must match pinned checksum."""
        import os, hashlib
        from src.hillstrom_analysis import HILLSTROM_SHA256
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'hillstrom.csv')
        if not os.path.exists(csv_path):
            self.skipTest("Hillstrom CSV not downloaded yet")
        with open(csv_path, 'rb') as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(actual, HILLSTROM_SHA256,
            msg="Hillstrom CSV has changed since last pin")


if __name__ == "__main__":
    unittest.main()
