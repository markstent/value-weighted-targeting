import unittest

import src.simulation_analysis as simulation_analysis


class SimulationAnalysisTests(unittest.TestCase):
    def test_strategy_list_includes_all_reported_baselines(self):
        self.assertEqual(
            simulation_analysis.STRATEGIES,
            [
                "Random",
                "Response Model",
                "S-Learner",
                "Uplift Only",
                "Uplift x LTV",
                "Full Framework",
            ],
        )

    def test_expected_segment_mean_ate_matches_documented_value(self):
        self.assertEqual(simulation_analysis.expected_segment_mean_ate_pp(), 6.7)

    def test_generate_population_has_required_columns(self):
        df = simulation_analysis.generate_population(200, seed=7)
        expected_columns = {
            "account_age",
            "weekly_spend",
            "login_freq",
            "bonus_redeem",
            "ltv",
            "churn_prob",
            "treatment",
            "y_obs",
            "segment",
            "true_tau",
            "y0",
            "y1",
            "responsible_gaming_flag",
        }
        self.assertTrue(expected_columns.issubset(df.columns))
        self.assertEqual(len(df), 200)


    def test_segment_prevalences_match_stated(self):
        """Issue #1: DGP must produce segment shares matching SEGMENTS config."""
        from src.simulation_analysis import generate_population, SEGMENTS
        counts = {}
        n_total = 0
        for seed in range(10):
            df = generate_population(10000, seed=seed)
            for seg in SEGMENTS:
                counts[seg] = counts.get(seg, 0) + (df['segment'] == seg).sum()
            n_total += len(df)
        for seg, (prevalence, _, _) in SEGMENTS.items():
            actual = counts[seg] / n_total
            self.assertAlmostEqual(actual, prevalence, delta=0.02,
                msg=f"{seg}: expected ~{prevalence}, got {actual:.3f}")


    def test_paired_permutation_test_respects_pairing(self):
        """Issue #2: Permutation test must be paired (sign-flip), not pooled."""
        from src.simulation_analysis import paired_permutation_test
        import numpy as np
        a = np.array([10, 12, 11, 13, 14])
        b = np.array([8, 9, 7, 10, 11])
        diff, p_val = paired_permutation_test(a, b, n_perms=5000)
        self.assertGreater(diff, 0)
        self.assertLess(p_val, 0.1)

    def test_paired_permutation_requires_equal_length(self):
        """Paired test requires matched observations."""
        from src.simulation_analysis import paired_permutation_test
        import numpy as np
        with self.assertRaises(ValueError):
            paired_permutation_test(np.array([1, 2, 3]), np.array([1, 2]))


if __name__ == "__main__":
    unittest.main()
