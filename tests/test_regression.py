"""Regression tests that pin core numerical properties of the simulation.
These will break if the DGP or analysis code changes in ways that alter
the paper's reported results. That breakage is intentional.
"""
import unittest
import numpy as np
from src.simulation_analysis import (
    generate_population, SEGMENTS, N_PLAYERS,
    train_t_learner, score_uplift, evaluate_strategy,
    BUDGET_FRACTION,
)


class RegressionTests(unittest.TestCase):
    """Pin key properties so code changes that alter results are caught."""

    @classmethod
    def setUpClass(cls):
        cls.df = generate_population(N_PLAYERS, seed=13)

    def test_population_size(self):
        self.assertEqual(len(self.df), N_PLAYERS)

    def test_treatment_fraction_near_half(self):
        frac = self.df['treatment'].mean()
        self.assertAlmostEqual(frac, 0.50, delta=0.03)

    def test_segment_distribution_matches_config(self):
        for seg, (prev, _, _) in SEGMENTS.items():
            actual = (self.df['segment'] == seg).mean()
            self.assertAlmostEqual(actual, prev, delta=0.03,
                msg=f"{seg}: expected ~{prev}, got {actual:.3f}")

    def test_sleeping_dogs_have_negative_mean_tau(self):
        sd = self.df[self.df['segment'] == 'Sleeping Dogs']
        self.assertLess(sd['true_tau'].mean(), 0,
            msg="Sleeping Dogs should have negative treatment effect")

    def test_persuadables_have_largest_positive_tau(self):
        means = {}
        for seg in SEGMENTS:
            means[seg] = self.df[self.df['segment'] == seg]['true_tau'].mean()
        self.assertEqual(max(means, key=means.get), 'Persuadables')

    def test_rg_flag_excludes_roughly_3pct(self):
        frac = self.df['responsible_gaming_flag'].mean()
        self.assertGreater(frac, 0.005)
        self.assertLess(frac, 0.10)

    def test_uplift_targeting_beats_random_single_run(self):
        """The T-Learner should beat random on a single seed."""
        from sklearn.model_selection import train_test_split
        df_train, df_test = train_test_split(self.df, test_size=0.30, random_state=13)
        model_t, model_c, features = train_t_learner(df_train)
        uplift_scores = score_uplift(model_t, model_c, df_test, features)
        rng = np.random.RandomState(13)
        random_scores = rng.uniform(0, 1, len(df_test))
        r_uplift = evaluate_strategy(df_test, uplift_scores, BUDGET_FRACTION, 'Uplift')
        r_random = evaluate_strategy(df_test, random_scores, BUDGET_FRACTION, 'Random')
        self.assertGreater(r_uplift['ate_pp'], r_random['ate_pp'])


if __name__ == '__main__':
    unittest.main()
