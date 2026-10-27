"""Lock the RNG draw order of every DGP (values recorded from the original
monolithic generators, which the split generators reproduce bit-for-bit)."""

import numpy as np
import pytest

from core import dgp

EXPECTED = {
    'main': (5.335997775736868, 370013.7799934644, 301),
    'null': (19.932283248597937, 370013.7799934644, 301),
    'correlated': (10.166188765300934, 393033.06286155805, 313),
    'asym': (12.485980981122763, 370013.7799934644, 301),
    'harm': (5.335997775736868, 370013.7799934644, 331),
    'adv': (8.316289395662949, 370013.7799934644, 293),
    'corr_gamma': (4.801716681061246, 367466.472795284, 294),
    'heavytail': (4.801716681061246, 579756.8316242677, 294),
}


def test_every_dgp_is_registered():
    assert set(dgp.DGPS) == set(EXPECTED)


@pytest.mark.parametrize('name', sorted(EXPECTED))
def test_dgp_draws_are_unchanged(name):
    rng = np.random.RandomState(42)
    data = dgp.DGPS[name](500, rng)
    tau_sum, ltv_sum, y_sum = EXPECTED[name]
    assert data['true_tau'].sum() == pytest.approx(tau_sum, rel=1e-12)
    assert data['true_ltv'].sum() == pytest.approx(ltv_sum, rel=1e-12)
    assert int(data['y_obs'].sum()) == y_sum


def test_harm_dgp_exposes_true_harm():
    data = dgp.generate_harm(200, np.random.RandomState(1))
    assert data['true_harm'].min() >= 0.0 and data['true_harm'].max() <= 1.0
