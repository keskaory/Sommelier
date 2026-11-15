# sanity checks for analysis/power.py: the simulation should behave like the stats say it must.
import numpy as np
import pytest

from analysis.power import MARGIN, chance_to_pass


def test_loss_at_the_margin_almost_never_passes():
    # if the true loss is exactly the margin, the upper bound is under it only ~2.5% of the time
    p = chance_to_pass(2000, MARGIN, 0.2, np.random.default_rng(1))
    assert p < 0.06


def test_no_loss_and_lots_of_data_almost_always_passes():
    p = chance_to_pass(5000, 0.0, 0.1, np.random.default_rng(1))
    assert p > 0.99


def test_more_data_helps():
    rng = np.random.default_rng(1)
    assert chance_to_pass(2000, 0.01, 0.2, rng) > chance_to_pass(300, 0.01, 0.2, rng)


def test_loss_bigger_than_flip_rate_is_rejected():
    with pytest.raises(AssertionError):
        chance_to_pass(500, 0.05, 0.02, np.random.default_rng(1))
