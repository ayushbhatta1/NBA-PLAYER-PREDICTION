"""The calibrated XGBoost probability must win over the raw one."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'predictions'))

import ev_optimizer  # noqa: E402
import parlay_optimizer  # noqa: E402


def test_ev_optimizer_prefers_calibrated_over_raw():
    prop = {'xgb_prob': 0.71, 'xgb_prob_calibrated': 0.58}
    assert ev_optimizer._get_prop_prob(prop) == 0.58


def test_ev_optimizer_ensemble_still_first_and_none_skipped():
    assert ev_optimizer._get_prop_prob(
        {'ensemble_prob': 0.6, 'xgb_prob_calibrated': 0.55}) == 0.6
    assert ev_optimizer._get_prop_prob(
        {'ensemble_prob': None, 'xgb_prob': 0.62}) == 0.62
    assert ev_optimizer._get_prop_prob({}) == 0.50


def test_parlay_optimizer_prefers_calibrated_over_raw():
    pick = {'xgb_prob': 0.71, 'xgb_prob_calibrated': 0.58}
    assert parlay_optimizer._pick_prob(pick) == 0.58
    assert parlay_optimizer._pick_prob({'ensemble_prob': None}) == 0.50


def test_kelly_and_ev_basics():
    assert abs(ev_optimizer.compute_ev(0.60, 1.91) - 0.146) < 1e-9
    assert ev_optimizer.kelly_stake(0.40, 1.91) == 0.0
    assert ev_optimizer.kelly_stake(0.99, 5.0) == ev_optimizer.MAX_KELLY_CAP
