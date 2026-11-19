"""Stacking folds must not split one date across train and validation."""
import os
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "predictions"))

# model_arena imports xgb_model, which exits when xgboost isn't installed.
# Only these helpers are needed here, so stand in a light fake for CI.
if "xgb_model" not in sys.modules:
    import types
    fake = types.ModuleType("xgb_model")
    fake.engineer_features = lambda records: (None, None, None)
    fake.collect_all_training_data = lambda *a, **k: []
    fake.FEATURE_COLS = []
    fake._compute_auc = lambda y, p: 0.5
    fake._compute_logloss = lambda y, p: 0.0
    fake._safe_float = float
    sys.modules["xgb_model"] = fake

import model_arena  # noqa: E402

LOG = []


class SpyModel(ClassifierMixin, BaseEstimator):
    """Records which dates (column 0) it was fit on and predicted for."""

    def fit(self, X, y, **kw):
        self.fit_dates_ = set(X[:, 0].tolist())
        return self

    def predict_proba(self, X):
        LOG.append((self.fit_dates_, set(X[:, 0].tolist())))
        p = np.full(len(X), 0.5)
        return np.column_stack([1 - p, p])


def _results():
    return [{"name": f"spy{i}", "model": SpyModel(), "brier": 0.25, "auc": 0.5,
             "accuracy": 0.5, "cal_error": 0.0} for i in range(2)]


def test_grouped_folds_keep_dates_together():
    LOG.clear()
    rng = np.random.default_rng(0)
    dates = np.repeat(np.arange(10), 20)           # 10 dates, 20 props each
    X_train = np.column_stack([dates, rng.normal(size=len(dates))])
    y_train = rng.integers(0, 2, len(dates))
    X_test = np.column_stack([np.full(20, 99), rng.normal(size=20)])
    y_test = rng.integers(0, 2, 20)

    model_arena.build_ensemble(_results(), X_train, y_train, X_test, y_test,
                               groups_train=dates.astype(str))

    oof_calls = [c for c in LOG if 99 not in c[1]]
    assert oof_calls, "expected out-of-fold predictions"
    for fit_dates, val_dates in oof_calls:
        assert fit_dates.isdisjoint(val_dates)
