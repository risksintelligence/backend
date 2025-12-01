import pytest

from app.ml.regime import _prepare_regime_features
from app.ml.forecast import _prepare_forecast_features
from app.ml.anomaly import _prepare_anomaly_features
from app.services.ingestion import Observation


def _mk_obs(values):
    return [Observation(series_id="test", observed_at=None, value=v) for v in values]


def test_prepare_regime_features_minimum():
    obs = {"s1": _mk_obs([1, 2, 3, 4, 5, 6, 7])}
    features = _prepare_regime_features(obs)
    assert features and len(features) == 5


def test_prepare_forecast_features_minimum():
    obs = {"s1": _mk_obs([1, 2, 3, 4, 5, 6, 7])}
    features = _prepare_forecast_features(obs)
    assert features and len(features) == 5


def test_prepare_anomaly_features_minimum():
    obs = {"s1": _mk_obs([1, 2, 3, 4, 5, 6, 7])}
    features = _prepare_anomaly_features(obs)
    assert features and len(features) == 4
