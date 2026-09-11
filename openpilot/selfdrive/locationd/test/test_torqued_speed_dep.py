"""Tests for speed-binned learning in torqued (vehicle-agnostic).

Uses get_speed_dep_config() to discover configured cars.
All tests are driven by config, not hardcoded fingerprints.
"""
import numpy as np
import pytest

from unittest.mock import MagicMock, patch  # noqa: TID251
from opendbc.sunnypilot.car.interfaces import get_speed_dep_config
from openpilot.cereal import log
from openpilot.selfdrive.locationd.torqued import (
  TorqueEstimator, TorqueBuckets, VERSION, MIN_FILTER_DECAY,
)
from openpilot.sunnypilot.selfdrive.locationd.torqued_ext import (
  DEFAULT_SPEED_BIN_BOUNDS as SPEED_BIN_BOUNDS, DEFAULT_SPEED_BIN_CENTERS as SPEED_BIN_CENTERS,
  LIVE_TORQUE_PARAMETERS_SP_KEY, LIVE_TORQUE_PARAMETERS_SP_SERVICE, TorqueEstimatorExt,
)
from openpilot.sunnypilot.selfdrive.locationd.tests.speed_dep_helpers import FakePubMaster, in_bounds_values
from openpilot.sunnypilot.selfdrive.locationd.tests.test_torqued_cache_restore import _cache, _sp

# Discover configured cars
SPEED_DEP_CARS = get_speed_dep_config()
SPEED_DEP_FINGERPRINT = next(iter(SPEED_DEP_CARS)) if SPEED_DEP_CARS else None

# Sentinel fingerprint that must not appear in speed_dependent.toml
NON_SPEED_DEP_FINGERPRINT = 'NOT_IN_SPEED_DEP_TOML'
assert NON_SPEED_DEP_FINGERPRINT not in SPEED_DEP_CARS, f"{NON_SPEED_DEP_FINGERPRINT} unexpectedly in speed_dependent.toml"


def _get_car_bins(fingerprint):
  """Get actual bin centers and bounds for a configured car (or defaults for unconfigured)."""
  cfg = SPEED_DEP_CARS.get(fingerprint, {})
  if 'speed_bp' in cfg:
    centers = list(cfg['speed_bp'])
    bounds = TorqueEstimatorExt._centers_to_bounds(centers)
  else:
    centers = list(SPEED_BIN_CENTERS)
    bounds = list(SPEED_BIN_BOUNDS)
  return centers, bounds

# Both Params locations need mocking: torqued.py (cache) and torqued_ext.py (toggles)
PATCH_PARAMS = 'openpilot.selfdrive.locationd.torqued.Params'
PATCH_EXT_PARAMS = 'openpilot.sunnypilot.selfdrive.locationd.torqued_ext.Params'


def _setup_ext_mock(mock_ext_params_cls, speed_dep_on):
  """Configure the torqued_ext Params mock for toggle state.

  Speed-dep learning requires the full activation chain the UI enforces before the toggle
  is even reachable (Enforce Torque Control + Self-Tune), so enable those alongside it —
  otherwise speed_binned stays False and the learner (correctly) never runs."""
  def _get_bool(param):
    if param in ("SpeedDependentTorqueToggle", "EnforceTorqueControl", "LiveTorqueParamsToggle"):
      return speed_dep_on
    return False
  mock_ext_params_cls.return_value.get_bool.side_effect = _get_bool
  mock_ext_params_cls.return_value.get.return_value = None


def make_mock_CP(fingerprint=None, lat_accel_factor=1.25, friction=0.125):
  if fingerprint is None:
    fingerprint = SPEED_DEP_FINGERPRINT
  CP = MagicMock()
  CP.brand = 'test'
  CP.carFingerprint = fingerprint
  CP.minSteerSpeed = 0.0  # steer-to-zero EPS: entries flagged requires_steer_to_zero stay valid
  CP.lateralTuning.which.return_value = 'torque'
  CP.lateralTuning.torque.friction = friction
  CP.lateralTuning.torque.latAccelFactor = lat_accel_factor
  return CP


class TestSpeedDepConfig:
  """Config-level tests that don't need a TorqueEstimator."""

  def test_speed_dep_config_has_entries(self):
    assert len(SPEED_DEP_CARS) > 0

  def test_version_exists(self):
    assert VERSION >= 1

  def test_speed_bin_bounds_cover_full_range(self):
    all_bounds = [b for bounds in SPEED_BIN_BOUNDS for b in bounds]
    assert min(all_bounds) == 5
    assert max(all_bounds) >= 35

  def test_speed_bin_centers_match_bounds(self):
    for center, (lo, hi) in zip(SPEED_BIN_CENTERS, SPEED_BIN_BOUNDS, strict=True):
      assert center >= lo
      assert center <= hi


@pytest.mark.skipif(SPEED_DEP_FINGERPRINT is None, reason="No cars in speed_dependent.toml")
class TestSpeedBinnedLearning:
  """Test speed-binned learning with toggle ON."""

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_speed_bins_initialized(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    for fingerprint in SPEED_DEP_CARS:
      centers, bounds = _get_car_bins(fingerprint)
      est = TorqueEstimator(make_mock_CP(fingerprint=fingerprint))
      assert est.speed_binned
      assert len(est.speed_bin_points) == len(bounds)

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_speed_bin_routing(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    centers, bounds = _get_car_bins(SPEED_DEP_FINGERPRINT)
    for bin_idx, (lo, hi) in enumerate(bounds):
      est = TorqueEstimator(make_mock_CP())
      vego = (lo + hi) / 2.0
      est._on_torque_point(0.1, 0.3, vego)
      assert len(est.speed_bin_points[bin_idx]) == 1, \
        f"bin {bin_idx} ({lo}-{hi} m/s) should have 1 point at vego={vego}"
      for j in range(len(bounds)):
        if j != bin_idx:
          assert len(est.speed_bin_points[j]) == 0, \
            f"bin {j} should be empty when vego={vego}"

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_cereal_message_fields(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    for fingerprint in SPEED_DEP_CARS:
      centers, bounds = _get_car_bins(fingerprint)
      est = TorqueEstimator(make_mock_CP(fingerprint=fingerprint))
      est._pm = FakePubMaster()
      est.get_msg()
      ltp = est._pm.last()
      assert len(ltp.speedBinCenters) == len(centers)
      assert len(ltp.speedBinLatAccelFactors) == len(bounds)
      assert len(ltp.speedBinFrictions) == len(bounds)
      assert len(ltp.speedBinValid) == len(bounds)

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_global_fit_unchanged(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP(lat_accel_factor=1.25, friction=0.125))
    msg = est.get_msg()
    ltp = msg.lateralTorqueParameters
    assert ltp.latAccelFactorFiltered == pytest.approx(1.25, abs=1e-2)
    assert ltp.frictionCoefficientFiltered == pytest.approx(0.125, abs=1e-3)

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_global_buckets_still_require_min_vel(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP())
    assert len(est.filtered_points) == 0


class TestToggleGate:
  """Toggle OFF should disable speed-binning even for configured cars."""

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_toggle_off_no_speed_bins(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=False)
    if SPEED_DEP_FINGERPRINT:
      est = TorqueEstimator(make_mock_CP(fingerprint=SPEED_DEP_FINGERPRINT))
      assert not est.speed_binned


class TestBackwardCompatibility:
  """Cars with toggle OFF should be unaffected."""

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_unconfigured_car_no_speed_bins(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=False)
    est = TorqueEstimator(make_mock_CP(fingerprint=NON_SPEED_DEP_FINGERPRINT))
    assert not est.speed_binned

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_unconfigured_car_no_speed_bin_fields(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=False)
    est = TorqueEstimator(make_mock_CP(fingerprint=NON_SPEED_DEP_FINGERPRINT))
    msg = est.get_msg()
    ltp = msg.lateralTorqueParameters
    assert len(ltp.speedBinCenters) == 0
    assert len(ltp.speedBinLatAccelFactors) == 0
    assert len(ltp.speedBinFrictions) == 0
    assert len(ltp.speedBinValid) == 0

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_unconfigured_car_global_params_still_work(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=False)
    est = TorqueEstimator(make_mock_CP(fingerprint=NON_SPEED_DEP_FINGERPRINT, lat_accel_factor=2.0, friction=0.15))
    msg = est.get_msg()
    ltp = msg.lateralTorqueParameters
    assert ltp.latAccelFactorFiltered == pytest.approx(2.0, abs=1e-2)
    assert ltp.frictionCoefficientFiltered == pytest.approx(0.15, abs=1e-3)
    assert not est.speed_binned

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_unconfigured_car_no_speed_bin_attributes(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=False)
    est = TorqueEstimator(make_mock_CP(fingerprint=NON_SPEED_DEP_FINGERPRINT))
    assert not hasattr(est, 'speed_bin_points')
    assert not hasattr(est, 'speed_bin_filtered')

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_cal_percent_works_for_both(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    fingerprints = [NON_SPEED_DEP_FINGERPRINT]
    if SPEED_DEP_FINGERPRINT:
      fingerprints.append(SPEED_DEP_FINGERPRINT)
    for fp in fingerprints:
      est = TorqueEstimator(make_mock_CP(fingerprint=fp))
      msg = est.get_msg()
      assert msg.lateralTorqueParameters.calPerc == 0


class TestCentersToBounds:
  """Tests for _centers_to_bounds static method."""

  def test_midpoints_between_centers(self):
    bounds = TorqueEstimatorExt._centers_to_bounds([10.0, 20.0, 30.0])
    assert bounds[0] == (5, 15.0)   # lo=DEFAULT[0][0], hi=midpoint(10,20)
    assert bounds[1] == (15.0, 25.0)
    assert bounds[2] == (25.0, 40)  # hi=DEFAULT[-1][1]

  def test_single_center(self):
    bounds = TorqueEstimatorExt._centers_to_bounds([20.0])
    assert bounds == [(5, 40)]

  def test_edges_use_default_bounds(self):
    bounds = TorqueEstimatorExt._centers_to_bounds([7.0, 35.0])
    assert bounds[0][0] == 5    # DEFAULT_SPEED_BIN_BOUNDS[0][0]
    assert bounds[-1][1] == 40  # DEFAULT_SPEED_BIN_BOUNDS[-1][1]
    assert bounds[0][1] == pytest.approx((7.0 + 35.0) / 2)
    assert bounds[1][0] == pytest.approx((7.0 + 35.0) / 2)

  def test_contiguous_coverage(self):
    """Each bin's upper bound must equal the next bin's lower bound."""
    centers = [8.0, 15.0, 22.0, 30.0]
    bounds = TorqueEstimatorExt._centers_to_bounds(centers)
    for i in range(len(bounds) - 1):
      assert bounds[i][1] == pytest.approx(bounds[i + 1][0])


@pytest.mark.skipif(SPEED_DEP_FINGERPRINT is None, reason="No cars in speed_dependent.toml")
class TestCacheRestore:
  """Retained restore coverage using the current global/SP cache contract."""

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_successful_restore_updates_filters(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP())
    lafs, frictions = in_bounds_values(est)
    est._restore_ext_cache(_cache(), cache_CP=est.CP, cache_sp=_sp(est, lafs, frictions))
    for i in range(len(est.speed_bin_bounds)):
      assert est.speed_bin_filtered[i]['latAccelFactor'].x == lafs[i]
      assert est.speed_bin_filtered[i]['frictionCoefficient'].x == frictions[i]

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_mismatched_laf_length_rejected(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP())
    original_laf = est.speed_bin_filtered[0]['latAccelFactor'].x
    lafs, frictions = in_bounds_values(est)
    est._restore_ext_cache(_cache(), cache_CP=est.CP, cache_sp=_sp(est, lafs[:1], frictions))
    assert est.speed_bin_filtered[0]['latAccelFactor'].x == original_laf

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_mismatched_friction_length_rejected(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP())
    original_laf = est.speed_bin_filtered[0]['latAccelFactor'].x
    lafs, frictions = in_bounds_values(est)
    est._restore_ext_cache(_cache(), cache_CP=est.CP, cache_sp=_sp(est, lafs, frictions[:1]))
    assert est.speed_bin_filtered[0]['latAccelFactor'].x == original_laf

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_missing_points_still_restores_filters(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP())
    lafs, frictions = in_bounds_values(est)
    est._restore_ext_cache(_cache(), cache_CP=est.CP, cache_sp=_sp(est, lafs, frictions))
    for i in range(len(est.speed_bin_bounds)):
      assert est.speed_bin_filtered[i]['latAccelFactor'].x == lafs[i]
      assert est.speed_bin_filtered[i]['frictionCoefficient'].x == frictions[i]
    assert all(len(b) == 0 for b in est.speed_bin_points)

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_cache_decay_used_without_estimator_decay(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP())
    saved_decay = est.decay
    del est.decay
    try:
      lafs, frictions = in_bounds_values(est)
      est._restore_ext_cache(_cache(decay=MIN_FILTER_DECAY), cache_CP=est.CP, cache_sp=_sp(est, lafs, frictions))
      assert est.speed_bin_filtered[0]['latAccelFactor'].x == lafs[0]
      assert all(d == MIN_FILTER_DECAY for d in est.speed_bin_decays)
    finally:
      est.decay = saved_decay

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_cached_decay_overrides_estimator_decay(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP())
    est.decay = 200
    est._restore_ext_cache(_cache(decay=120.0), cache_CP=est.CP, cache_sp=_sp(est, *in_bounds_values(est)))
    assert all(d == 120.0 for d in est.speed_bin_decays)


@pytest.mark.skipif(SPEED_DEP_FINGERPRINT is None, reason="No cars in speed_dependent.toml")
class TestNaNHandling:
  """Tests for bin behavior when SVD fails."""

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_svd_failure_returns_false(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP())

    target_bin = 1
    mock_bucket = MagicMock()
    mock_bucket.is_calculable.return_value = True
    mock_bucket.is_valid.return_value = False
    mock_bucket.get_points.return_value = np.zeros((10, 3))
    est.speed_bin_points[target_bin] = mock_bucket
    est._speed_bin_last_len[target_bin] = -1  # force recalculation

    with patch('numpy.linalg.svd', side_effect=np.linalg.LinAlgError):
      results = est._estimate_params_speed_binned()

    bin_results = {idx: valid for idx, valid in results}
    assert bin_results[target_bin] is False

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_valid_bin_svd_failure_resets_bin(self, mock_params_cls, mock_ext):
    """A bin with enough data that produces NaN/error should be reset."""
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP())

    target_bin = 1
    mock_bucket = MagicMock()
    mock_bucket.is_calculable.return_value = True
    mock_bucket.is_valid.return_value = True  # enough data → triggers reset
    mock_bucket.get_points.return_value = np.zeros((10, 3))
    est.speed_bin_points[target_bin] = mock_bucket
    est._speed_bin_last_len[target_bin] = -1  # force recalculation

    with patch('numpy.linalg.svd', side_effect=np.linalg.LinAlgError):
      est._estimate_params_speed_binned()

    assert est.speed_bin_points[target_bin] is not mock_bucket
    assert isinstance(est.speed_bin_points[target_bin], TorqueBuckets)
    assert est.speed_bin_decays[target_bin] == MIN_FILTER_DECAY

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_non_valid_bin_svd_failure_preserves_bin(self, mock_params_cls, mock_ext):
    """A bin that is calculable but not valid should NOT be reset on SVD failure."""
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP())

    target_bin = 1
    mock_bucket = MagicMock()
    mock_bucket.is_calculable.return_value = True
    mock_bucket.is_valid.return_value = False  # not enough data → no reset
    mock_bucket.get_points.return_value = np.zeros((10, 3))
    est.speed_bin_points[target_bin] = mock_bucket
    est._speed_bin_last_len[target_bin] = -1  # force recalculation

    with patch('numpy.linalg.svd', side_effect=np.linalg.LinAlgError):
      est._estimate_params_speed_binned()

    assert est.speed_bin_points[target_bin] is mock_bucket  # preserved


@pytest.mark.skipif(SPEED_DEP_FINGERPRINT is None, reason="No cars in speed_dependent.toml")
class TestGetMsgWithPoints:
  """get_msg(with_points=True) should populate speedBinPoints."""

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_speed_bin_points_populated(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    centers, bounds = _get_car_bins(SPEED_DEP_FINGERPRINT)
    est = TorqueEstimator(make_mock_CP())
    # Pick speeds inside two different bins
    v1 = (bounds[0][0] + bounds[0][1]) / 2
    v2 = (bounds[-1][0] + bounds[-1][1]) / 2
    est._on_torque_point(0.1, 0.3, v1)
    est._on_torque_point(0.2, 0.4, v2)

    est._pm = FakePubMaster()
    est.get_msg(with_points=True)
    key, cache_bytes = mock_ext.return_value.put.call_args.args
    assert key == LIVE_TORQUE_PARAMETERS_SP_KEY
    with log.Event.from_bytes(cache_bytes) as evt:
      sp = getattr(evt, LIVE_TORQUE_PARAMETERS_SP_SERVICE)
      assert len(sp.speedBinPoints) == len(bounds)
      assert sum(len(bin_pts) for bin_pts in sp.speedBinPoints) >= 2
    assert len(est._pm.last().speedBinPoints) == 0

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_speed_bin_points_empty_without_flag(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    centers, bounds = _get_car_bins(SPEED_DEP_FINGERPRINT)
    est = TorqueEstimator(make_mock_CP())
    v1 = (bounds[0][0] + bounds[0][1]) / 2
    est._on_torque_point(0.1, 0.3, v1)

    est._pm = FakePubMaster()
    est.get_msg(with_points=False)
    assert len(est._pm.last().speedBinPoints) == 0
    mock_ext.return_value.put.assert_not_called()


class TestUnconfiguredCarToggleOn:
  """Unconfigured car with speed-dep ON should use default bins and offline seeds."""

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_default_bins_created(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP(fingerprint=NON_SPEED_DEP_FINGERPRINT))
    assert est.speed_binned
    est._on_torque_point(0.1, 0.3, 10.0)
    assert len(est.speed_bin_bounds) == len(SPEED_BIN_BOUNDS)
    assert est.speed_bin_centers == list(SPEED_BIN_CENTERS)

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_seeded_with_offline_values(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP(fingerprint=NON_SPEED_DEP_FINGERPRINT,
                                       lat_accel_factor=2.5, friction=0.18))
    est._on_torque_point(0.1, 0.3, 10.0)
    for i in range(len(SPEED_BIN_BOUNDS)):
      assert est.speed_bin_filtered[i]['latAccelFactor'].x == pytest.approx(2.5)
      assert est.speed_bin_filtered[i]['frictionCoefficient'].x == pytest.approx(0.18)


class TestOnTorquePointWhenOff:
  """_on_torque_point should be a no-op when speed_binned is False."""

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_no_bins_created_when_off(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=False)
    est = TorqueEstimator(make_mock_CP(fingerprint=NON_SPEED_DEP_FINGERPRINT))
    est._on_torque_point(0.1, 0.3, 10.0)
    assert not hasattr(est, 'speed_bin_points')


@pytest.mark.skipif(SPEED_DEP_FINGERPRINT is None, reason="No cars in speed_dependent.toml")
class TestSpeedBinInitIdempotency:
  """Lazy speed-bin init (triggered by _on_torque_point) must not re-init on later points."""

  @patch(PATCH_EXT_PARAMS)
  @patch(PATCH_PARAMS)
  def test_second_call_preserves_points(self, mock_params_cls, mock_ext):
    mock_params_cls.return_value.get.return_value = None
    _setup_ext_mock(mock_ext, speed_dep_on=True)
    est = TorqueEstimator(make_mock_CP())

    # Pick the midpoint of the first bin
    centers, bounds = _get_car_bins(SPEED_DEP_FINGERPRINT)
    vego = (bounds[0][0] + bounds[0][1]) / 2
    est._on_torque_point(0.1, 0.3, vego)
    assert len(est.speed_bin_points[0]) == 1

    est._on_torque_point(0.2, 0.4, vego)
    # Should now have 2 points (not re-initialized)
    assert len(est.speed_bin_points[0]) == 2
