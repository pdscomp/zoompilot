from openpilot.cereal import messaging
from openpilot.selfdrive.locationd.torqued import TorqueEstimator
from openpilot.sunnypilot.selfdrive.locationd.tests.speed_dep_helpers import make_cp
import pytest


@pytest.mark.parametrize("live", [False, True])
def test_lateral_delay_dispatch_uses_selected_delay(fake_params, monkeypatch, live):
  original_get_bool = fake_params.get_bool
  monkeypatch.setattr(
    fake_params, "get_bool",
    lambda key: live if key == "LagdToggle" else original_get_bool(key),
  )
  fake_params.store["LagdToggleDelay"] = 0.235
  fake_params.store["LagdValueCache"] = 9.9  # stale live cache is not fixed-delay authority
  cp = make_cp()
  cp.steerActuatorDelay = 0.1
  estimator = TorqueEstimator(cp)
  event = messaging.new_message("lateralDelay")
  event.lateralDelay.lateralDelay = 0.485

  estimator.handle_log(1.0, "lateralDelay", event.lateralDelay)

  expected = event.lateralDelay.lateralDelay if live else cp.steerActuatorDelay + 0.235
  assert estimator.lag == pytest.approx(expected)
