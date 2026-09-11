import pytest

from openpilot.cereal import log, messaging
from openpilot.selfdrive.locationd.torqued import VERSION
from openpilot.selfdrive.test.process_replay.process_replay import get_custom_params_from_lr, get_process_config
from openpilot.sunnypilot.selfdrive.locationd.torqued_ext import (
  LIVE_TORQUE_PARAMETERS_SP_KEY, LIVE_TORQUE_PARAMETERS_SP_SERVICE,
)


def test_replay_wires_the_live_torque_sp_channel():
  assert LIVE_TORQUE_PARAMETERS_SP_SERVICE in get_process_config("controlsd").pubs
  assert LIVE_TORQUE_PARAMETERS_SP_SERVICE in get_process_config("torqued").subs


@pytest.mark.parametrize("initial_state, expected_seed", [("first", 4), ("last", 5)])
def test_replay_caches_actual_sp_event(initial_state, expected_seed):
  cp = messaging.new_message("carParams")
  events = [cp]
  for seed in (4, 5):
    evt = messaging.new_message(LIVE_TORQUE_PARAMETERS_SP_SERVICE)
    sp = getattr(evt, LIVE_TORQUE_PARAMETERS_SP_SERVICE)
    sp.version = VERSION
    sp.seedVersion = seed
    sp.speedBinCenters = [10.0]
    sp.speedBinLatAccelFactors = [2.0]
    sp.speedBinFrictions = [0.125]
    sp.speedBinValid = [True]
    events.append(evt)
  params = get_custom_params_from_lr([messaging.log_from_bytes(evt.to_bytes()) for evt in events], initial_state=initial_state)
  assert "CarParamsPrevRoute" in params
  with log.Event.from_bytes(params[LIVE_TORQUE_PARAMETERS_SP_KEY]) as evt:
    assert evt.which() == LIVE_TORQUE_PARAMETERS_SP_SERVICE
    sp = getattr(evt, LIVE_TORQUE_PARAMETERS_SP_SERVICE)
    assert sp.version == VERSION
    assert sp.seedVersion == expected_seed
    assert list(sp.speedBinCenters) == [10.0]
    assert len(sp.speedBinPoints) == 0
  empty_cp = messaging.new_message("carParams")
  assert LIVE_TORQUE_PARAMETERS_SP_KEY not in get_custom_params_from_lr(
    [messaging.log_from_bytes(empty_cp.to_bytes())], initial_state=initial_state)
