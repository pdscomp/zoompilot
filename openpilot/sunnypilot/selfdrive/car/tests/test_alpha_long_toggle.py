"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from opendbc.car import structs
import pytest

from openpilot.cereal import custom
from opendbc.car.structs import car
from opendbc.car.mazda.radar_session import RADAR_RESTORE_FRAMES, RADAR_SESSION_LIMIT_FRAMES, RadarSessionManager, RadarSessionState
from opendbc.car.mazda.values import CarControllerParams
from openpilot.selfdrive.car.helpers import convert_carControlSP
from openpilot.sunnypilot.selfdrive.car.alpha_long_toggle import AlphaLongToggleMonitor, \
  STANDSTILL_V, STANDSTILL_T, StandstillGate

from openpilot.sunnypilot.selfdrive.car.stock_ecu_handback import StockEcuHandBackGate
from openpilot.sunnypilot.selfdrive.car.tests.fakes import FakeClock, FakeParams
from openpilot.sunnypilot.system.hardware.hardwared_ext import HardwaredExt

MOVING_V = 12.0


class FakeSession:
  """The interface's radar session manager as the monitor sees it."""

  def __init__(self):
    self.handback_completed = False
    self.handback_failed = False


def _monitor(toggle: bool, brand="mazda", op_long=True, alpha_avail=True, cycle_attempted=False, parked=True):
  cp = structs.CarParams()
  cp.brand = brand
  cp.openpilotLongitudinalControl = op_long
  cp.alphaLongitudinalAvailable = alpha_avail
  params = FakeParams(AlphaLongitudinalEnabled=toggle, AlphaLongCycleAttempted=cycle_attempted)
  # card_ext hands the session over only on a platform that silences a stock ECU under op long
  session = FakeSession() if brand == "mazda" and op_long else None
  m = AlphaLongToggleMonitor(cp, params, session)
  m.update_params()
  if parked:
    # the car has been sitting still since card started; the standstill debounce is already satisfied
    m.standstill.stopped_frames = m.standstill.frames_needed
  return m, params


def _step(monitor, restored=False, restore_failed=False, acc_faulted=False, enabled=False,
          v_ego=0.0, can_valid=True, flush_params=True):
  if monitor.session is not None:
    monitor.session.handback_completed = restored
    monitor.session.handback_failed = restore_failed
  cs = structs.CarState()
  cs.canValid = can_valid
  cs.accFaulted = acc_faulted
  cs.vEgo = v_ego
  cc = structs.CarControl()
  cc.enabled = enabled
  cc_sp = structs.CarControlSP()
  monitor.update(cs, cc, cc_sp)
  if flush_params:
    monitor.update_params()
  return cc_sp


class TestAlphaLongToggleMonitor:
  def test_no_mismatch_no_action(self):
    m, params = _monitor(toggle=True, op_long=True)
    cc_sp = _step(m)
    assert not cc_sp.stockEcuHandBack
    assert not params.get_bool("OnroadCycleRequested")

  def test_enable_direction_cycles_immediately(self):
    m, params = _monitor(toggle=True, op_long=False)
    cc_sp = _step(m)
    assert not cc_sp.stockEcuHandBack
    assert params.get_bool("OnroadCycleRequested")

  def test_disable_runs_handback_until_radar_returns(self):
    m, params = _monitor(toggle=False, op_long=True)
    # radar still silent: hand-back asserted, no cycle yet
    for _ in range(50):
      cc_sp = _step(m, acc_faulted=False)
      assert cc_sp.stockEcuHandBack
      assert not params.get_bool("OnroadCycleRequested")
    # stock radar heard again: cycle requested
    cc_sp = _step(m, restored=True)
    assert cc_sp.stockEcuHandBack
    assert params.get_bool("OnroadCycleRequested")

  def test_timeout_does_not_authorize_a_cycle(self):
    m, params = _monitor(toggle=False, op_long=True)
    for _ in range(RADAR_SESSION_LIMIT_FRAMES):
      _step(m, restore_failed=True)
    assert not params.get_bool("OnroadCycleRequested")

  def test_waits_for_disengagement(self):
    m, params = _monitor(toggle=False, op_long=True)
    cc_sp = _step(m, enabled=True)
    assert not cc_sp.stockEcuHandBack
    # once started, engagement no longer pauses the sequence
    _step(m, enabled=False)
    cc_sp = _step(m, enabled=True)
    assert cc_sp.stockEcuHandBack

  def test_handback_stays_asserted_after_done(self):
    # CC_SP is rebuilt each frame; dropping the assert once done latched made the session
    # manager read a withdrawal and re-silence the radar it had just handed back, right
    # before shutdown
    m, params = _monitor(toggle=False, op_long=True)
    _step(m)
    _step(m, restored=True)
    assert params.get_bool("OnroadCycleRequested")
    for _ in range(10):
      cc_sp = _step(m, restored=True)
      assert cc_sp.stockEcuHandBack

  def test_no_assert_after_done_when_nothing_was_handed_back(self):
    # the enable direction never starts a hand-back, so there is nothing to keep asserting
    m, params = _monitor(toggle=True, op_long=False)
    _step(m)
    assert params.get_bool("OnroadCycleRequested")
    cc_sp = _step(m)
    assert not cc_sp.stockEcuHandBack

  def test_non_mazda_disable_cycles_immediately(self):
    m, params = _monitor(toggle=False, brand="toyota", op_long=True)
    cc_sp = _step(m)
    assert not cc_sp.stockEcuHandBack
    assert params.get_bool("OnroadCycleRequested")

  def test_unavailable_never_acts(self):
    m, params = _monitor(toggle=True, op_long=False, alpha_avail=False)
    cc_sp = _step(m)
    assert not cc_sp.stockEcuHandBack
    assert not params.get_bool("OnroadCycleRequested")

  def test_cycle_requested_only_once(self):
    m, params = _monitor(toggle=True, op_long=False)
    _step(m)
    params.put_bool("OnroadCycleRequested", False)  # hardwared consumed it
    _step(m)
    assert not params.get_bool("OnroadCycleRequested")


class TestEngagedDefersFinish:
  # the UIs block both actions while engaged, but the params can flip from anywhere

  def test_enable_direction_waits_for_disengage(self):
    m, params = _monitor(toggle=True, op_long=False)
    for _ in range(50):
      _step(m, enabled=True)
      assert not params.get_bool("OnroadCycleRequested")
    _step(m, enabled=False)
    assert params.get_bool("OnroadCycleRequested")

  def test_non_mazda_disable_waits_for_disengage(self):
    m, params = _monitor(toggle=False, brand="toyota", op_long=True)
    _step(m, enabled=True)
    assert not params.get_bool("OnroadCycleRequested")
    _step(m, enabled=False)
    assert params.get_bool("OnroadCycleRequested")

  def test_mazda_radar_return_while_engaged_holds_cycle_not_handback(self):
    m, params = _monitor(toggle=False, op_long=True)
    _step(m)  # hand-back starts disengaged
    # engaged when the radar comes back: keep asserting, do not cycle yet
    for _ in range(50):
      cc_sp = _step(m, restored=True, enabled=True)
      assert cc_sp.stockEcuHandBack
      assert not params.get_bool("OnroadCycleRequested")
    cc_sp = _step(m, restored=True, enabled=False)
    assert cc_sp.stockEcuHandBack
    assert params.get_bool("OnroadCycleRequested")

  def test_mazda_timeout_while_engaged_holds_cycle(self):
    m, params = _monitor(toggle=False, op_long=True)
    _step(m)
    for _ in range(RADAR_SESSION_LIMIT_FRAMES + 10):
      cc_sp = _step(m, enabled=True)
      assert cc_sp.stockEcuHandBack
    assert not params.get_bool("OnroadCycleRequested")
    _step(m, enabled=False, restored=True)
    assert params.get_bool("OnroadCycleRequested")


class TestOneCyclePerIgnition:
  def test_cycle_marks_the_ignition(self):
    m, params = _monitor(toggle=True, op_long=False)
    _step(m)
    assert params.get_bool("AlphaLongCycleAttempted")

  def test_persisting_mismatch_does_not_cycle_again(self):
    # card restarted after the cycle and the fingerprint still does not satisfy the toggle
    m, params = _monitor(toggle=True, op_long=False, cycle_attempted=True)
    for _ in range(50):
      _step(m)
    assert not params.get_bool("OnroadCycleRequested")
    assert params.get_bool("AlphaLongCycleAttempted")

  def test_persisting_mismatch_mazda_disable_does_not_hand_back(self):
    m, params = _monitor(toggle=False, op_long=True, cycle_attempted=True)
    cc_sp = _step(m, restored=True)
    assert not cc_sp.stockEcuHandBack
    assert not params.get_bool("OnroadCycleRequested")

  def test_satisfied_toggle_clears_the_marker(self):
    # the cycle took: a later flip this ignition is a fresh request
    m, params = _monitor(toggle=True, op_long=True, cycle_attempted=True)
    assert not params.get_bool("AlphaLongCycleAttempted")
    params.put_bool("AlphaLongitudinalEnabled", False)
    m.update_params()
    _step(m, restored=True)
    assert params.get_bool("OnroadCycleRequested")


class TestStandstillGate:
  # The cycle may not end the session under a moving car: the UI only refuses the toggle
  # while engaged, and on file it was flipped at up to 19 m/s

  def _stop(self, m, **kw):
    for _ in range(m.standstill.frames_needed - 1):
      _step(m, v_ego=0.0, **kw)
      assert not m.done
    return _step(m, v_ego=0.0, **kw)

  def test_toggle_cycle_waits_for_standstill(self):
    m, params = _monitor(toggle=False, op_long=True, parked=False)
    # Neither restoration nor the cycle may begin while moving.
    for _ in range(100):
      cc_sp = _step(m, restored=True, v_ego=MOVING_V)
      assert not cc_sp.stockEcuHandBack
      assert not params.get_bool("OnroadCycleRequested")
    self._stop(m, restored=True)
    assert params.get_bool("OnroadCycleRequested")

  def test_handback_timeout_still_waits_for_standstill(self):
    m, params = _monitor(toggle=False, op_long=True)
    assert _step(m).stockEcuHandBack  # restoration already started while parked
    for _ in range(RADAR_SESSION_LIMIT_FRAMES + 100):
      cc_sp = _step(m, v_ego=MOVING_V)
      assert cc_sp.stockEcuHandBack
    assert not params.get_bool("OnroadCycleRequested")
    self._stop(m, restored=True)
    assert params.get_bool("OnroadCycleRequested")

  def test_enable_direction_waits_for_standstill(self):
    m, params = _monitor(toggle=True, op_long=False, parked=False)
    for _ in range(100):
      _step(m, v_ego=MOVING_V)
    assert not params.get_bool("OnroadCycleRequested")
    self._stop(m)
    assert params.get_bool("OnroadCycleRequested")

  def test_creep_below_threshold_counts_as_stopped(self):
    m, params = _monitor(toggle=True, op_long=False, parked=False)
    for _ in range(m.standstill.frames_needed):
      _step(m, v_ego=STANDSTILL_V / 2)
    assert params.get_bool("OnroadCycleRequested")

  def test_brief_stop_does_not_count(self):
    m, params = _monitor(toggle=True, op_long=False, parked=False)
    for _ in range(m.standstill.frames_needed - 1):
      _step(m, v_ego=0.0)
    _step(m, v_ego=MOVING_V)
    for _ in range(m.standstill.frames_needed - 1):
      _step(m, v_ego=0.0)
    assert not params.get_bool("OnroadCycleRequested")

  def test_engaged_at_standstill_still_holds(self):
    # op-long holding the car at a light: stopped but engaged, so nothing fires until disengaged
    m, params = _monitor(toggle=False, op_long=True, parked=False)
    _step(m, v_ego=MOVING_V)
    for _ in range(m.standstill.frames_needed + 10):
      _step(m, restored=True, enabled=True, v_ego=0.0)
    assert not params.get_bool("OnroadCycleRequested")
    _step(m, restored=True, enabled=False, v_ego=0.0)
    assert params.get_bool("OnroadCycleRequested")

  def test_gate_needs_at_least_one_frame(self):
    assert StandstillGate(1 / (2 * STANDSTILL_T)).frames_needed == 1


class TestExternalStop:
  """A reboot, shutdown, forced offroad or calibration reset asks card for the hand-back
  through StockEcuHandBackRequested and waits on StockEcuHandBackDone."""

  @staticmethod
  def _request(m, params):
    params.put_bool("StockEcuHandBackRequested", True)
    m.update_params()

  def test_no_takeover_brand_answers_at_once(self):
    for brand, op_long in (("toyota", True), ("mazda", False)):
      m, params = _monitor(toggle=True, brand=brand, op_long=op_long, alpha_avail=False)
      self._request(m, params)
      cc_sp = _step(m)
      assert not cc_sp.stockEcuHandBack
      assert params.get_bool("StockEcuHandBackDone")
      assert not params.get_bool("OnroadCycleRequested")

  def test_mazda_hands_back_then_answers_without_cycling(self):
    m, params = _monitor(toggle=True, op_long=True, parked=False)
    self._request(m, params)
    # moving is fine: the stop is happening either way
    for _ in range(50):
      cc_sp = _step(m, v_ego=MOVING_V)
      assert cc_sp.stockEcuHandBack
      assert not params.get_bool("StockEcuHandBackDone")
    cc_sp = _step(m, v_ego=MOVING_V, restored=True)
    assert cc_sp.stockEcuHandBack
    assert params.get_bool("StockEcuHandBackDone")
    # the toggle is satisfied, so no onroad cycle rides along with the stop
    assert not params.get_bool("OnroadCycleRequested")
    # the assert holds until the process dies
    for _ in range(10):
      cc_sp = _step(m, restored=True)
      assert cc_sp.stockEcuHandBack

  def test_failure_still_answers(self):
    m, params = _monitor(toggle=True, op_long=True)
    self._request(m, params)
    _step(m)
    _step(m, restore_failed=True)
    assert params.get_bool("StockEcuHandBackDone")

  def test_waits_for_disengagement_to_start(self):
    m, params = _monitor(toggle=True, op_long=True)
    self._request(m, params)
    for _ in range(20):
      cc_sp = _step(m, enabled=True)
      assert not cc_sp.stockEcuHandBack
    cc_sp = _step(m, enabled=False)
    assert cc_sp.stockEcuHandBack
    # once started, a re-engagement no longer pauses it
    cc_sp = _step(m, enabled=True, restored=True)
    assert cc_sp.stockEcuHandBack
    assert params.get_bool("StockEcuHandBackDone")

  def test_joins_a_toggle_handback_already_running(self):
    m, params = _monitor(toggle=False, op_long=True)
    _step(m)
    self._request(m, params)
    cc_sp = _step(m, enabled=True)  # engaged would block a fresh start, not a running one
    assert cc_sp.stockEcuHandBack
    _step(m, restored=True)
    assert params.get_bool("StockEcuHandBackDone")
    assert params.get_bool("OnroadCycleRequested")  # the toggle path still owns its cycle

  def test_answered_once(self):
    m, params = _monitor(toggle=True, op_long=True)
    self._request(m, params)
    _step(m)
    _step(m, restored=True)
    params.put_bool("StockEcuHandBackDone", False)
    _step(m, restored=True)
    assert not params.get_bool("StockEcuHandBackDone")


class TestOffroadCancellation:
  @staticmethod
  def _start_request(m, params):
    ext = HardwaredExt(params)
    ext.handback.now = FakeClock()
    params.put_bool("OffroadModeRequested", True)
    assert not ext.update(started=True)
    assert not params.get_bool("StockEcuHandBackRequested")
    return ext

  def test_cancel_before_sampling_or_before_disengagement(self):
    for sampled in (False, True):
      m, params = _monitor(toggle=True)
      ext = self._start_request(m, params)
      if sampled:
        m.update_params()
        assert not _step(m, enabled=True, flush_params=False).stockEcuHandBack
      params.put_bool("OffroadModeRequested", False)
      m.update_params()
      assert not _step(m, flush_params=False).stockEcuHandBack
      assert not m.handback_started
      assert not m.handback_requested
      assert not params.get_bool("StockEcuHandBackDone")
      assert not params.get_bool("OnroadCycleRequested")
      assert not ext.update(started=True)
      assert not ext.handback.pending
      params.put_bool("OffroadModeRequested", True)
      assert not ext.update(started=True)
      m.update_params()
      assert _step(m).stockEcuHandBack

  def test_cancel_after_start_finishes_then_cycles_safely(self):
    m, params = _monitor(toggle=True, parked=False)
    ext = self._start_request(m, params)
    m.update_params()
    assert _step(m, v_ego=MOVING_V).stockEcuHandBack
    params.put_bool("OffroadModeRequested", False)
    assert not ext.update(started=True)
    m.update_params()
    for kw in (
      {"restore_failed": True},
      {"restored": True, "v_ego": MOVING_V},
      {"restored": True, "enabled": True},
      {"restored": True, "can_valid": False},
    ):
      assert _step(m, **kw).stockEcuHandBack
      assert not m.done
      assert not params.get_bool("OnroadCycleRequested")
    for _ in range(m.standstill.frames_needed + 1):
      assert _step(m, restored=True).stockEcuHandBack
      if m.done:
        break
    assert m.done
    assert params.get_bool("StockEcuHandBackDone")
    assert params.get_bool("AlphaLongCycleAttempted")
    assert params.get_bool("OnroadCycleRequested")
    assert not params.get_bool("OffroadMode")
    assert not params.get_bool("OffroadModeRequested")
    assert not ext.update(started=True)
    assert ext.update(started=True)
    assert not params.get_bool("OnroadCycleRequested")
    assert _step(m, restored=True).stockEcuHandBack
    assert not params.get_bool("OnroadCycleRequested")

  def test_manager_stop_survives_cancellation_before_or_after_start(self):
    for already_started in (False, True):
      m, params = _monitor(toggle=True, parked=False)
      ext = self._start_request(m, params)
      m.update_params()
      cc_sp = _step(m, enabled=not already_started, v_ego=MOVING_V)
      assert cc_sp.stockEcuHandBack == already_started
      manager_gate = StockEcuHandBackGate(params, now=FakeClock())
      assert not manager_gate.ready(started=True)
      params.put_bool("OffroadModeRequested", False)
      assert not ext.update(started=True)
      assert params.get_bool("StockEcuHandBackRequested")
      m.update_params()
      assert _step(m, v_ego=MOVING_V).stockEcuHandBack
      assert _step(m, restored=True, v_ego=MOVING_V).stockEcuHandBack
      assert manager_gate.ready(started=True)
      assert not params.get_bool("OnroadCycleRequested")
      assert not params.get_bool("OffroadMode")

  def test_applying_offroad_cannot_look_like_cancellation_between_writes(self):
    m, params = _monitor(toggle=True)
    ext = self._start_request(m, params)
    m.update_params()
    assert _step(m).stockEcuHandBack
    assert _step(m, restored=True).stockEcuHandBack
    assert params.get_bool("StockEcuHandBackDone")
    put_bool = params.put_bool

    def observe_write(key, value, **kwargs):
      put_bool(key, value, **kwargs)
      if key in ("OffroadMode", "OffroadModeRequested"):
        m.update_params()
        assert _step(m, restored=True, flush_params=False).stockEcuHandBack
        assert not m.done

    params.put_bool = observe_write
    try:
      assert not ext.update(started=True)
    finally:
      params.put_bool = put_bool
    assert params.get_bool("OffroadMode")
    assert not params.get_bool("OffroadModeRequested")
    assert not params.get_bool("OnroadCycleRequested")
    params.put_bool("OffroadMode", False)
    m.update_params()
    assert _step(m, restored=True).stockEcuHandBack
    assert params.get_bool("OnroadCycleRequested")

  def test_control_update_has_no_params_access_and_flushes_once(self):
    for external in (False, True):
      m, params = _monitor(toggle=external)
      if external:
        params.put_bool("StockEcuHandBackRequested", True)
        m.update_params()
      m.params = None
      try:
        assert _step(m, restored=True, flush_params=False).stockEcuHandBack
      finally:
        m.params = params
      assert not params.get_bool("StockEcuHandBackDone")
      assert not params.get_bool("OnroadCycleRequested")
      m.update_params()
      key = "StockEcuHandBackDone" if external else "OnroadCycleRequested"
      assert params.get_bool(key)
      if not external:
        assert params.get_bool("AlphaLongCycleAttempted")
      params.put_bool(key, False)
      m.update_params()
      assert not params.get_bool(key)


def _mads_radar_monitor(*, toggle=True, op_long=True):
  # The fake session flags are replaced by a real radar session manager driven through
  # the production CC_SP conversion, so lateral/MADS engagement is what the code reads.
  monitor, params = _monitor(toggle=toggle, op_long=op_long, parked=False)
  if op_long:
    monitor.session = RadarSessionManager()
    monitor.session.update(True, False, False, True, False, True, frame=0)
    assert monitor.session.state == RadarSessionState.SILENCED
  frame = 0

  def step(mode="disabled", *, alive=False, v_ego=0.0, can_valid=True, can_timeout=False):
    nonlocal frame
    frame += 1
    cc = car.CarControl.new_message(enabled=mode == "longitudinal",
                                    latActive=mode in ("active", "lateral"))
    wire = custom.CarControlSP.new_message()
    wire.mads.available = True
    wire.mads.enabled = mode in ("active", "paused")
    wire.mads.active = mode == "active"
    wire.mads.state = {"active": "enabled", "paused": "paused"}.get(mode, "disabled")
    cc_sp = convert_carControlSP(wire.as_reader())
    cs = structs.CarState(canValid=can_valid, canTimeout=can_timeout, vEgo=v_ego)
    monitor.update(cs, cc.as_reader(), cc_sp)
    if monitor.session is not None:
      monitor.session.update(True, alive, cc_sp.stockEcuHandBack,
                             v_ego < STANDSTILL_V, False, not alive,
                             bus_healthy=can_valid and not can_timeout, frame=frame)
    monitor.update_params()
    return cc_sp

  return monitor, params, step


@pytest.mark.parametrize("mode", ["longitudinal", "lateral", "active", "paused"])
@pytest.mark.parametrize("request_kind", ["offroad", "manager", "toggle", "no_session"])
def test_mads_fresh_actions_wait_for_disengagement(mode, request_kind):
  monitor, params, step = _mads_radar_monitor(toggle=request_kind != "toggle",
                                              op_long=request_kind != "no_session")
  if request_kind in ("offroad", "manager"):
    key = "OffroadModeRequested" if request_kind == "offroad" else "StockEcuHandBackRequested"
    params.put_bool(key, True)
    monitor.update_params()
  for _ in range(monitor.standstill.frames_needed + 1):
    assert not step(mode).stockEcuHandBack
    assert not monitor.handback_started
    assert not params.get_bool("OnroadCycleRequested")
  cc_sp = step()
  if request_kind == "no_session":
    assert params.get_bool("OnroadCycleRequested")
  else:
    assert cc_sp.stockEcuHandBack


@pytest.mark.parametrize("mode", ["longitudinal", "lateral", "active", "paused"])
@pytest.mark.parametrize("request_kind", ["cancelled_offroad", "toggle"])
def test_mads_restoration_continues_but_cycle_waits(mode, request_kind):
  monitor, params, step = _mads_radar_monitor(toggle=request_kind != "toggle")
  if request_kind == "cancelled_offroad":
    params.put_bool("OffroadModeRequested", True)
    monitor.update_params()
  for _ in range(monitor.standstill.frames_needed):
    step()
  assert monitor.handback_started
  assert not monitor.restored
  if request_kind == "cancelled_offroad":
    params.put_bool("OffroadModeRequested", False)
    monitor.update_params()
  budget = CarControllerParams.RADAR_UDS_STEP + RADAR_RESTORE_FRAMES + monitor.standstill.frames_needed
  for _ in range(budget):
    assert step(mode, alive=True).stockEcuHandBack
    assert not params.get_bool("OnroadCycleRequested")
  assert monitor.restored  # real stock-traffic window, not a stubbed success flag

  # Fully disengaged alone is insufficient: movement and bad CAN reset the debounce.
  for bad_state in ({"v_ego": MOVING_V}, {"can_valid": False}, {"can_timeout": True}):
    step(alive=True, **bad_state)
    assert not params.get_bool("OnroadCycleRequested")
  for _ in range(monitor.standstill.frames_needed - 1):
    step(alive=True)
    assert not params.get_bool("OnroadCycleRequested")
  step(alive=True)
  assert params.get_bool("OnroadCycleRequested")
  assert params.get_bool("AlphaLongCycleAttempted")
  if request_kind == "cancelled_offroad":
    assert params.get_bool("StockEcuHandBackDone")
