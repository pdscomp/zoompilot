"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from opendbc.car import structs
from openpilot.sunnypilot.selfdrive.car.alpha_long_toggle import AlphaLongToggleMonitor, \
  STANDSTILL_V, STANDSTILL_T, StandstillGate

from opendbc.car.mazda.radar_session import RADAR_SESSION_LIMIT_FRAMES
from openpilot.sunnypilot.selfdrive.car.tests.fakes import FakeParams

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


def _step(monitor, restored=False, restore_failed=False, acc_faulted=False, enabled=False, v_ego=0.0, can_valid=True):
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
