"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Onroad orchestration for the AlphaLongitudinalEnabled toggle.

The alpha-long param is read once at fingerprint, so applying a change requires an
onroad cycle. The UI only writes the param; card owns the cycle request so brands that
silence a stock ECU can hand it back first. pandad blocks TX within ~100 ms of
`started` dropping, so the hand-back must finish before the cycle is requested
(docs/zoompilot/mazda-longitudinal.md).

The cycle also waits for the car to stand still: dropping the session mid-drive leaves
the radar to recover through its ~5 s S3 timeout in a degraded state that blocks cruise
until the next ignition. The UI only refuses the flip while engaged, so a toggle flip
while rolling would otherwise end the session at speed.

Mazda op-long hand-back: assert CarControlSP.stockEcuHandBack -> carcontroller stops
tester present and requests the radar's default session -> the stock radar's CRZ_INFO
returns -> the session manager confirms restoration -> take the action.

The same hand-back serves every other software stop (calibration reset, restart-needed
toggle, forced offroad, reboot, shutdown, uninstall): the consumer sets
StockEcuHandBackRequested and waits for StockEcuHandBackDone (stock_ecu_handback.py).
Those stops are answered as soon as the radar is back, moving or not: the stop is
happening either way and a hand-back at speed beats the unattended S3 recovery. The
answer is "done" on failure too, so a radar that never answers cannot pin a reboot.
"""

from opendbc.car import DT_CTRL, structs
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

STANDSTILL_V = 0.1   # m/s, below this the car counts as stopped
STANDSTILL_T = 0.5   # s the car has to stay below STANDSTILL_V before the cycle is acted on


class StandstillGate:
  """Debounced 'stopped for STANDSTILL_T' at a fixed update rate."""

  def __init__(self, rate_hz: float):
    self.frames_needed = max(1, int(round(STANDSTILL_T * rate_hz)))
    self.stopped_frames = 0

  def update(self, v_ego: float) -> bool:
    if v_ego < STANDSTILL_V:
      self.stopped_frames = min(self.stopped_frames + 1, self.frames_needed)
    else:
      self.stopped_frames = 0
    return self.stopped_frames >= self.frames_needed

  @property
  def stopped(self) -> bool:
    return self.stopped_frames >= self.frames_needed


class AlphaLongToggleMonitor:
  def __init__(self, CP: structs.CarParams, params: Params, stock_ecu_session=None):
    """stock_ecu_session: the interface's session manager (handback_completed / handback_failed)
    when this platform silences a stock ECU under openpilot longitudinal, else None."""
    self.params = params
    self.session = stock_ecu_session
    self.op_long = CP.openpilotLongitudinalControl
    self.alpha_available = CP.alphaLongitudinalAvailable
    self._params_snapshot = (self.op_long, False, False)
    self.handback_started = False
    self.done = False
    self.repeat_logged = False
    self.restore_failure_logged = False
    self.handback_requested = False   # irrevocable manager stop/onroad cycle
    self.toggle_handback = False      # the toggle path started the hand-back and owns the cycle
    self.handback_answered = False
    self._handback_answer_written = False
    self._cycle_written = False
    self.standstill = StandstillGate(1 / DT_CTRL)
    # One cycle per ignition: the marker is CLEAR_ON_IGNITION_ON and card restarts on the
    # cycle, so a mismatch that survives the restart would otherwise cycle forever. Read
    # once; request_cycle latches done and update_params persists the attempt once.
    self.cycle_attempted = params.get_bool("AlphaLongCycleAttempted")
    if self.cycle_attempted and params.get_bool("AlphaLongitudinalEnabled") == self.op_long:
      # the cycle took; a later flip this ignition is a new request, not a persisting one
      params.put_bool("AlphaLongCycleAttempted", False)
      self.cycle_attempted = False

  def update_params(self) -> None:
    # Publish a coherent input snapshot; never clear controls-owned latches here.
    toggle_enabled = self.params.get_bool("AlphaLongitudinalEnabled")
    committed_stop = self.params.get_bool("StockEcuHandBackRequested")
    offroad = self.params.get_bool("OffroadModeRequested") or self.params.get_bool("OffroadMode")
    self._params_snapshot = (toggle_enabled, committed_stop, offroad)
    # Persist each controls-thread action once, restoration before recovery cycle.
    if self.handback_answered and not self._handback_answer_written:
      self.params.put_bool("StockEcuHandBackDone", True, block=True)
      self._handback_answer_written = True
    if self.done and not self._cycle_written:
      self.params.put_bool("AlphaLongCycleAttempted", True, block=True)
      self.params.put_bool("OnroadCycleRequested", True, block=True)
      self._cycle_written = True

  def request_cycle(self) -> None:
    self.done = True

  @property
  def restored(self) -> bool:
    return self.session is not None and self.session.handback_completed

  @property
  def restore_failed(self) -> bool:
    return self.session is not None and self.session.handback_failed

  def update(self, CS: structs.CarState, CC: structs.CarControl, CC_SP: structs.CarControlSP) -> None:
    """Runs at 100 Hz from controls_update, before CI.apply."""
    toggle_enabled, committed_stop, offroad = self._params_snapshot
    self.handback_requested |= committed_stop
    external_stop = self.handback_requested or offroad
    stopped = self.standstill.update(CS.vEgo if CS.canValid and not CS.canTimeout else float("inf"))
    # MADS stays engaged while paused, even with longitudinal/lateral output inactive.
    engaged = CC.enabled or CC.latActive or CC_SP.mads.enabled
    self._serve_external_stop(CC_SP, external_stop, engaged=engaged)
    if self.handback_started:
      CC_SP.stockEcuHandBack = True
    if self.done:
      return
    if self.handback_started and not external_stop and not self.toggle_handback:
      if self.restored and not engaged and stopped:
        self._answer_external_stop()
        self.request_cycle()
      return
    toggle_mismatch = self.alpha_available and toggle_enabled != self.op_long
    if toggle_mismatch and self.cycle_attempted:
      if not self.repeat_logged:
        cloudlog.warning("alpha long toggle mismatch persists after this ignition's onroad cycle, not cycling again")
        self.repeat_logged = True
      toggle_mismatch = False
    if not toggle_mismatch and not self.toggle_handback:
      return

    # Wait for disengagement and standstill because parameters can change outside the UI.
    # Once started, hand-back remains asserted while the final cycle waits.
    if self.session is None:
      # No ECU hand-back is required when enabling or on unaffected platforms.
      if not engaged and stopped:
        self.request_cycle()
      return

    if not self.handback_started and (engaged or not stopped):
      return

    self.toggle_handback = True
    self.handback_started = True
    CC_SP.stockEcuHandBack = True
    # A reversed toggle still finishes the outstanding restoration, then rebuilds the
    # interface with the latest setting. Never alternate diagnostic sessions mid-handback.
    if self.restore_failed and not self.restore_failure_logged:
      cloudlog.error("Mazda radar restoration failed; waiting for stock traffic before cycling")
      self.restore_failure_logged = True
    if self.restored and not engaged and stopped:
      self.request_cycle()

  def _serve_external_stop(self, CC_SP: structs.CarControlSP, requested: bool, *, engaged: bool) -> None:
    if not requested or self.handback_answered:
      return
    if self.session is None:
      self._answer_external_stop()
      return
    # Never start on an engaged car: the hand-back revokes availability under the driver.
    # Once started (by either path) it runs to its end.
    if not self.handback_started and engaged:
      return
    self.handback_started = True
    CC_SP.stockEcuHandBack = True
    if self.restored or self.restore_failed:
      if self.restore_failed:
        cloudlog.error("Mazda radar restoration failed before the requested stop; stopping anyway")
      self._answer_external_stop()

  def _answer_external_stop(self) -> None:
    self.handback_answered = True
