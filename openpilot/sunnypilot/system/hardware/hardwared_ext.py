"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.common.params import Params, ParamKeyFlag
from openpilot.sunnypilot.selfdrive.car.stock_ecu_handback import StockEcuHandBackGate


class HardwaredExt:
  """zoompilot's hooks into hardwared's hardware_thread, one object so hardwared.py carries one-line call sites.

  Ordering contract with hardware_thread: update runs at the top of every loop, before the
  onroad conditions are evaluated. It returns True on the loop that starts an onroad cycle.

  Two software stops are brokered here. OnroadCycleRequested is upstream's; OffroadModeRequested
  replaces a direct OffroadMode write from the UI, because pandad reads OffroadMode itself and
  drops the panda's ignition within 100 ms, before any hand-back could run. Both wait for the
  stock ECU hand-back while onroad (stock_ecu_handback.py).
  """

  def __init__(self, params: Params) -> None:
    self.params = params
    self.handback = StockEcuHandBackGate(params)

  def update(self, started: bool) -> bool:
    cycle = self.params.get_bool("OnroadCycleRequested")
    offroad = self.params.get_bool("OffroadModeRequested")
    if not (cycle or offroad):
      self.handback.reset()
      return False
    if not self.handback.ready(started):
      return False
    if offroad:
      self.params.put_bool("OffroadModeRequested", False, block=True)
      self.params.put_bool("OffroadMode", True, block=True)
    if cycle:
      self.params.put_bool("OnroadCycleRequested", False, block=True)
      self.on_onroad_cycle()
    return cycle

  def on_onroad_cycle(self) -> None:
    # pandad races manager's onroad-transition param clearing when the cycle restarts.
    # If it wins, it applies the previous session's CarParams safety immediately and
    # opens the harness relay seconds before controls come up, cutting the camera off
    # from the car long enough to fault it. Run the same clear early so the new
    # session sequences like a normal boot: ELM327 (relay closed) until the fresh
    # CarParams is ready.
    self.params.clear_all(ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION)
