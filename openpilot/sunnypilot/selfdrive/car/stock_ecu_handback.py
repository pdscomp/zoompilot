"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Consumer side of the stock ECU hand-back.

Any software-initiated end of the onroad state (onroad cycle for a calibration reset or a
restart-needed toggle, forced offroad, reboot, shutdown, uninstall) kills TX within about
100 ms of `started` dropping. A brand that silences a stock ECU for the drive needs those
processes alive long enough to hand it back first, or the ECU recovers unattended through
its S3 timeout in a degraded state that blocks cruise until the next ignition.

The consumer that would end the onroad state asks card through StockEcuHandBackRequested
and holds its action until card answers StockEcuHandBackDone or the wait runs out. card
answers at once when nothing was taken over. Ignition off and power loss cannot be held.
"""
import time

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

# The producer's own budget is one UDS attempt (10 s) plus the restore window; past this the
# processes are assumed dead or the radar unanswering, and the stop goes ahead regardless.
HANDBACK_WAIT_T = 15.0


class StockEcuHandBackGate:
  def __init__(self, params: Params, now=time.monotonic) -> None:
    self.params = params
    self.now = now
    self.requested_ts: float | None = None

  @property
  def pending(self) -> bool:
    return self.requested_ts is not None

  def ready(self, started: bool) -> bool:
    """True when the caller may end the onroad state now; the wait is cleared with it.
    Call every loop while the stop is wanted, reset() if it is withdrawn."""
    if not started:
      self.requested_ts = None
      return True
    if self.requested_ts is None:
      self.requested_ts = self.now()
      self.params.put_bool("StockEcuHandBackRequested", True, block=True)
      cloudlog.warning("stock ECU hand-back requested before ending onroad")
      return False
    if self.params.get_bool("StockEcuHandBackDone"):
      self.requested_ts = None
      return True
    if self.now() - self.requested_ts > HANDBACK_WAIT_T:
      cloudlog.error("stock ECU hand-back did not complete in time, ending onroad anyway")
      self.requested_ts = None
      return True
    return False

  def reset(self) -> None:
    self.requested_ts = None
