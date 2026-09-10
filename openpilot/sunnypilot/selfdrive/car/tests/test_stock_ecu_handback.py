"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.sunnypilot.selfdrive.car.stock_ecu_handback import HANDBACK_WAIT_T, StockEcuHandBackGate
from openpilot.sunnypilot.selfdrive.car.tests.fakes import FakeClock, FakeParams


def _gate(**bools):
  params = FakeParams(**bools)
  clock = FakeClock()
  gate = StockEcuHandBackGate(params, now=clock)
  return gate, params, clock


class TestStockEcuHandBackGate:
  def test_offroad_is_always_ready_and_asks_nothing(self):
    gate, params, _ = _gate()
    assert gate.ready(started=False)
    assert not params.get_bool("StockEcuHandBackRequested")

  def test_onroad_asks_once_then_waits_for_done(self):
    gate, params, _ = _gate()
    assert not gate.ready(started=True)
    assert params.get_bool("StockEcuHandBackRequested")
    for _ in range(5):
      assert not gate.ready(started=True)
    params.put_bool("StockEcuHandBackDone", True)
    assert gate.ready(started=True)

  def test_wait_is_bounded(self):
    gate, params, clock = _gate()
    assert not gate.ready(started=True)
    clock.t = HANDBACK_WAIT_T - 0.1
    assert not gate.ready(started=True)
    clock.t = HANDBACK_WAIT_T + 0.1
    assert gate.ready(started=True)

  def test_going_offroad_mid_wait_releases(self):
    gate, params, _ = _gate()
    assert not gate.ready(started=True)
    assert gate.ready(started=False)
    assert not gate.pending

  def test_reset_re_asks(self):
    gate, params, _ = _gate()
    gate.ready(started=True)
    params.put_bool("StockEcuHandBackRequested", False)  # card cleared it on its own transition
    gate.reset()
    assert not gate.ready(started=True)
    assert params.get_bool("StockEcuHandBackRequested")
