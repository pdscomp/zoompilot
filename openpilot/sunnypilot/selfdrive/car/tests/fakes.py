"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of zoompilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Test doubles shared by the sunnypilot overlay tests.
"""


class FakeParams:
  """Dict-backed Params: bools only, records clear_all flags."""

  def __init__(self, **bools):
    self.bools = dict(bools)
    self.cleared: list = []

  def get_bool(self, key):
    return self.bools.get(key, False)

  def put_bool(self, key, value, **kwargs):
    self.bools[key] = value

  def clear_all(self, flag):
    self.cleared.append(flag)


class FakeClock:
  """Injectable monotonic clock: set .t directly."""

  def __init__(self):
    self.t = 0.0

  def __call__(self):
    return self.t
