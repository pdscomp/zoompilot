"""Drive-start params dump + lateral-extension identity on carControlSP."""
from types import SimpleNamespace

from openpilot.cereal import custom
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.sunnypilot.selfdrive.controls.controlsd_ext import build_param_dump



def test_param_dump_records_settings_and_hides_secrets_and_blobs():
  with OpenpilotPrefix():
    params = Params()
    params.put_bool("NeuralNetworkLateralControl", True, block=True)
    params.put("TorqueControlTune", 2.0, block=True)
    params.put("AccessToken", "super-secret", block=True)          # DONT_LOG flag
    params.put("CarParamsPersistent", b"x" * 4096, block=True)     # oversized blob

    dump = {p["key"]: p for p in build_param_dump(params)}

    assert dump["NeuralNetworkLateralControl"]["value"] == b"True"
    assert dump["TorqueControlTune"]["value"] == b"2.0"
    assert "AccessToken" not in dump
    assert "CarParamsPersistent" not in dump

    # entries land in the capnp message and are absent on later frames
    cc = custom.CarControlSP.new_message()
    cc.params = build_param_dump(params)
    keys = [p.key for p in cc.params]
    assert "NeuralNetworkLateralControl" in keys and "AccessToken" not in keys



