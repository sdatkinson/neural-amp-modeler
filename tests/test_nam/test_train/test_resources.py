import json as _json
from importlib import resources as _resources

from nam.models.wavenet import PackedWaveNet as _PackedWaveNet


def _load_packed_model_config():
    resource = _resources.files("nam.train._resources").joinpath(
        "config_model_packed.json"
    )
    with resource.open("r") as fp:
        return _json.load(fp)


def test_packaged_packed_model_config_loads():
    config = _load_packed_model_config()

    assert config["net"]["name"] == "PackedWaveNet"
    model = _PackedWaveNet.init_from_config(config["net"]["config"])
    assert model.num_submodels == len(config["net"]["config"]["submodels"])
