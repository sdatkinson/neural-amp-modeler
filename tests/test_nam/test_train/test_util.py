import pytest

from nam.train import lightning_module as _lightning_module
from nam.train import util as _util


@pytest.mark.parametrize(
    ("net_name", "expected"),
    (
        ("PackedWaveNet", _lightning_module.PackedLightningModule),
        ("WaveNet", _lightning_module.LightningModule),
        ("LSTM", _lightning_module.LightningModule),
    ),
)
def test_resolve_lightning_module_class(net_name, expected):
    model_config = {"net": {"name": net_name}}

    assert _util.resolve_lightning_module_class(model_config) is expected
