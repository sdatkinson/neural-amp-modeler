"""
Parametric NAM models and datasets.

Importing this package installs the parametric registrations that are available in this
package.
"""

from ...data import register_dataset_initializer as _register_dataset_initializer
from .. import factory as _factory
from ._dataset import data_config_from_model
from ._dataset import init_dataset as _init_dataset
from ._dataset import ParametricDataset
from ._export import bake
from ._export import bake_to_files
from ._export import export_parametric
from ._export import output_scale_from_datasets
from ._active_learning_params import abbreviate_param_names
from ._active_learning_params import assemble_raw_params
from ._active_learning_params import decode_named_params
from ._active_learning_params import format_param_value
from ._active_learning_params import make_capture_y_path
from ._active_learning_params import quantize_to_capture_grid
from ._active_learning_params import split_param_indices
from ._active_learning_params import switch_combinations
from ._concat_lstm import ConcatLSTM
from ._hypernet import Hypernetwork
from ._hyperwavenet import HyperWaveNet
from ._spec import ParamSpec

_register_dataset_initializer("parametric", _init_dataset)
_factory.register("ConcatLSTM", ConcatLSTM.init_from_config)
_factory.register("HyperWaveNet", HyperWaveNet.init_from_config)

__all__ = [
    "abbreviate_param_names",
    "assemble_raw_params",
    "ConcatLSTM",
    "decode_named_params",
    "format_param_value",
    "Hypernetwork",
    "HyperWaveNet",
    "make_capture_y_path",
    "ParamSpec",
    "ParametricDataset",
    "bake",
    "bake_to_files",
    "data_config_from_model",
    "export_parametric",
    "output_scale_from_datasets",
    "quantize_to_capture_grid",
    "split_param_indices",
    "switch_combinations",
]
