# File: export.py
# Created Date: Sunday February 6th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Export a model to TorchScript
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import torch

from nam.models import Model
from nam.models._base import ParametricBaseNet
from nam.models.parametric.catnets import Param


class Dummy(torch.nn.Module):
    def forward(self, x):
        return x[8191:]


def main(args):
    outdir = Path(args.outdir)
    with open(args.model_config_path, "r") as fp:
        net = Model.load_from_checkpoint(
            args.checkpoint, **Model.parse_config(json.load(fp))
        ).net
    if not isinstance(net, ParametricBaseNet):
        export_args = (outdir,)
    else:
        if args.param_config is None:
            raise ValueError("Require param config for parametric model")
        with open(Path(args.param_config), "r") as fp:
            param_config = {
                k: Param.init_from_config(v) for k, v in json.load(fp).items()
            }
        export_args = (outdir, param_config)
    net.eval()
    outdir.mkdir(parents=True, exist_ok=True)
    net.export(*export_args, include_snapshot=args.include_snapshot)
    net.export_cpp_header(Path(export_args[0], "HardCodedModel.h"), *export_args[1:])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_config_path", type=str)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("outdir")
    parser.add_argument(
        "--include-snapshot",
        "-s",
        help="Computes an example input-output pair for the model for debugging "
        "purposes",
    )
    parser.add_argument(
        "--param-config", type=str, help="Configuration for a parametric model"
    )
    main(parser.parse_args())
