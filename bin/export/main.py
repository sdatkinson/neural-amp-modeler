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
    net.cpu()
    net.eval()
    outdir.mkdir(parents=True, exist_ok=True)
    net.export(*export_args, include_snapshot=args.snapshot)
    if args.cpp:
        net.export_cpp_header(
            Path(export_args[0], "HardCodedModel.h"), *export_args[1:]
        )
    if args.onnx:
        net.export_onnx(Path(outdir, "model.onnx"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_config_path", type=str)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("outdir")
    parser.add_argument(
        "--param-config", type=str, help="Configuration for a parametric model"
    )
    parser.add_argument("--onnx", action="store_true", help="Export an ONNX model")
    parser.add_argument(
        "--cpp", action="store_true", help="Export a CPP header for hard-coding a model"
    )
    parser.add_argument(
        "--snapshot",
        "-s",
        action="store_true",
        help="Computes an example input-output pair for the model for debugging "
        "purposes",
    )

    main(parser.parse_args())
