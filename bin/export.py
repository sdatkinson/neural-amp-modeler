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


class Dummy(torch.nn.Module):
    def forward(self, x):
        return x[8191:]


def main(args):
    with open(args.model_config_path, "r") as fp:
        net = Model.load_from_checkpoint(
            args.checkpoint, **Model.parse_config(json.load(fp))
        ).net
    net.eval()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    net.export(outdir)
    net.export_cpp_header(Path(outdir, "HardCodedModel.h"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_config_path", type=str)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("outdir")
    main(parser.parse_args())
