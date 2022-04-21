# File: run.py
# Created Date: Sunday February 6th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Load up a model, process a WAV, and save.
"""

import json
from argparse import ArgumentParser

from nam.data import wav_to_tensor, tensor_to_wav
from nam.models import Model


def main(args):
    source = wav_to_tensor(args.source_path)
    with open(args.model_config_path, "r") as fp:
        model = Model.load_from_checkpoint(
            args.checkpoint, **Model.parse_config(json.load(fp))
        )
    model.eval()
    output = model(source)
    tensor_to_wav(output, args.outfile)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("source_path", type=str)
    parser.add_argument("model_config_path", type=str)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("outfile")
    main(parser.parse_args())
