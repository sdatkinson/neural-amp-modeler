"""
Reamp a .wav file

Assumes 24-bit WAV files
"""

from argparse import ArgumentParser
import tensorflow as tf
import os
import wavio
import matplotlib.pyplot as plt

import models


def _sampwidth_to_bits(x):
    return {2: 16, 3: 24, 4: 32}[x]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("architecture", type=str, 
        help="JSON filename containing NN architecture")
    parser.add_argument("checkpoint_dir", type=str, 
        help="directory holding model checkpoint to use")
    parser.add_argument("input_file", type=str, 
        help="Input .wav file to convert")
    parser.add_argument("--output_file", type=str, default=None,
        help="Where to save the output")
    parser.add_argument("--batch_size", type=int, default=8192,
        help="How many samples to process at a time.  " + 
        "Reduce if there are out-of-memory issues.")
    parser.add_argument("--target_file", type=str, default=None,
        help=".wav file of the true output (if you want to compare)")
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = args.input_file.rstrip(".wav") + "_reamped.wav"
    
    if os.path.isfile(args.output_file):
        print("Output file exists; skip")
        exit(1)
    
    x = wavio.read(args.input_file)
    rate, sampwidth = x.rate, x.sampwidth
    bits = _sampwidth_to_bits(sampwidth)
    x_data = x.data.flatten() / 2 ** (bits - 1)
    
    with tf.Session() as sess:
        model = models.from_json(args.architecture, 
            checkpoint_path=args.checkpoint_dir)
        model.load()
        y = model.predict(x_data, batch_size=args.batch_size, verbose=True)
    wavio.write(args.output_file, y * 2 ** (bits - 1), rate, scale="none", 
        sampwidth=sampwidth)
    
    if args.target_file is not None and os.path.isfile(args.target_file):
        t = wavio.read(args.target_file)
        t_data = t.data.flatten() / 2 ** (_sampwidth_to_bits(t.sampwidth) - 1)
        plt.figure()
        plt.plot(x_data)
        plt.plot(t_data)
        plt.plot(y)
        plt.legend(["Input", "Target", "Prediction"])
        plt.show()
    