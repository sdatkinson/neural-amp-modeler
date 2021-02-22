# File: train.py
# # File Created: Sunday, 30th December 2018 2:08:54 pm
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Here's a script for training new models.
"""

from argparse import ArgumentParser
import numpy as np
import abc
import tensorflow as tf
# tf.compat.v1.Session()
import matplotlib.pyplot as plt
from time import time
import wavio
import json
import os

import models

# Parameters for training
check_training_at = np.concatenate([10 ** pwr * np.array([1, 2, 5])
    for pwr in np.arange(0, 7)])
plot_kwargs = {
    "window": (30000, 40000)
}
wav_kwargs = {"window": (0, 5 * 44100)}


class Data(object):
    """
    Object for holding data and spitting out minibatches for training/segments
    for testing.
    """
    def __init__(self, fname, input_length, batch_size=None):
        xy = np.load(fname)
        self.x = xy[0]
        self.y = xy[1]
        self._n = self.x.size - input_length + 1
        self.input_length = input_length
        self.batch_size = batch_size

    @property
    def n(self):
        return self._n

    def minibatch(self, n=None, ilist=None):
        """
        Pull a random minibatch out of the data set
        """
        if ilist is None:
            n = n or self.batch_size
            ilist = np.random.randint(0, self.n, size=(n,))
        x = np.stack([self.x[i: i + self.input_length] for i in ilist])
        y = np.array([self.y[i + self.input_length - 1] for i in ilist]) \
            [:, np.newaxis]
        return x, y

    def sequence(self, start, end):
        end += self.input_length - 1
        return self.x[start: end], self.y[start + self.input_length - 1: end]


def train(model, train_data, batch_size=None, n_minibatches=10,
        validation_data=None, plot_at=(), wav_at=(), plot_kwargs={},
        wav_kwargs={}, save_every=100, validate_every=100):
    save_dir = os.path.dirname(model.checkpoint_path)
    sess = model.sess
    opt = tf.compat.v1.train.AdamOptimizer().minimize(model.loss)
    sess.run(tf.compat.v1.global_variables_initializer())  # For opt
    t_loss_list, v_loss_list = [], []
    t0 = time()
    for i in range(n_minibatches):
        x, y = train_data.minibatch(batch_size)
        t_loss, _ = sess.run((model.rmse, opt),
            feed_dict={model.x: x, model.target: y})
        t_loss_list.append([i, t_loss])
        print("t={:7} | MB {:>7} / {:>7} | TLoss={:8}".format(int(time() - t0),
            i + 1, n_minibatches, t_loss))

        # Callbacks, basically...
        if i + 1 in plot_at:
            plot_predictions(model,
                validation_data if validation_data is not None else train_data,
                title="Minibatch {}".format(i + 1),
                fname="{}/mb_{}.png".format(save_dir, i + 1),
                **plot_kwargs)
        if i + 1 in wav_at:
            print("Making wav for mb {}".format(i + 1))
            predict(model, validation_data,
                save_wav_file="{}/predict_{}.wav".format(save_dir, i + 1),
                **wav_kwargs)
        if (i + 1) % save_every == 0:
            model.save(iter=i + 1)
        if i == 0 or (i + 1) % validate_every == 0:
            v_loss, _ = sess.run((model.rmse, opt),
                feed_dict={model.x: x, model.target: y})
            print("VLoss={:8}".format(v_loss))
            v_loss_list.append([i, v_loss])

    # After training loop...
    if validation_data is not None:
        x, y = validation_data.minibatch(train_data.batch_size)
        v_loss = sess.run(model.rmse,
            feed_dict={model.x: x, model.target: y})
        print("Validation loss={}".format(v_loss))
    return np.array(t_loss_list).T, np.array(v_loss_list).T


def plot_predictions(model, data, title=None, fname=None, window=None):
    x, y, t = predict(model, data, window=window)
    plt.figure(figsize=(12, 4))
    plt.plot(x)
    plt.plot(t)
    plt.plot(y)
    plt.legend(('Input', 'Target', 'Prediction'))
    if title is not None:
        plt.title(title)
    if fname is not None:
        print("Saving to {}...".format(fname))
        plt.savefig(fname)
        plt.close()
    else:
        plt.show()


def plot_loss(t_loss, v_loss, fname):
    plt.figure()
    plt.loglog(t_loss[0], t_loss[1])
    plt.loglog(v_loss[0], v_loss[1])
    plt.xlabel("Minibatch")
    plt.ylabel("RMSE")
    plt.legend(("Training", "Validation"))
    plt.savefig(fname)
    plt.close()


def predict(model, data, window=None, save_wav_file=None):
    x, t = data.x, data.y
    if window is not None:
        x, t = x[window[0]: window[1]], t[window[0]: window[1]]
    y = model.predict(x).flatten()

    if save_wav_file is not None:
        rate = 44100  # TODO from data
        sampwidth = 3  # 24-bit
        wavio.write(save_wav_file, y * 2 ** 23, rate, scale="none",
            sampwidth=sampwidth)

    return x, y, t


def _get_input_length(archfile):
    return json.load(open(archfile, "r"))["input_length"]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_arch", type=str,
        help="JSON containing model architecture")
    parser.add_argument("train_data", type=str,
        help="Filename for training data")
    parser.add_argument("validation_data", type=str,
        help="Filename for validation data")
    parser.add_argument("--save_dir", type=str, default=None,
        help="Where to save the run data (checkpoints, prediction...)")
    parser.add_argument("--batch_size", type=str, default=4096,
        help="Number of data per minibatch")
    parser.add_argument("--minibatches", type=int, default=10,
        help="Number of minibatches to train for")
    args = parser.parse_args()
    input_length = _get_input_length(args.model_arch)  # Ugh, kludge

    # Load the data
    train_data = Data(args.train_data, input_length,
        batch_size=args.batch_size)
    validate_data = Data(args.validation_data, input_length)

    # Training
    with tf.compat.v1.Session() as sess:
        model = models.from_json(args.model_arch, train_data.n,
            checkpoint_path=os.path.join(args.save_dir, "model.ckpt"))
        t_loss_list, v_loss_list = train(
            model,
            train_data,
            validation_data=validate_data,
            n_minibatches=args.minibatches,
            plot_at=check_training_at,
            wav_at=check_training_at,
            plot_kwargs=plot_kwargs,
            wav_kwargs=wav_kwargs)
        plot_predictions(model, validate_data, window=(0, 44100))
        print("Predict the full output")
        predict(model, validate_data,
            save_wav_file="{}/predict.wav".format(
                os.path.dirname(model.checkpoint_path)))

    plot_loss(t_loss_list, v_loss_list, "{}/loss.png".format(args.save_dir))
