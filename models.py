# File: models.py
# File Created: Sunday, 30th December 2018 9:42:29 pm
# Author: Steven Atkinson (steven@atkinson.mn)

import abc
import json
import os
from tempfile import mkdtemp

import numpy as np
import torch
import torch.nn as nn


def from_json(f):
    if isinstance(f, str):
        with open(f, "r") as json_file:
            f = json.load(json_file)

    if f["type"] == "FullyConnected":
        return mlp(f["input_length"], 1, layer_sizes=f["layer_sizes"])
    else:
        raise NotImplementedError("Model type {} unrecognized".format(f["type"]))


class Model(nn.Module):
    """
    Model parent class
    """

    @abc.abstractmethod
    def predict_sequence(self, x):
        raise NotImplementedError()

    def _build(self):
        self.prediction = self._build_prediction()
        self.loss = self._build_loss()

        # Launch the session
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())

    def _build_loss(self):
        self.total_prediction_loss = tf.losses.mean_squared_error(
            self.target, self.prediction, weights=self.n_train
        )

        # Don't count this as a loss!
        self.rmse = tf.sqrt(self.total_prediction_loss / self.n_train)

        return tf.losses.get_total_loss()

    @abc.abstractclassmethod
    def _build_prediction(self):
        raise NotImplementedError("Implement prediction for model")


class Autoregressive(Model):
    """
    Autoregressive models that take in a few of the most recent input samples
    and predict the output at the last time point.
    """

    @abc.abstractproperty
    def input_length(self):
        raise NotImplementedError()

    def predict_sequence(self, x: torch.Tensor, batch_size=None, verbose=False):
        """
        Return 1D array of predictions same length as x
        """
        n = x.numel()
        batch_size = batch_size or n
        # Pad x with leading zeros:
        x = torch.cat((torch.zeros(self.input_length - 1).to(x.device), x))
        i = 0
        y = []
        while i < n:
            this_batch_size = np.min([batch_size, n - i])
            # Reshape into a batch:
            x_mtx = torch.stack(
                [x[j : j + self.input_length] for j in range(i, i + this_batch_size)]
            )
            # Predict and flatten.
            y.append(self(x_mtx).squeeze())
            i += this_batch_size
        return torch.cat(y)


class FullyConnected(Autoregressive):
    """
    Autoregressive model taking in a sequence of the most recent inputs, putting
    them through a series of FC layers, and outputting the single output at the
    last time step.
    """

    def __init__(self, net):
        super().__init__()
        self._net = net

    @property
    def input_length(self):
        return self._net[0][0].weight.data.shape[1]

    def forward(self, inputs):
        return self._net(inputs)


def mlp(dx, dy, layer_sizes=None):
    def block(dx, dy, Activation=nn.ReLU):
        return nn.Sequential(nn.Linear(dx, dy), Activation())

    layer_sizes = [256, 256] if layer_sizes is None else layer_sizes

    net = nn.Sequential()
    in_features = dx
    for i, out_features in enumerate(layer_sizes):
        net.add_module("layer_%i" % i, block(in_features, out_features))
        in_features = out_features
    net.add_module("head", block(in_features, dy, Activation=nn.Tanh))
    return FullyConnected(net)
