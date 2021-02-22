# File: models.py
# File Created: Sunday, 30th December 2018 9:42:29 pm
# Author: Steven Atkinson (steven@atkinson.mn)

import numpy as np
import tensorflow as tf
import abc
from tempfile import mkdtemp
import os
import json
from tf_slim.layers import layers as _layers;


def from_json(f, n_train=1, checkpoint_path=None):
    if isinstance(f, str):
        with open(f, "r") as json_file:
            f = json.load(json_file)

    if f["type"] == "FullyConnected":
        return FullyConnected(n_train, f["input_length"],
            layer_sizes=f["layer_sizes"], checkpoint_path=checkpoint_path)
    else:
        raise NotImplementedError("Model type {} unrecognized".format(
            f["type"]))


class Model(object):
    """
    Model parent class
    """
    def __init__(self, n_train, sess=None, checkpoint_path=None):
        """
        Make sure child classes call _build() after this!
        """
        if sess is None:
            sess = tf.compat.v1.get_default_session()
        self.sess = sess

        if checkpoint_path is None:
            checkpoint_path = os.path.join(mkdtemp(), "model.ckpt")
        if not os.path.isdir(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        self.checkpoint_path = checkpoint_path

        # self._batch_size = batch_size
        self._n_train = n_train
        self.target = tf.compat.v1.placeholder(tf.float32, shape=(None, 1))
        self.prediction = None
        self.total_prediction_loss = None
        self.rmse = None
        self.loss = None  # Includes any regularization

    # @property
    # def batch_size(self):
    #     return self._batch_size

    @property
    def n_train(self):
        return self._n_train

    def load(self, checkpoint_path=None):
        checkpoint_path = checkpoint_path or self.checkpoint_path

        try:
            ckpt = tf.train.get_checkpoint_state(checkpoint_path)
            print("Loading model: {}".format(ckpt.model_checkpoint_path))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        except Exception as e:
            print("Error while attempting to load model: {}".format(e))

    @abc.abstractclassmethod
    def predict(self, x):
        """
        A nice function for prediction.
        :param x: input array (length=n)
        :type x: array-like
        :return: (array-like) corresponding predicted outputs (length=n)
        """
        raise NotImplementedError("Implement predict()")

    def save(self, iter, checkpoint_path=None):
        checkpoint_path = checkpoint_path or self.checkpoint_path
        self.saver.save(self.sess, checkpoint_path, global_step=iter)

    def _build(self):
        self.prediction = self._build_prediction()
        self.loss = self._build_loss()

        # Launch the session
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

    def _build_loss(self):
        self.total_prediction_loss = tf.compat.v1.losses.mean_squared_error(self.target,
            self.prediction, weights=self.n_train)

        # Don't count this as a loss!
        self.rmse = tf.sqrt(
            self.total_prediction_loss / self.n_train)

        return tf.compat.v1.losses.get_total_loss()

    @abc.abstractclassmethod
    def _build_prediction(self):
        raise NotImplementedError('Implement prediction for model')


class Autoregressive(Model):
    """
    Autoregressive models that take in a few of the most recent input samples
    and predict the output at the last time point.
    """
    def __init__(self, n_train, input_length, sess=None,
            checkpoint_path=None):
        super().__init__(n_train, sess=sess, checkpoint_path=checkpoint_path)
        self._input_length = input_length
        self.x = tf.compat.v1.placeholder(tf.float32, shape=(None, self.input_length))

    @property
    def input_length(self):
        return self._input_length

    def predict(self, x, batch_size=None, verbose=False):
        """
        Return 1D array of predictions same length as x
        """
        n = x.size
        batch_size = batch_size or n
        # Pad x with leading zeros:
        x = np.concatenate((np.zeros(self.input_length - 1), x))
        i = 0
        y = []
        while i < n:
            if verbose:
                print("model.predict {}/{}".format(i, n))
            this_batch_size = np.min([batch_size, n - i])
            # Reshape into a batch:
            x_mtx = np.stack([x[j: j + self.input_length]
                for j in range(i, i + this_batch_size)])
            # Predict and flatten.
            y.append(self.sess.run(self.prediction, feed_dict={self.x: x_mtx}) \
                .flatten())
            i += this_batch_size
        return np.concatenate(y)


class FullyConnected(Autoregressive):
    """
    Autoregressive model taking in a sequence of the most recent inputs, putting
    them through a series of FC layers, and outputting the single output at the
    last time step.
    """
    def __init__(self, n_train, input_length, layer_sizes=(512,),
            sess=None, checkpoint_path=None):
        super().__init__(n_train, input_length, sess=sess,
            checkpoint_path=checkpoint_path)
        self._layer_sizes = layer_sizes
        self._build()

    def _build_prediction(self):
        h = self.x
        for m in self._layer_sizes:
            h = _layers.fully_connected(h, m)
            
        y = -1.0 + 2.0 * _layers.fully_connected(h, 1,
            activation_fn=tf.nn.sigmoid)
        return y
