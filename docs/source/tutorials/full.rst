Training locally with the full-featured NAM
===========================================

The command line trainer is the full-featured option for training models with 
NAM. To start, you'll want to follow the installation instructions here at
:ref:`installation`.

After completing this, you will be able to use the full-featured NAM trainer by
typing

.. code-block:: console

    $ nam-full

from the command line.

Training
--------

Training uses three configuration files to specify:

1. What data you're training with: (``nam_full_configs/data/``),
2. What model architecture you're using (``nam_full_configs/models/``), and
3. Details of the learning algorithm model (``nam_full_configs/learning/``).

To train a model of your own gear, you'll need to have a paired input/output
signal from it (either by reamping a pre-recorded test signal or by
simultaneously recording your DI and the effected tone). For your first time, 
you can download the following pre-made files:

* `input.wav <https://drive.google.com/file/d/1KbaS4oXXNEuh2aCPLwKrPdf5KFOjda8G/view?usp=sharing>`_ 
* `output.wav <https://drive.google.com/file/d/1NrpQLBbCDHyu0RPsne4YcjIpi5-rEP6w/view?usp=sharing>`_ 

Next, make a file called e.g. ``data.json`` by copying
`nam_full_configs/data/single_pair.json <https://github.com/sdatkinson/neural-amp-modeler/blob/main/nam_full_configs/data/single_pair.json>`_
and editing it to point to your audio files like this: 

.. code-block:: json

    "common": {
        "x_path": "C:\\path\\to\\input.wav",
        "y_path": "C:\\path\\to\\output.wav",
        "delay": 0
    }

.. note:: If you're providing your own audio files, then you need to provide 
    the latency (in samples) between the input and output file. A positive 
    number of samples means that the output lags the input by the provided 
    number of samples; a negative value means that the output `precedes` the 
    input (e.g. because your DAW over-compensated). If you're not sure exactly 
    how much latency there is, it's usually a good idea to add a few samples 
    just so that the model doesn't need to "predict the future"!

Next, copy to e.g. ``model.json`` a file for whicever model architecture you want to
use (e.g. 
`nam_full_configs/models/wavenet.json <https://github.com/sdatkinson/neural-amp-modeler/blob/main/nam_full_configs/models/wavenet.json>`_ 
for the standard WaveNet from the simplified trainers), and copy to e.g. 
``learning.json`` the contents of 
`nam_full_configs/learning/demo.json <https://github.com/sdatkinson/neural-amp-modeler/blob/main/nam_full_configs/learning/demo.json>`_
(for a quick demo run) or
`default.json <https://github.com/sdatkinson/neural-amp-modeler/blob/main/nam_full_configs/learning/default.json>`_
(for something more like a normal use case).

Next, to train, open up a terminal. Activate your ``nam`` environment and call
the training script with

.. code-block:: console

    nam-full \
    path/to/data.json \
    path/to/model.json \
    path/to/learning.json \
    path/to/outputs

where the first three input paths are where you saved for files, and you choose
the final output path to save your training results where you'd like.

.. note:: NAM uses 
    `PyTorch Lightning <https://lightning.ai/pages/open-source/>`_
    under the hood as a modeling framework, and you can control many of the 
    PyTorch Lightning configuration options from 
    ``nam_full_configs/learning/default.json``.

Once training is done, a file called ``model.nam`` is created in the output 
directory. To use it, point 
`the plugin <https://github.com/sdatkinson/NeuralAmpModelerPlugin>`_ at the file
and you're good to go!
