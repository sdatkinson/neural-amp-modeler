Training locally from the command line
======================================

The command line trainer is the full-featured option for training models with 
NAM.

Installation
------------

Currently, you'll want to clone the source repo to train from the command line.

Installation uses [Anaconda](https://www.anaconda.com/) for package management.

For computers with a CUDA-capable GPU (recommended):

.. code-block:: console

    conda env create -f environment_gpu.yml

.. note:: You may need to modify the CUDA version if your GPU is older. Have a 
    look at 
    `nVIDIA's documentation <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions>`_` 
    if you're not sure._

Otherwise, for a CPU-only install (will train much more slowly):

.. code-block:: console

    conda env create -f environment_cpu.yml

.. note:: If Anaconda takes a long time "`Solving environment...`", then you can
    speed up installing the environment by using the mamba experimental sovler
    with ``--experimental-solver=libmamba``.

Then activate the environment you've created with

.. code-block:: console

    conda activate nam

Training
--------

Since the command-line trainer is intended for maximum flexibiility, you can 
train from any input/output pair of reamp files you want. However, if you want
to skip the reamping and use some pre-made files for your first time, you can
download these files:

* `v1_1_1.wav <https://drive.google.com/file/d/1CMj2uv_x8GIs-3X1reo7squHOVfkOa6s/view?usp=drive_link>`_ 
  (input)
* `output.wav <https://drive.google.com/file/d/1e0pDzsWgtqBU87NGqa-4FbriDCkccg3q/view?usp=drive_link>`_ 
  (output)

Next, edit ``bin/train/data/single_pair.json`` to point to relevant audio files: 

.. code-block:: json

    "common": {
        "x_path": "C:\\path\\to\\v1_1_1.wav",
        "y_path": "C:\\path\\to\\output.wav",
        "delay": 0
    }

.. note:: If you're provideding your own audio files, then you need to provide 
    the latency (in samples) between the input and output file. A positive 
    number of samples means that the output lags the input by the provided 
    number of samples; a negative value means that the output `precedes` the 
    input (e.g. because your DAW over-compensated). If you're not sure exactly 
    how much latency there is, it's usually a good idea to add a few samples 
    just so that the model doesn't need to predict the future!

Next, to train, open up a terminal. Activate your nam environment and call the 
training with

.. code-block:: console

    python bin/train/main.py \
    bin/train/inputs/data/single_pair.json \
    bin/train/inputs/models/demonet.json \
    bin/train/inputs/learning/demo.json \
    bin/train/outputs/MyAmp

* ``data/single_pair.json`` contains the information about the data you're 
  training on.   
* ``models/demonet.json`` contains information about the model architecture that
  is being trained. The example used here uses a `feather` configured `wavenet`.  
* ``learning/demo.json`` contains information about the training run itself 
  (e.g. number of epochs).

The configuration above runs a short (demo) training. For a real training you 
may prefer to run something like:

.. code-block:: console

    python bin/train/main.py \
    bin/train/inputs/data/single_pair.json \
    bin/train/inputs/models/wavenet.json \
    bin/train/inputs/learning/default.json \
    bin/train/outputs/MyAmp

.. note:: NAM uses 
    `PyTorch Lightning <https://lightning.ai/pages/open-source/>`_
    under the hood as a modeling framework, and you can control many of the 
    PyTorch Lightning configuration options from 
    ``bin/train/inputs/learning/default.json``.

Once training is done, a file called ``model.nam`` is created in the output 
directory. To use it, point 
`the plugin <https://github.com/sdatkinson/NeuralAmpModelerPlugin>`_ at the file
and you're good to go!
