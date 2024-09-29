.. _installation:

Local Installation
==================

It's recommended to use Anaconda to manage your install. Get Anaconda from
https://www.anaconda.com/download

If your computer has an nVIDIA GPU, you should install a GPU-compatible version 
of PyTorch first:

.. code-block:: console

   $ conda install -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

Next, install NAM using pip:

.. code-block:: console

   $ pip install neural-amp-modeler

To update an existing installation:

.. code-block:: console

   pip install --upgrade neural-amp-modeler
