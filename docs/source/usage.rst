Usage
=====

In the cloud
------------

TODO

.. _installation:

Local Installation
------------------

It's recommended to use Anaconda to manage your install. Get Anaconda from
https://www.anaconda.com/download

If your computer has an nVIDIA GPU, you should install a GPU-compatible version 
of PyTorch first:

.. code-block:: console

   $ conda install -y pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

Next, install NAM using pip:

.. code-block:: console

   $ pip install neural-amp-modeler

GUI Training (simplified)
-------------------------

After installing, type ``nam`` on the command line. You'll be greeted with a GUI
into which you can provide your input and output files, pick where your model
will be saved, add emtadata to your model, and configure advanecd settings.

TODO pictures, step by step.

CLI Training (All Features)
---------------------------

TODO
