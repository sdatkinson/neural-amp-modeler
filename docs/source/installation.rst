.. _installation:

Local Installation
==================

Step 1: Get Miniconda
^^^^^^^^^^^^^^^^^^^^^

This is a Python package, and it depends on other packages to work. To manage 
all this, it's recommended to use Miniconda. Get it from 
https://docs.anaconda.com/miniconda/

Step 2: Install NAM
^^^^^^^^^^^^^^^^^^^

Now that we have Miniconda, we can install NAM using it.

(Windows / Linux users) If your computer has an nVIDIA GPU, you should install a
GPU-compatible version of PyTorch first:

.. code-block:: console

   $ conda install -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

Finally, install NAM using pip:

.. code-block:: console

   $ pip install neural-amp-modeler

To update an existing installation:

.. code-block:: console

   pip install --upgrade neural-amp-modeler

Local development installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you're interested in developing this package, there are Anaconda environment
definitions included in the ``environments/`` directory. Use the one that's
appropriate for the platform you're developing on. The
``.github/workflows/python-pckage.yml`` is also helpful if you want to be sure
that you're testing your developments in the same way that contributions will be
automatically tested (via GitHub Actions).
