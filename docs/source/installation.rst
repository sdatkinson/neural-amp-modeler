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
GPU-compatible version of PyTorch first. 
`The PyTorch website <https://pytorch.org/get-started/locally/>`_ will always
have the most up-to-date guidance for this. Currently, this is the command:

.. code-block:: console
   $ pip install -r requirements-gpu.txt

Then, install NAM using pip:

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


Trouble using the GPU?
^^^^^^^^^^^^^^^^^^^^^^

If you're using a Windows or Linux machine with an NVIDIA GPU and NAM isn't
using it (Apple machines with Apple Silicon don't use an nVIDIA GPU, but MPS, an 
accelerator with somewhat similar functionality), the reason is 99.999% probably
an issue with your PyTorch installation, not NAM. Google (or ChatGPT) should be 
able to help you fix the issue, but here are a few handy things you can do (in 
case you're not familiar with Python):

To check if PyTorch can see the GPU, you can do:

.. code-block:: console

   $ python -c "import torch; print(torch.cuda.is_available())"

If this prints ``True``, then PyTorch can see the GPU. If it prints ``False``, 
then PyTorch can't see the GPU and you need to fix your PyTorch installation.

To check whether you've installed a version of PyTorch that supports the GPU,
you can do:

.. code-block:: console

   $ python -c "import torch; print(torch.__version__)"

If this prints a version of PyTorch that includes ``cu`` in the version string, 
then PyTorch can see the GPU. If it doesn't, then you need to fix your PyTorch 
installation.

To uninstall PyTorch and reinstall it, you can do:

.. code-block:: console

   $ pip uninstall torch torchvision torchaudio

and then use the install command above (or check the PyTorch website for the
most up-to-date instructions). If you notice that this documentation is out of 
date, please let us know so we can update it (or even better, make a PR!)
