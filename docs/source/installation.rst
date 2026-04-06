.. _installation:

Local Installation
==================

Step 1: Install uv
^^^^^^^^^^^^^^^^^^

This is a Python package, and it depends on other packages to work. To manage
all this, it's recommended to use uv. Install it from
https://github.com/astral-sh/uv

Step 2: Install NAM
^^^^^^^^^^^^^^^^^^^

Install NAM with uv:

.. code-block:: console

   $ uv pip install neural-amp-modeler

Or, for the latest development version:

.. code-block:: console

   $ uv pip install -e .

To update an existing installation:

.. code-block:: console

   uv pip install --upgrade neural-amp-modeler

Local development installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you're interested in developing this package, use uv to install the
development dependencies:

.. code-block:: console

   $ uv sync --extra dev

GPU Support (NVIDIA)
^^^^^^^^^^^^^^^^^^^^

If your computer has an NVIDIA GPU, you can install a GPU-compatible version
of PyTorch by adding the ``gpu`` extra:

.. code-block:: console

   $ uv sync --extra gpu

Or for development with GPU support:

.. code-block:: console

   $ uv sync --extra gpu --extra dev

The ``gpu`` extra will automatically install PyTorch with CUDA support from
PyTorch's index. For other GPU configurations (e.g., Apple Silicon MPS), the
standard ``torch`` package works automatically.


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
