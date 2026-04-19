.. _installation:

Local Installation
==================

Step 1: Install Python
^^^^^^^^^^^^^^^^^^^^^^

Install Python 3.9 or newer.

Step 2: Create a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It's recommended to install NAM in a virtual environment.

On macOS / Linux:

.. code-block:: console

   $ python -m venv .venv
   $ source .venv/bin/activate
   $ python -m pip install --upgrade pip

On Windows, in ``cmd.exe``:

.. code-block:: console

   > python -m venv .venv
   > .venv\Scripts\activate
   > python -m pip install --upgrade pip

On Windows, in PowerShell:

.. code-block:: console

   > python -m venv .venv
   > .\.venv\Scripts\Activate.ps1
   > python -m pip install --upgrade pip

Step 3: Install PyTorch
^^^^^^^^^^^^^^^^^^^^^^^

If your computer has an NVIDIA GPU, install a GPU-compatible version of
PyTorch using the instructions on the PyTorch website:

https://pytorch.org/get-started/locally/

If you're not using an NVIDIA GPU, install the default CPU version of PyTorch:

.. code-block:: console

   $ python -m pip install torch

Step 4: Install NAM
^^^^^^^^^^^^^^^^^^^

.. code-block:: console

   $ python -m pip install neural-amp-modeler

To update an existing installation:

.. code-block:: console

   $ python -m pip install --upgrade neural-amp-modeler

Local development installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create and activate a virtual environment, then install NAM in editable mode
with its test dependencies. On Windows, use the activation command above for
your shell. Install any additional development tooling you need alongside it:

.. code-block:: console

   $ python -m venv .venv
   $ source .venv/bin/activate
   $ python -m pip install --upgrade pip
   $ python -m pip install -e ".[test]"
   $ python -m pip install flake8 black pre-commit

``.github/workflows/python-package.yml`` is also helpful if you want to be sure
that you're testing your developments in the same way that contributions will be
automatically tested (via GitHub Actions).


Trouble using the GPU?
^^^^^^^^^^^^^^^^^^^^^^

If you're using a Windows or Linux machine with an NVIDIA GPU and NAM isn't
using it (Apple machines with Apple Silicon don't use an NVIDIA GPU, but MPS, an 
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

   $ python -m pip uninstall torch torchvision torchaudio

and then use the install command above (or check the PyTorch website for the
most up-to-date instructions). If you notice that this documentation is out of 
date, please let us know so we can update it (or even better, make a PR!)
