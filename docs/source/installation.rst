.. _installation:

Installing NAM
===============

NAM is distributed as a Python package, and the recommended version is ``3.14``.
But first you need and appropiate version of ``pytorch`` installed, and to choose between two different environments depending on your operating system and hardware setup:

(Windows / Linux users) If your computer has an nVIDIA GPU, you should install a GPU-compatible version of PyTorch first.
`The PyTorch website <https://pytorch.org/get-started/locally/>`_ will always
have the most up-to-date guidance for this. Currently, this is the command:

.. code-block:: console

   $ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

Then, install NAM using ``pip`` and selecting the ``gpu`` variant:

.. code-block:: console

   $ pip install "neural-amp-modeler[gpu]"

For other scenarios, including macOS (CPU/MPS), use the ``cpu`` variant:

.. code-block:: console

   $ pip install torch torchvision torchaudio
   $ pip install "neural-amp-modeler[cpu]"

To update an existing installation:

.. code-block:: console

   $ pip install --upgrade "neural-amp-modeler[gpu]"  # (or)
   $ pip install --upgrade "neural-amp-modeler[cpu]"

Local development installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you're interested in developing this package, this project uses ``mise`` and ``uv`` for the local setup:

`Mise <https://mise.jdx.dev/>`_ is used to bootstrap the tooling. It installs a local version of Python, isolated from your system installation, and `uv <https://docs.astral.sh/uv/>`_, which is an extremly fast Python package and project manager.

You might need to check if ``openssl``, ``pkg-config``, and ``tcl-tk`` are installed on your system. Then install ``mise``, e.g. in macOS:

.. code-block:: console

   $ brew install openssl tcl-tk@8 pkg-config mise

These are needed to compile Python with optional modules needed by NAM.
Once done, clone and navigate to the repository, then run:

.. code-block:: console

   $ mise install --verbose

It will automatically set a virtual environment for you with all the tooling. By default ``uv`` performs an editable local install of the package, meaning you can run your changes without having to repackage each time.

Perform the first project sync as below. This will install all dependencies. Repeat
it after dependency changes, selecting exactly one accelerator extra:

.. code-block:: console

   $ uv sync --extra gpu   # (or)
   $ uv sync --extra cpu

The ``dev`` dependency group is included by default and contains the test, lint,
notebook, and pre-commit tools. When running commands, pass the same extra so
that a fresh or resynchronized environment gets the correct PyTorch variant:

.. code-block:: console

   $ uv run --extra gpu pytest  # (or)
   $ uv run --extra cpu pytest

The ``.github/workflows/python-package.yml`` is also helpful if you want to be
sure that you're testing your developments in the same way that contributions
will be automatically tested via GitHub Actions.


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
