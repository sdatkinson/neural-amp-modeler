from importlib.metadata import PackageNotFoundError, version

import pytest


_COMPROMISED_PYTORCH_LIGHTNING_VERSIONS = {"2.6.2", "2.6.3"}


def pytest_configure(config):
    """
    Block test execution if a known-compromised PyTorch Lightning version is installed.
    """
    try:
        pl_version = version("pytorch_lightning")
    except PackageNotFoundError:
        return

    if pl_version in _COMPROMISED_PYTORCH_LIGHTNING_VERSIONS:
        pytest.exit(
            "Blocked compromised pytorch_lightning version "
            f"{pl_version}. Uninstall immediately and upgrade.",
            returncode=2,
        )
