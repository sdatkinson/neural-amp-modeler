# File: setup.py
# Created Date: 2020-04-08
# Author: Steven Atkinson (steven@atkinson.mn)

from distutils.util import convert_path
from setuptools import setup, find_packages


def get_additional_requirements():
    additional_requirements = []
    # Issue 294
    try:
        import transformers

        # This may not be unnecessarily straict a requirement, but I'd rather
        # fix this promptly than leave a chance that it wouldn't be fixed
        # properly.
        additional_requirements.append("transformers>=4")
    except ModuleNotFoundError:
        pass

    # Issue 494
    def get_numpy_requirement() -> str:
        need_numpy_1 = True  # Until proven otherwise
        try:
            import torch

            version_split = torch.__version__.split(".")
            major = int(version_split[0])
            if major >= 2:
                minor = int(version_split[1])
                if minor >= 3:  # Hooray, PyTorch 2.3+!
                    need_numpy_1 = False
        except ModuleNotFoundError:
            # Until I see PyTorch 2.3 come out:
            pass
        return "numpy<2" if need_numpy_1 else "numpy"

    additional_requirements.append(get_numpy_requirement())

    return additional_requirements


main_ns = {}
ver_path = convert_path("nam/_version.py")
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

requirements = [
    "auraloss==0.3.0",
    "matplotlib",
    "pydantic>=2.0.0",
    "pytorch_lightning",
    "scipy",
    "sounddevice",
    "tensorboard",
    "torch",
    "transformers>=4",  # Issue-294
    "tqdm",
    "wavio>=0.0.5",  # Breaking change with older versions
]
requirements.extend(get_additional_requirements())

setup(
    name="neural-amp-modeler",
    version=main_ns["__version__"],
    description="Neural amp modeler",
    author="Steven Atkinson",
    author_email="steven@atkinson.mn",
    url="https://github.com/sdatkinson/",
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "nam = nam.cli:nam_gui",  # GUI trainer
            "nam-full = nam.cli:nam_full",  # Full-featured trainer
        ]
    },
)
