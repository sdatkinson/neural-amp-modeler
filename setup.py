# File: setup.py
# Created Date: 2020-04-08
# Author: Steven Atkinson (steven@atkinson.mn)

from distutils.util import convert_path
from setuptools import setup, find_packages

def get_additional_requirements():
    # Issue 294
    try:
        import transformers
        # This may not be unnecessarily straict a requirement, but I'd rather
        # fix this promptly than leave a chance that it wouldn't be fixed 
        # properly.
        return ["transformers>=4"]
    except ModuleNotFoundError:
        return []

main_ns = {}
ver_path = convert_path("nam/_version.py")
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

requirements = [
    "auraloss==0.3.0",
    "matplotlib",
    "numpy",
    "onnx",
    "onnxruntime",
    "pydantic",
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
            "nam = nam.train.gui:run",
        ]
    },
)
