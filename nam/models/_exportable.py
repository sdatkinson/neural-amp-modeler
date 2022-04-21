# File: _exportable.py
# Created Date: Tuesday February 8th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

import abc
from pathlib import Path


class Exportable(abc.ABC):
    """
    Interface for my custon export format for use in the plugin.
    """

    @abc.abstractmethod
    def export(self, outdir: Path):
        """
        Interface for exporting.
        You should create at least a `config.json` containing the two fields:
        * "version" (str)
        * "architecture" (str)
        * "config": (dict w/ other necessary data like tensor shapes etc)

        :param outdir: Assumed to exist. Can be edited inside at will.
        """
        pass

    @abc.abstractmethod
    def export_cpp_header(self, filename: Path):
        """
        Export a .h file to compile into the plugin with the weights written right out
        as text
        """
        pass
