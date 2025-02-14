# File: _version.py
# Created Date: Tuesday December 20th 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Version utility
"""

from typing import Optional as _Optional

from .._version import __version__


class IncomparableVersionError(ValueError):
    """
    Error raised when two versions can't be compared.
    """

    pass


class Version:
    def __init__(self, major: int, minor: int, patch: int, dev: _Optional[int] = None):
        self.major = major
        self.minor = minor
        self.patch = patch
        self.dev = dev
        self.dev_int = self._parse_dev_int(dev)

    @classmethod
    def from_string(cls, s: str):
        parts = s.split(".")
        if len(parts) == 3:  # e.g. "0.7.1"
            dev = None
        elif len(parts) == 4:  # e.g. "0.7.1.dev7"
            dev = parts[3]
        else:
            raise ValueError(f"Invalid version string {s}")
        major, minor, patch = [int(x) for x in parts[:3]]
        return cls(major=major, minor=minor, patch=patch, dev=dev)

    def __eq__(self, other) -> bool:
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and self.dev == other.dev
        )

    def __lt__(self, other) -> bool:
        if self == other:
            return False
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        if self.patch != other.patch:
            return self.patch < other.patch
        if self.dev != other.dev:
            # None is defined as least
            if self.dev is None and other.dev is not None:
                return True
            elif self.dev is not None and other.dev is None:
                return False
            assert self.dev is not None
            assert other.dev is not None
            if self.dev_int is None:
                raise IncomparableVersionError(
                    f"Version {str(self)} has incomparable dev version {self.dev}"
                )
            if other.dev_int is None:
                raise IncomparableVersionError(
                    f"Version {str(other)} has incomparable dev version {other.dev}"
                )
            return self.dev_int < other.dev_int
        raise RuntimeError(
            f"Unhandled comparison between versions {str(self)} and {str(other)}"
        )

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def _parse_dev_int(self, dev: _Optional[str]) -> _Optional[int]:
        """
        Turn the string into an int that can be compared if possible.
        """
        if dev is None:
            return None
        if not isinstance(dev, str):
            raise TypeError(f"Invalid dev string type {type(dev)}")
        if not dev.startswith("dev") or len(dev) <= 3:  # "misc", "dev", etc
            return None
        return int(dev.removeprefix("dev"))


PROTEUS_VERSION = Version(4, 0, 0)


def get_current_version() -> Version:
    return Version.from_string(__version__)
