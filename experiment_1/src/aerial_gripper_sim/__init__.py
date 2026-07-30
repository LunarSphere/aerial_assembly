"""Actuatorless aerial gripper mechanics simulation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aerial-gripper-sim")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
