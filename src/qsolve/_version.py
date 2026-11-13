"""The installed SDK version, read once from the package metadata."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("qsolve-sdk")
except PackageNotFoundError:  # pragma: no cover - running from a source tree that is not installed
    __version__ = "0.0.0+local"
