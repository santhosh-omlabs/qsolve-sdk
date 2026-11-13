import os

from .aer import AerBackend
from .base import AsyncBackend, Backend, ProviderJobFailed, SampleResult

__all__ = [
    "AerBackend",
    "AsyncBackend",
    "Backend",
    "ProviderJobFailed",
    "SampleResult",
    "get_backend",
]


def get_backend(target: str) -> Backend:
    """Resolve a target string to a backend.

    On the platform (QSOLVE_EXECUTOR_ADDR set) the workflow only ever gets the credential-free
    remote backend; the executor decides the real device. Locally, simulators work out of the box
    and `ibm:<device>` uses the developer's own token. `fake:<device>[@scale]` rehearses a device
    on its published noise model, optionally with every error rate multiplied by `scale`.

    Raises:
        ValueError: the target is unknown, or `fake:` has a malformed device or scale.
    """
    if os.environ.get("QSOLVE_EXECUTOR_ADDR"):
        from .remote import RemoteBackend

        return RemoteBackend()
    if target == "sim-cpu":
        return AerBackend(gpu=False)
    if target == "sim-gpu":
        return AerBackend(gpu=True)
    if target.startswith("ibm:"):
        from .ibm import IBMBackend

        return IBMBackend(device=target.split(":", 1)[1])
    if target.startswith("fake:"):
        from .fake import FakeHardwareBackend

        device, _, scale = target.split(":", 1)[1].partition("@")
        try:
            factor = float(scale or 1.0)
        except ValueError:
            raise ValueError(
                f"bad noise scale {scale!r} in {target!r}: use fake:<device>@1.6"
            ) from None
        return FakeHardwareBackend(device, factor)
    raise ValueError(
        f"Unknown target {target!r}. Use sim-cpu, sim-gpu, ibm:<device> or fake:<device>[@scale]."
    )
