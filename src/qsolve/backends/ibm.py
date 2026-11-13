"""IBM Quantum backend (qiskit-ibm-runtime). Credentials come from the environment,
never from workflow code: QSOLVE_IBM_TOKEN, optional QSOLVE_IBM_INSTANCE."""

from __future__ import annotations

import contextlib
import math
import os
import time
from collections.abc import Sequence
from typing import Any

from qiskit import QuantumCircuit

from .base import AsyncBackend, ProviderJobFailed, SampleResult
from .hardware import device_metadata, readout_errors, transpile_best

_TERMINAL_FAILURES = {"ERROR", "CANCELLED"}

# Billed quantum time per job and per shot: an empirical fit to ibm_fez jobs. IBM can change its
# billing, so re-fit these if estimates drift from the bill.
S_PER_JOB = 2.4
S_PER_SHOT = 0.00023


def max_execution_seconds(n_circuits: int, shots: int) -> int:
    """IBM-side cap on a job's quantum time: three times the expected bill, plus 10 s. A job that
    somehow needs more is stopped by IBM instead of draining the account."""
    return math.ceil(10 + 3 * (S_PER_JOB + S_PER_SHOT * n_circuits * shots))


class IBMBackend(AsyncBackend):
    """IBM Quantum hardware through `qiskit-ibm-runtime` (target `ibm:<device>`; `ibm` extra).

    Jobs are collected by id, so an interrupted run can re-attach to its job. Dynamical
    decoupling and gate twirling are switched on; measurement twirling is off.
    """

    target = "qpu"
    provider = "ibm"

    def __init__(self, device: str, token: str | None = None, instance: str | None = None) -> None:
        """Connect to IBM and look up the device.

        Args:
            device: Device name, for example `ibm_fez`.
            token: IBM API token; defaults to the `QSOLVE_IBM_TOKEN` environment variable.
            instance: IBM instance; defaults to `QSOLVE_IBM_INSTANCE`.

        Raises:
            RuntimeError: the `ibm` extra is missing, or no token is available.
        """
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Install the ibm extra: pip install 'qsolve-sdk[ibm]'") from e
        token = token or os.environ.get("QSOLVE_IBM_TOKEN")
        if not token:
            raise RuntimeError("QSOLVE_IBM_TOKEN is not set")
        self.device = device
        self._service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=token,
            instance=instance or os.environ.get("QSOLVE_IBM_INSTANCE"),
        )
        self._backend = self._service.backend(device)

    batches = True  # a batch is one Sampler job: one queue wait, one per-job overhead

    def submit(self, circuit: QuantumCircuit, shots: int, seed: int) -> str:
        """Send one circuit as a one-circuit batch and return the IBM job id without waiting."""
        return self.submit_batch([circuit], shots, [seed])

    def collect(self, job_id: str) -> SampleResult:
        """Wait for a job made with `submit` and return its single result."""
        return self.collect_batch(job_id)[0]

    def cancel(self, job_id: str) -> None:
        """Cancel a job that is still waiting at IBM. Finished jobs are left alone."""
        job = self._service.job(job_id)
        status = str(getattr(job.status(), "name", job.status())).upper()
        if status not in ("DONE", *_TERMINAL_FAILURES):  # finished jobs cannot be cancelled
            job.cancel()

    def submit_batch(
        self, circuits: Sequence[QuantumCircuit], shots: int, seeds: Sequence[int]
    ) -> str:
        """Transpile the circuits (cheapest of several tries) and send them as one Sampler job.

        Returns:
            The IBM job id. Its quantum time is capped at three times the expected bill plus 10 s.
        """
        from qiskit_ibm_runtime import SamplerV2

        sampler = SamplerV2(mode=self._backend)
        # idle qubits are refocused, and gate errors are randomised into unbiased noise (what
        # extrapolation-based mitigation assumes); measurement is not twirled, so the per-qubit
        # readout errors stay as calibrated and can be corrected from `readout` below
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.enable_measure = False
        sampler.options.max_execution_time = max_execution_seconds(len(circuits), shots)
        job = sampler.run(transpile_best(circuits, self._backend, seeds[0]), shots=shots)
        return str(job.job_id())

    def collect_batch(self, job_id: str) -> list[SampleResult]:
        """Wait for a batch job.

        Returns:
            One `SampleResult` per circuit, in submission order. Metadata holds `job_id`,
            `batch_index`, `batch_size`, `quantum_seconds` (first circuit only) and `readout`.

        Raises:
            ProviderJobFailed: IBM ended the job as ERROR or CANCELLED.
        """
        t0 = time.perf_counter()
        job = self._service.job(job_id)
        try:
            pubs = list(job.result())
        except Exception as e:
            status = str(getattr(job.status(), "name", job.status())).upper()
            if status in _TERMINAL_FAILURES:
                raise ProviderJobFailed(f"IBM job {job_id} ended with status {status}") from e
            raise
        usage: Any = None
        with contextlib.suppress(Exception):  # usage reporting is best-effort
            usage = job.usage()
        sent = _sent_circuits(job, len(pubs))
        properties: Any = None
        with contextlib.suppress(Exception):  # calibration at the time the job ran
            properties = job.properties()
        wall = time.perf_counter() - t0
        device = device_metadata(sent) if sent is not None else [{} for _ in pubs]
        out = []
        for i, pub in enumerate(pubs):
            creg = next(iter(pub.data.keys()))
            counts = getattr(pub.data, creg).get_counts()
            meta: dict[str, Any] = {
                "job_id": job_id,
                "batch_index": i,
                "batch_size": len(pubs),
                # the job's billed time is reported once, on its first circuit
                "quantum_seconds": usage if i == 0 else 0,
            }
            if sent is not None:
                meta["readout"] = readout_errors(sent[i], properties)
            meta |= device[i]
            out.append(
                SampleResult(
                    counts={k: int(v) for k, v in counts.items()},
                    shots=int(sum(counts.values())),
                    exec_time_s=wall / len(pubs),
                    metadata=meta,
                )
            )  # fmt: skip
        return out

    def calibration_snapshot(self) -> dict[str, Any]:
        """The device name and its qubit count."""
        cfg = self._backend.configuration()
        return {"backend": self.device, "n_qubits": cfg.n_qubits}

    @property
    def max_qubits(self) -> int:
        """Qubit count of the device."""
        return int(self._backend.configuration().n_qubits)


def _sent_circuits(job: Any, n: int) -> list[QuantumCircuit] | None:
    """The transpiled circuits a job ran, read back from IBM (so a run collected by another
    process still knows which physical qubit fed each bit); None if they cannot be read."""
    with contextlib.suppress(Exception):
        circuits = []
        for pub in job.inputs["pubs"]:
            c = pub[0] if isinstance(pub, (tuple, list)) else getattr(pub, "circuit", pub)
            circuits.append(c)
        if len(circuits) == n and all(isinstance(c, QuantumCircuit) for c in circuits):
            return circuits
    return None
