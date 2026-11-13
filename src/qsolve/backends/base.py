"""Backend abstraction. Workflows never touch providers directly."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from qiskit import QuantumCircuit

MAX_BATCH = 300  # circuits per batch: provider job limits and the executor's message size


@dataclass
class SampleResult:
    """What a backend returns for one circuit.

    Attributes:
        counts: Bit string to count, in Qiskit order (qubit 0 is the rightmost character).
        shots: Shots the counts add up to.
        exec_time_s: Seconds the backend spent on it (0.0 when not measured).
        metadata: Backend-specific facts, for example `readout` errors per measured bit, `job_id`
            and `two_qubit_gates`. The bridge adds `dispatch`, the recorded circuit's index.
    """

    counts: dict[str, int]  # qiskit bitstrings (little-endian: qubit 0 is rightmost)
    shots: int
    exec_time_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Backend(ABC):
    """Executes measured circuits and returns counts."""

    target: str  # sim-cpu | sim-gpu | qpu
    provider: str
    device: str

    @abstractmethod
    def sample(self, circuit: QuantumCircuit, shots: int, seed: int) -> SampleResult:
        """Run one measured circuit.

        Args:
            circuit: The circuit, with measurements.
            shots: Number of shots.
            seed: Seed for simulators; hardware ignores it.

        Returns:
            The counts and metadata for this circuit.
        """
        ...

    def sample_batch(
        self, circuits: Sequence[QuantumCircuit], shots: int, seeds: Sequence[int]
    ) -> list[SampleResult]:
        """Run several independent circuits, each with its own seed. Backends that can send them
        as one provider job (one hardware queue wait, one per-job overhead); this default runs
        them one by one, so a simulator gives exactly the counts of separate calls."""
        if len(circuits) != len(seeds):
            raise ValueError("one seed per circuit")
        return [self.sample(c, shots, s) for c, s in zip(circuits, seeds, strict=True)]

    def calibration_snapshot(self) -> dict[str, Any]:
        """Device facts recorded in the run's provenance. The base class returns an empty dict."""
        return {}

    @property
    def max_qubits(self) -> int:
        """Largest circuit (in qubits) this backend accepts. The base class says 30."""
        return 30


def sample_all(
    backend: Backend, circuits: Sequence[QuantumCircuit], shots: int, seeds: Sequence[int]
) -> list[SampleResult]:
    """`backend.sample_batch` in chunks of MAX_BATCH. Workflows use this for circuits that do not
    depend on each other's results (measurement groups, kernel entries)."""
    out: list[SampleResult] = []
    for i in range(0, len(circuits), MAX_BATCH):
        out += backend.sample_batch(circuits[i : i + MAX_BATCH], shots, seeds[i : i + MAX_BATCH])
    return out


class ProviderJobFailed(RuntimeError):
    """The provider finished the job without a result (failed or cancelled on their side).
    Resubmitting is the right response; any other exception may be transient."""


class AsyncBackend(Backend):
    """A backend whose jobs live at the provider and can be collected later by id (QPUs).

    The platform records the job id between `submit` and `collect`, so a run interrupted while
    waiting in a hardware queue re-attaches to the same job instead of paying for it twice.
    """

    @abstractmethod
    def submit(self, circuit: QuantumCircuit, shots: int, seed: int) -> str:
        """Send the job and return the provider's job id without waiting."""

    @abstractmethod
    def collect(self, job_id: str) -> SampleResult:
        """Wait for a submitted job and return its result."""

    def cancel(self, job_id: str) -> None:
        """Cancel a job still waiting at the provider (its run ended without needing it)."""

    batches = False  # True when submit_batch/collect_batch send several circuits as one job

    def submit_batch(
        self, circuits: Sequence[QuantumCircuit], shots: int, seeds: Sequence[int]
    ) -> str:
        """Send several circuits as one provider job; return its id without waiting."""
        raise NotImplementedError

    def collect_batch(self, job_id: str) -> list[SampleResult]:
        """Wait for a batch job; results in submission order."""
        raise NotImplementedError

    def sample(self, circuit: QuantumCircuit, shots: int, seed: int) -> SampleResult:
        """Submit one circuit and wait for it: `collect(submit(...))`."""
        return self.collect(self.submit(circuit, shots, seed))

    def sample_batch(
        self, circuits: Sequence[QuantumCircuit], shots: int, seeds: Sequence[int]
    ) -> list[SampleResult]:
        """Run several circuits: one provider job when `batches` is true, else one by one.

        Raises:
            ValueError: `circuits` and `seeds` differ in length.
        """
        if not self.batches:
            return super().sample_batch(circuits, shots, seeds)
        if len(circuits) != len(seeds):
            raise ValueError("one seed per circuit")
        return self.collect_batch(self.submit_batch(circuits, shots, seeds))
