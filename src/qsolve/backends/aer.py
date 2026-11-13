"""Local Aer simulator backend (CPU, or GPU where qiskit-aer-gpu is installed)."""

from __future__ import annotations

import time

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from ..sim import simulator_threads
from .base import Backend, SampleResult

_DIRECTIVES = {"barrier", "measure"}


class AerBackend(Backend):
    """Local Qiskit Aer simulator: target `sim-cpu`, or `sim-gpu` with `qiskit-aer-gpu` installed.

    Noise-free, so every run measures an ideal device. Results are reproducible for a given seed.
    """

    provider = "aer"

    def __init__(self, gpu: bool = False) -> None:
        """Create the simulator.

        Args:
            gpu: Run on the GPU (needs `qiskit-aer-gpu`) instead of the CPU. Sets `target` to
                `sim-gpu` and `device` to `aer-gpu` (otherwise `sim-cpu` and `aer-cpu`).
        """
        self.gpu = gpu
        self.target = "sim-gpu" if gpu else "sim-cpu"
        self.device = "aer-gpu" if gpu else "aer-cpu"
        self._sim = AerSimulator(device="GPU" if gpu else "CPU")
        self._native = set(self._sim.target.operation_names) | _DIRECTIVES

    def runnable(self, circuit: QuantumCircuit, seed: int = 0) -> QuantumCircuit:
        """The circuit itself when Aer supports every gate in it (transpiling would cost far more
        than simulating a small circuit), else its transpiled form."""
        if all(i.operation.name in self._native for i in circuit.data):
            return circuit
        return transpile(circuit, self._sim, seed_transpiler=seed)

    def sample(self, circuit: QuantumCircuit, shots: int, seed: int) -> SampleResult:
        """Simulate one circuit and return its counts.

        Args:
            circuit: A measured circuit. Transpiled first only if it holds a gate Aer lacks.
            shots: Number of shots.
            seed: Simulator seed; the same seed gives the same counts.

        Returns:
            A `SampleResult` with the counts and the wall-clock time taken.
        """
        t0 = time.perf_counter()
        tqc = self.runnable(circuit, seed)
        with simulator_threads():
            counts = self._sim.run(tqc, shots=shots, seed_simulator=seed).result().get_counts()
        return SampleResult(
            counts={k: int(v) for k, v in counts.items()},
            shots=shots,
            exec_time_s=time.perf_counter() - t0,
        )

    @property
    def max_qubits(self) -> int:
        """Largest circuit this simulator accepts: 30 qubits on GPU, 26 on CPU."""
        return 30 if self.gpu else 26
