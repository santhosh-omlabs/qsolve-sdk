"""Simulator threading and run-time prediction.

Aer updates each amplitude of a statevector independently, so a simulation gives bit-identical
results at any thread count. Simulations may therefore use every core a run was given, while BLAS
and other OpenMP code stay on one thread, where summation order (and so the result) could change.

Run times are predicted from a simple cost model: a statevector simulation costs about
`gates x 2^qubits` amplitude updates plus a fixed per-circuit overhead. The constants are for one
core of a current server CPU; the platform scales the prediction to its own machines.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from threadpoolctl import ThreadpoolController

S_PER_GATE_AMPLITUDE = 1.8e-9  # one gate applied to one amplitude
S_PER_CIRCUIT = 0.012  # Aer call, result decoding and Python around one circuit
S_PER_SHOT = 2e-6  # drawing one shot and decoding it into counts
S_PER_TRANSPILE = 0.15  # transpiling a circuit the simulator cannot run as written
RUN_OVERHEAD_S = 20.0  # sandbox start, imports, classical baseline, interpretation
_PARALLEL_FROM_QUBITS = 15  # Aer parallelises statevector updates from 15 qubits up
_THREAD_EFFICIENCY = 0.7  # measured: 2 threads run 20-24 qubit circuits 1.6-1.7x faster

_controller: ThreadpoolController | None = None


def sim_threads() -> int:
    """Threads a simulation may use: QSOLVE_SIM_THREADS (set by the platform to the sandbox's
    CPUs), else every core this process may run on."""
    raw = os.environ.get("QSOLVE_SIM_THREADS", "")
    if raw.strip().isdigit() and int(raw) > 0:
        return int(raw)
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


@contextmanager
def simulator_threads() -> Iterator[None]:
    """Let OpenMP (Aer) use `sim_threads()` for the duration of a simulator call."""
    global _controller
    if _controller is None:  # created lazily so it sees Aer's OpenMP runtime once loaded
        _controller = ThreadpoolController()
    with _controller.limit(limits=sim_threads(), user_api="openmp"):
        yield


def statevector_seconds(
    qubits: int, gates: int, circuits: int = 1, shots: int = 0, threads: int | None = None
) -> float:
    """Predicted seconds to simulate `circuits` circuits of `gates` gates on `qubits` qubits,
    drawing `shots` shots from each."""
    work = S_PER_GATE_AMPLITUDE * gates * 2.0**qubits
    th = threads or sim_threads()
    if qubits >= _PARALLEL_FROM_QUBITS and th > 1:
        work /= 1 + _THREAD_EFFICIENCY * (th - 1)
    return circuits * (work + S_PER_CIRCUIT + S_PER_SHOT * shots)
