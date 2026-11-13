"""Qiskit-native execution on Qsolve.

A workflow written in plain Qiskit runs here unchanged: `QsolveBackend` is a Qiskit `BackendV2`
(`backend.run(circuits, shots=...)`), `QsolveSampler` is a `SamplerV2` and `QsolveEstimator` an
`EstimatorV2`. Behind them is whatever Qsolve target the run was given (simulator, noise model,
QPU). Circuits are sent as the workflow built them; the platform transpiles for the device.

Every dispatch is recorded (see `qsolve.metering`). A backend built with `allow_quantum=False`
refuses to run anything: classical methods get one, so "classical" is enforced.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from math import ceil
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import BaseEstimatorV2, BasePrimitiveJob, BaseSamplerV2
from qiskit.primitives.containers import (
    BitArray,
    DataBin,
    EstimatorPubLike,
    PrimitiveResult,
    PubResult,
    SamplerPubLike,
    SamplerPubResult,
)
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.providers import BackendV2, JobStatus, JobV1, Options
from qiskit.quantum_info import SparsePauliOp
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.transpiler import Target

from .backends.base import MAX_BATCH, Backend, SampleResult
from .metering import Recorder
from .workflow import QuantumNotAllowed

_GATES = [
    "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg", "rx", "ry", "rz", "p", "u",
    "cx", "cy", "cz", "ch", "swap", "crx", "cry", "crz", "cp", "rxx", "ryy", "rzz", "ecr", "ccx",
]  # fmt: skip
DEFAULT_SHOTS = 1024


class _Done(JobV1):  # type: ignore[misc]
    """A job that has already finished: runs are synchronous from the workflow's point of view."""

    def __init__(self, backend: BackendV2, job_id: str, result: Result) -> None:
        super().__init__(backend, job_id)
        self._result = result

    def submit(self) -> None:  # pragma: no cover - already done
        return None

    def result(self) -> Result:
        return self._result

    def status(self) -> JobStatus:
        return JobStatus.DONE

    def cancel(self) -> None:  # pragma: no cover - nothing to cancel
        return None


class QsolveBackend(BackendV2):  # type: ignore[misc]
    """A Qiskit backend over a Qsolve target."""

    def __init__(
        self, inner: Backend, recorder: Recorder, *, seed: int = 0, allow_quantum: bool = True
    ) -> None:
        """Wrap a Qsolve backend as a Qiskit backend.

        Args:
            inner: The Qsolve target to run on.
            recorder: Where every dispatch is recorded.
            seed: Base seed; each circuit gets `seed + running count`.
            allow_quantum: When false, `dispatch` raises `QuantumNotAllowed`.
        """
        super().__init__(name=f"qsolve-{inner.device}", backend_version="1")
        self.inner = inner
        self.recorder = recorder
        self.allow_quantum = allow_quantum
        self._seed = seed
        self._count = 0
        self._lock = threading.Lock()
        self._target = Target.from_configuration(basis_gates=_GATES, num_qubits=inner.max_qubits)

    @property
    def target(self) -> Target:
        """The Qiskit `Target`: a generic gate set sized to the inner backend's qubit count."""
        return self._target

    @property
    def max_circuits(self) -> int | None:
        """No limit (batches are split internally)."""
        return None

    @classmethod
    def _default_options(cls) -> Options:
        return Options(shots=DEFAULT_SHOTS, seed_simulator=None)

    # -- the one place circuits leave the workflow -------------------------------------------------

    def dispatch(
        self, circuits: Sequence[QuantumCircuit], shots: int, seed: int | None = None
    ) -> list[SampleResult]:
        """The one place circuits leave the workflow: copy, run, record and return results.

        Args:
            circuits: Circuits to run.
            shots: Shots per circuit.
            seed: If given, circuit `k` of this call gets seed `seed + k` (the same call always
                gives the same counts); otherwise seeds continue from the backend's base seed.

        Returns:
            One `SampleResult` per circuit; its metadata gains `dispatch`, the recorded index.

        Raises:
            QuantumNotAllowed: this backend was built with `allow_quantum=False`.
        """
        if not self.allow_quantum:
            raise QuantumNotAllowed(
                "a classical method tried to run a circuit: "
                "only the quantum method may use the backend"
            )
        out: list[SampleResult] = []
        for i in range(0, len(circuits), MAX_BATCH):
            # A snapshot: what is recorded is what is sent, even if the workflow edits its own
            # circuit object afterwards.
            chunk = [c.copy() for c in circuits[i : i + MAX_BATCH]]
            with self._lock:
                base = self._seed + self._count if seed is None else seed + i
                seeds = [base + k for k in range(len(chunk))]
                self._count += len(chunk)
            for circuit, res in zip(
                chunk, self.inner.sample_batch(chunk, shots, seeds), strict=True
            ):
                index = self.recorder.add(circuit, shots, res.counts, res.metadata, res.exec_time_s)
                res.metadata = {**res.metadata, "dispatch": index}  # lets a workflow cite it
                out.append(res)
        return out

    def run(self, run_input: QuantumCircuit | Sequence[QuantumCircuit], **options: Any) -> JobV1:
        """Qiskit `BackendV2.run`: run circuits and return a finished job.

        Args:
            run_input: One circuit or a sequence of circuits.
            **options: `shots` (default 1024) and `seed_simulator` (an int: circuit `k` of this
                call is seeded `seed_simulator + k`; default `None`, which continues the base seed).

        Returns:
            A job that is already done; `.result().get_counts()` works as usual.
        """
        circuits = [run_input] if isinstance(run_input, QuantumCircuit) else list(run_input)
        shots = int(options.get("shots") or self.options.shots)
        seed = options.get("seed_simulator", self.options.seed_simulator)
        results = self.dispatch(circuits, shots, None if seed is None else int(seed))
        job_id = f"qsolve-{self._count}"
        return _Done(self, job_id, _to_result(self.name, job_id, circuits, results))


def _hex(bits: str) -> str:
    return hex(int(bits.replace(" ", ""), 2))


def _to_result(
    name: str, job_id: str, circuits: Sequence[QuantumCircuit], results: Sequence[SampleResult]
) -> Result:
    experiments = []
    for qc, res in zip(circuits, results, strict=True):
        header = {
            "name": qc.name,
            "memory_slots": qc.num_clbits,
            "creg_sizes": [[r.name, r.size] for r in qc.cregs],
            "clbit_labels": [[r.name, i] for r in qc.cregs for i in range(r.size)],
            "qreg_sizes": [[r.name, r.size] for r in qc.qregs],
            "qubit_labels": [[r.name, i] for r in qc.qregs for i in range(r.size)],
            "metadata": res.metadata,
        }
        counts: dict[str, int] = {}
        for k, v in res.counts.items():
            counts[_hex(k)] = counts.get(_hex(k), 0) + int(v)
        experiments.append(
            ExperimentResult(
                shots=res.shots,
                success=True,
                data=ExperimentResultData(counts=counts),
                header=header,
            )
        )
    return Result(
        backend_name=name, backend_version="1", job_id=job_id, success=True, results=experiments
    )


# ---- Sampler -------------------------------------------------------------------------------------


class _Finished(BasePrimitiveJob[PrimitiveResult[Any], JobStatus]):  # type: ignore[misc]
    def __init__(self, result: PrimitiveResult[Any]) -> None:
        super().__init__("qsolve-primitive")
        self._r = result

    def result(self) -> PrimitiveResult[Any]:
        return self._r

    def status(self) -> JobStatus:
        return JobStatus.DONE

    def done(self) -> bool:
        return True

    def running(self) -> bool:
        return False

    def cancelled(self) -> bool:
        return False

    def in_final_state(self) -> bool:
        return True

    def cancel(self) -> None:  # pragma: no cover
        return None


def _creg_counts(counts: dict[str, int], qc: QuantumCircuit, name: str) -> dict[str, int]:
    """The counts restricted to one classical register (bit strings, clbit 0 on the right)."""
    reg = next(r for r in qc.cregs if r.name == name)
    idx = [qc.find_bit(b).index for b in reg]
    out: dict[str, int] = {}
    for key, v in counts.items():
        bits = key.replace(" ", "")
        n = len(bits)
        sub = "".join(bits[n - 1 - i] for i in reversed(idx))
        out[sub] = out.get(sub, 0) + int(v)
    return out


def _exact_shots(samples: np.ndarray[Any, Any], shots: int) -> np.ndarray[Any, Any]:
    """A `SamplerV2` result holds exactly the requested number of shots. Some devices round the
    count up (5000 asked, 5024 returned): keep a random subset of the requested size."""
    n = samples.shape[0]
    if n < shots:
        raise ValueError(f"the backend returned {n} shots for a request of {shots}")
    if n == shots:
        return samples
    keep = np.random.default_rng(n).choice(n, size=shots, replace=False)
    return samples[np.sort(keep)]


class QsolveSampler(BaseSamplerV2):  # type: ignore[misc]
    """`SamplerV2` over a `QsolveBackend`. Each pub result carries the per-circuit device metadata
    (readout errors, gate counts) under `metadata["qsolve"]`."""

    def __init__(self, backend: QsolveBackend, *, default_shots: int = DEFAULT_SHOTS) -> None:
        """Create the sampler.

        Args:
            backend: The `QsolveBackend` to run on.
            default_shots: Shots for pubs that give none.
        """
        self._backend = backend
        self._default_shots = default_shots

    def run(
        self, pubs: Iterable[SamplerPubLike], *, shots: int | None = None
    ) -> BasePrimitiveJob[Any, Any]:
        """Every circuit of every pub with the same shot count goes to the backend as one batch
        (on a QPU: one job, one queue wait, one per-job overhead)."""
        coerced = [SamplerPub.coerce(p, shots or self._default_shots) for p in pubs]
        bound = [pub.parameter_values.bind_all(pub.circuit) for pub in coerced]
        flat = [np.ravel(b).tolist() for b in bound]
        results: list[list[SampleResult]] = [[] for _ in coerced]
        for n_shots in dict.fromkeys(p.shots for p in coerced):
            members = [i for i, p in enumerate(coerced) if p.shots == n_shots]
            batch = [c for i in members for c in flat[i]]
            got = iter(self._backend.dispatch(batch, n_shots))
            for i in members:
                results[i] = [next(got) for _ in flat[i]]
        out = [
            self._pub_result(pub, b.shape, circuits, res)
            for pub, b, circuits, res in zip(coerced, bound, flat, results, strict=True)
        ]
        return _Finished(PrimitiveResult(out, metadata={"version": 2}))

    @staticmethod
    def _pub_result(
        pub: SamplerPub,
        shape: tuple[int, ...],
        circuits: list[QuantumCircuit],
        results: list[SampleResult],
    ) -> SamplerPubResult:
        data: dict[str, BitArray] = {}
        for reg in pub.circuit.cregs:
            arrays = [
                _exact_shots(
                    BitArray.from_counts(
                        _creg_counts(r.counts, qc, reg.name), num_bits=reg.size
                    ).array,
                    pub.shots,
                )
                for qc, r in zip(circuits, results, strict=True)
            ]
            data[reg.name] = BitArray(np.stack(arrays).reshape(*shape, pub.shots, -1), reg.size)
        return SamplerPubResult(
            DataBin(**data, shape=shape),
            metadata={
                "shots": pub.shots,
                "qsolve": [r.metadata for r in results],
                "circuit_metadata": pub.circuit.metadata,
            },
        )


# ---- Estimator -----------------------------------------------------------------------------------


def _basis_circuit(circuit: QuantumCircuit, basis: str) -> QuantumCircuit:
    """The circuit followed by the rotations that measure each qubit in the given Pauli basis
    (`basis[i]` acts on qubit `n - 1 - i`, Qiskit's Pauli-string order)."""
    n = circuit.num_qubits
    qc = QuantumCircuit(n, n)
    qc.compose(circuit.remove_final_measurements(inplace=False), inplace=True)
    for i, p in enumerate(reversed(basis)):
        if p == "X":
            qc.h(i)
        elif p == "Y":
            qc.sdg(i)
            qc.h(i)
    qc.measure(range(n), range(n))
    return qc


def _group_by_basis(op: SparsePauliOp) -> dict[str, list[tuple[str, complex]]]:
    """Qubit-wise commuting groups, greedily: each group is measured in one basis."""
    groups: dict[str, list[tuple[str, complex]]] = {}
    for label, coeff in zip(op.paulis.to_labels(), op.coeffs, strict=True):
        for basis in groups:
            if all(a == "I" or b == "I" or a == b for a, b in zip(basis, label, strict=True)):
                merged = "".join(b if a == "I" else a for a, b in zip(basis, label, strict=True))
                groups[merged] = [*groups.pop(basis), (label, coeff)]
                break
        else:
            groups[label] = [(label, coeff)]
    return groups


def _expectation(counts: dict[str, int], label: str) -> tuple[float, float]:
    """<P> and its variance per shot from counts measured in a basis that contains P."""
    support = [len(label) - 1 - i for i, p in enumerate(label) if p != "I"]
    total = sum(counts.values())
    s = 0.0
    for key, v in counts.items():
        bits = key.replace(" ", "")[::-1]
        s += v * (1 - 2 * (sum(int(bits[q]) for q in support) % 2))
    mean = s / total
    return mean, 1 - mean * mean


class QsolveEstimator(BaseEstimatorV2):  # type: ignore[misc]
    """`EstimatorV2` over a `QsolveBackend`: measures each group of qubit-wise commuting Pauli
    terms in its own basis and combines the results. No error mitigation is applied."""

    def __init__(self, backend: QsolveBackend, *, default_precision: float = 0.015625) -> None:
        """Create the estimator.

        Args:
            backend: The `QsolveBackend` to run on.
            default_precision: Target standard error when `run` gives none
                (shots = ceil(1 / precision^2)).
        """
        self._backend = backend
        self._default_precision = default_precision

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> BasePrimitiveJob[Any, Any]:
        """Estimate expectation values.

        Args:
            pubs: Estimator pubs (circuit, observables, optional parameter values).
            precision: Target standard error per value; defaults to `default_precision`.

        Returns:
            A finished job whose pub results hold `evs` and `stds`.
        """
        prec = precision or self._default_precision
        coerced = [EstimatorPub.coerce(p, prec) for p in pubs]
        return _Finished(PrimitiveResult([self._one(p) for p in coerced], metadata={"version": 2}))

    def _one(self, pub: EstimatorPub) -> PubResult:
        shots = ceil(1 / pub.precision**2)
        bound = pub.parameter_values.bind_all(pub.circuit)
        shape = np.broadcast_shapes(bound.shape, pub.observables.shape)
        evs = np.zeros(shape)
        stds = np.zeros(shape)
        for index in np.ndindex(*shape):
            circuit = bound[
                np.unravel_index(0, bound.shape) if bound.shape == () else _clip(index, bound.shape)
            ]
            obs = pub.observables[_clip(index, pub.observables.shape)]
            op = SparsePauliOp.from_list(list(obs.items()))
            groups = _group_by_basis(op)
            circuits = [_basis_circuit(circuit, b) for b in groups]
            results = self._backend.dispatch(circuits, shots)
            ev, var = 0.0, 0.0
            for (_basis, terms), res in zip(groups.items(), results, strict=True):
                for label, coeff in terms:
                    mean, v = _expectation(res.counts, label)
                    ev += float(np.real(coeff)) * mean
                    var += (float(np.real(coeff)) ** 2) * v / shots
            evs[index], stds[index] = ev, float(np.sqrt(var))
        return PubResult(
            DataBin(evs=evs, stds=stds, shape=shape), metadata={"target_precision": pub.precision}
        )


def _clip(index: tuple[int, ...], shape: tuple[int, ...]) -> tuple[int, ...]:
    """The index into an array that broadcasts to a larger shape."""
    lead = len(index) - len(shape)
    return tuple(0 if shape[i] == 1 else index[lead + i] for i in range(len(shape)))
