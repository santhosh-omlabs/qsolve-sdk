"""Measurement at the backend boundary.

Everything a workflow sends to a quantum backend passes through `Recorder`: the circuit, the shots,
the counts that came back. From those the SDK computes, without trusting the workflow, how big the
quantum run was (qubits, depth, two-qubit gates, circuits, shots) and how faithful it was: the
fidelity of the measured counts against the circuit's own ideal output, which the SDK simulates
noise-free. Mitigation is the workflow's own post-processing, so its output is only *reported* by
the workflow (`ctx.report_mitigation`); the SDK recomputes the reported distribution's fidelity
against the same ideal output and labels it as reported.

Fidelity here is the classical (Bhattacharyya) fidelity F = (sum_i sqrt(p_i q_i))^2 between two
distributions over bit strings. With finite shots even a perfect device scores below 1, so every
figure comes with `fidelity_floor`: the score a noise-free device would get with the same shots.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from .results import CircuitStats, Mitigation, Quality

IDEAL_MAX_QUBITS = 20  # noise-free simulation of the ideal output is exact up to here
MAX_IDEAL_CIRCUITS = 200  # distinct circuits simulated for fidelity (spread over the run)
MAX_STATS_CIRCUITS = 40  # distinct circuits transpiled for depth and gate counts
BASIS = ["cx", "rz", "sx", "x"]
_NOT_SIMULABLE = {"reset", "if_else", "while_loop", "for_loop", "switch_case", "delay"}


@dataclass
class Dispatch:
    """One circuit that crossed the backend boundary: index, sending method, circuit, shots,
    counts, backend metadata and seconds."""

    index: int
    method: str
    circuit: QuantumCircuit
    shots: int
    counts: dict[str, int]
    metadata: dict[str, Any]
    seconds: float


@dataclass
class MitigationReport:
    """What a workflow declared through `ctx.report_mitigation`."""

    circuits: list[int]  # dispatch indices whose output this mitigates
    distribution: dict[str, float]
    methods: list[str]
    kept: float | None
    steps: list[dict[str, Any]]


@dataclass
class Recorder:
    """Everything that crossed the backend boundary during one run."""

    dispatches: list[Dispatch] = field(default_factory=list)
    reports: list[MitigationReport] = field(default_factory=list)
    method: str = "quantum"  # the method currently running, set by the harness
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(
        self,
        circuit: QuantumCircuit,
        shots: int,
        counts: dict[str, int],
        metadata: dict[str, Any],
        seconds: float,
    ) -> int:
        """Record one dispatch and return its index."""
        with self.lock:
            i = len(self.dispatches)
            self.dispatches.append(
                Dispatch(i, self.method, circuit, shots, dict(counts), metadata, seconds)
            )
        return i

    def of(self, method: str) -> list[Dispatch]:
        """The dispatches sent by one method id."""
        return [d for d in self.dispatches if d.method == method]

    def report(
        self,
        circuits: list[int],
        distribution: dict[str, float],
        methods: list[str],
        kept: float | None,
        steps: list[dict[str, Any]],
    ) -> None:
        """Store a mitigation report (already validated by the runner)."""
        self.reports.append(MitigationReport(circuits, distribution, methods, kept, steps))


# ---- fidelity ----------------------------------------------------------------------------------


def _bits(key: str) -> str:
    return key.replace(" ", "")


def fidelity(p: dict[str, float], q: dict[str, float]) -> float:
    """Classical fidelity of two distributions (each normalised here; spaces in keys ignored)."""
    p = {_bits(k): v for k, v in p.items() if v > 0}
    q = {_bits(k): v for k, v in q.items() if v > 0}
    sp, sq = sum(p.values()), sum(q.values())
    if sp <= 0 or sq <= 0:
        return 0.0
    return float(sum(np.sqrt(v / sp * q[k] / sq) for k, v in p.items() if k in q) ** 2)


def shot_noise_floor(ideal: dict[str, float], shots: int, seed: int) -> float:
    """Fidelity of `shots` noise-free samples from `ideal` against `ideal` itself."""
    keys = list(ideal)
    probs = np.array([ideal[k] for k in keys], dtype=float)
    counts = np.random.default_rng(seed).multinomial(shots, probs / probs.sum())
    return fidelity({k: float(c) for k, c in zip(keys, counts, strict=True) if c}, ideal)


_PLAIN = frozenset({"measure", "barrier", "reset", "delay"})


def _param_key(p: Any) -> Any:
    if isinstance(p, int | float):
        return round(float(p), 12)
    if isinstance(p, np.ndarray):  # e.g. a unitary's matrix: str() would abbreviate it
        return ("array", p.shape, p.tobytes())
    return str(p)


def _body_key(op: Any) -> Any:
    """What a non-standard operation is made of. A Pauli-evolution gate keeps its operator outside
    `params`, so only its decomposition tells two such gates apart."""
    if (type(op).__module__ or "").startswith(
        "qiskit.circuit.library.standard_gates"
    ) or op.name in _PLAIN:
        return None
    definition = getattr(op, "definition", None)
    return circuit_key(definition) if definition is not None else None


def circuit_key(qc: QuantumCircuit) -> tuple[Any, ...]:
    """A hashable identity for a bound circuit: its size, operations, parameters and wiring."""
    ops = tuple(
        (
            i.operation.name,
            tuple(qc.find_bit(q).index for q in i.qubits),
            tuple(qc.find_bit(c).index for c in i.clbits),
            tuple(_param_key(p) for p in i.operation.params),
            _body_key(i.operation),
        )
        for i in qc.data
    )
    return (qc.num_qubits, qc.num_clbits, ops)


def _ideal_probs(qc: QuantumCircuit) -> np.ndarray | None:
    """The ideal output as an array indexed by the measured bit string read as a binary number
    (clbit 0 is the lowest bit), or None when it cannot be simulated (see `_scorable_body`)."""
    parts = _scorable_body(qc)
    if parts is None:
        return None
    body, measured = parts
    probs = Statevector(body).probabilities(qargs=[measured[c] for c in range(len(measured))])
    return np.asarray(probs, dtype=float)


def _fidelity_to(observed: dict[str, float], probs: np.ndarray) -> float:
    """Classical fidelity of an observed distribution (counts or weights, normalised here, spaces in
    keys ignored) with the ideal output `probs`. Only the observed outcomes are visited."""
    total = sum(v for v in observed.values() if v > 0)
    if total <= 0:
        return 0.0
    width = len(probs).bit_length() - 1
    acc = 0.0
    for key, v in observed.items():
        bits = _bits(key)
        if v <= 0 or len(bits) != width:
            continue
        try:
            i = int(bits, 2)
        except ValueError:
            continue
        acc += float(np.sqrt(probs[i] * v / total))
    return float(acc * acc)


def _floor_of(probs: np.ndarray, shots: int, seed: int) -> float:
    """Fidelity of `shots` noise-free samples from the ideal output against the ideal output."""
    p = probs / probs.sum()
    drawn = np.random.default_rng(seed).multinomial(shots, p)
    nz = np.flatnonzero(drawn)
    return float(np.sqrt(p[nz] * drawn[nz] / shots).sum() ** 2)


def _scorable_body(qc: QuantumCircuit) -> tuple[QuantumCircuit, dict[int, int]] | None:
    """The unitary part of a circuit and its measurement map (clbit -> qubit), or None when the
    ideal output cannot be simulated exactly here (too many qubits, mid-circuit measurement, resets,
    classical control, or clbits that are never measured). Cheap: nothing is simulated."""
    n = qc.num_qubits
    if n > IDEAL_MAX_QUBITS or n == 0:
        return None
    measured: dict[int, int] = {}  # clbit -> qubit
    body = QuantumCircuit(n)
    for inst in qc.data:
        op = inst.operation
        if op.name == "barrier":
            continue
        if op.name == "measure":
            measured[qc.find_bit(inst.clbits[0]).index] = qc.find_bit(inst.qubits[0]).index
            continue
        if measured or op.name in _NOT_SIMULABLE or getattr(op, "condition", None) is not None:
            return None
        body.append(op, [qc.find_bit(q).index for q in inst.qubits])
    if not measured or sorted(measured) != list(range(qc.num_clbits)):
        return None
    return body, measured


def ideal_distribution(qc: QuantumCircuit) -> dict[str, float] | None:
    """The noise-free distribution over the circuit's classical bits (bit string, clbit 0 on the
    right), or None when it cannot be simulated exactly here (see `_scorable_body`)."""
    probs = _ideal_probs(qc)
    if probs is None:
        return None
    width = len(probs).bit_length() - 1
    return {format(i, f"0{width}b"): float(probs[i]) for i in np.flatnonzero(probs > 1e-14)}


def active_qubits(qc: QuantumCircuit) -> int:
    """Number of distinct qubits the circuit's operations touch (barriers ignored)."""
    used = {
        qc.find_bit(q).index for i in qc.data for q in i.qubits if i.operation.name != "barrier"
    }
    return len(used)


def _spread(items: list[Any], limit: int) -> list[Any]:
    if len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


# ---- quality of the whole run ------------------------------------------------------------------


def _size(qc: QuantumCircuit) -> int:
    """Two-qubit operations in the circuit once composite gates are opened: a cheap measure of how
    much noise it collects, used to decide which circuits carry the run."""
    try:
        return int(qc.decompose(reps=2).num_nonlocal_gates())
    except Exception:
        return int(qc.num_nonlocal_gates())


def _distinct(dispatches: list[Dispatch]) -> dict[tuple[Any, ...], QuantumCircuit]:
    seen: dict[tuple[Any, ...], QuantumCircuit] = {}
    for d in dispatches:
        seen.setdefault(circuit_key(d.circuit), d.circuit)
    return seen


def circuit_stats(dispatches: list[Dispatch]) -> CircuitStats | None:
    """Measured size of what was sent: widest circuit, deepest and largest transpiled circuit
    (basis cx, rz, sx, x), circuit and shot totals; None when nothing was sent."""
    if not dispatches:
        return None
    seen = _distinct(dispatches)
    # Transpiling every distinct circuit can be too slow, so a sample is measured: the largest ones
    # first (a workflow cannot hide its deepest circuit among many small ones), then an even spread.
    ranked = sorted(seen.values(), key=_size, reverse=True)
    half = MAX_STATS_CIRCUITS // 2
    sample = ranked[:half] + _spread(ranked[half:], MAX_STATS_CIRCUITS - half)
    depth = gates = 0
    for qc in sample:
        t = transpile(qc, basis_gates=BASIS, optimization_level=1, seed_transpiler=0)
        depth = max(depth, t.depth())
        gates = max(gates, int(t.count_ops().get("cx", 0)))
    return CircuitStats(
        qubits=max(active_qubits(d.circuit) for d in dispatches),
        depth=depth,
        two_qubit_gates=gates,
        circuits=len(dispatches),
        shots=sum(d.shots for d in dispatches),
    )


def _pooled(dispatches: list[Dispatch]) -> dict[str, float]:
    total: dict[str, float] = {}
    for d in dispatches:
        for k, v in d.counts.items():
            total[_bits(k)] = total.get(_bits(k), 0.0) + v
    return total


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.allclose(a, b, atol=1e-9, rtol=0))


# A circuit counts towards the run's fidelity when it has at least this share of the two-qubit
# operations of the largest circuit: a trivial one-qubit circuit sent alongside carries no weight.
_CLASS_SHARE = 0.5


def _deterministic(probs: np.ndarray) -> bool:
    """The ideal output is a single bit string: the circuit computes nothing the answer depends on
    (a mirror or calibration circuit, an idle register)."""
    return bool(probs.max() > 1.0 - 1e-9)


def mitigation(rec: Recorder, method: str, source: str) -> Mitigation | None:
    """Raw fidelity from the recorded circuits; mitigated fidelity from the workflow's reports.

    Which circuits count: those a valid report covers, and every other circuit whose ideal output is
    not a single bit string. A circuit that is neither (a mirror circuit measured to calibrate a
    correction, a trivial padding circuit) says nothing about the answer. A circuit the workflow
    leaves unreported but that does compute something still counts, at its raw fidelity, so
    reporting only a flattering subset changes nothing. Of those, only the circuits that carry the
    run's noise are averaged (see `_CLASS_SHARE`), weighted by shots. A report counts only for
    circuits that share one ideal output. Circuits that cannot be scored (over 20 qubits,
    mid-circuit measurement) are left out and `coverage` says what share of the shots that is."""
    mine = rec.of(method)
    if not mine:
        return None
    ideal_cache: dict[tuple[Any, ...], np.ndarray | None] = {}
    size_cache: dict[tuple[Any, ...], int] = {}

    def ideal_of(d: Dispatch) -> np.ndarray | None:
        key = circuit_key(d.circuit)
        if key not in ideal_cache:
            ideal_cache[key] = _ideal_probs(d.circuit)
        return ideal_cache[key]

    def size_of(d: Dispatch) -> int:
        key = circuit_key(d.circuit)
        if key not in size_cache:
            size_cache[key] = _size(d.circuit)
        return size_cache[key]

    scorable = [d for d in mine if _scorable_body(d.circuit) is not None]
    if not scorable:
        return None
    total_shots = sum(d.shots for d in mine)
    coverage = sum(d.shots for d in scorable) / total_shots if total_shots else 1.0

    # reports: what the workflow says its post-processing produced, scored here against the ideal
    reported_f: dict[int, float] = {}
    methods: list[str] = []
    kept: list[float] = []
    steps: list[dict[str, Any]] = []
    used = 0
    for r in rec.reports:
        group = [rec.dispatches[i] for i in r.circuits]
        if not group or group[0].method != method:
            continue
        ideals = [ideal_of(d) for d in group]
        first = ideals[0]
        if first is None or any(i is None or not _same(first, i) for i in ideals[1:]):
            continue  # not comparable against one ideal output: the report is not used
        f = _fidelity_to(r.distribution, first)
        for d in group:
            reported_f[d.index] = min(f, reported_f.get(d.index, f))
        used += 1
        methods += [m for m in r.methods if m not in methods]
        kept.append(1.0 if r.kept is None else r.kept)
        steps = r.steps or steps  # the workflow's own ladder, whichever report carries it

    # the circuits that count: reported ones, and unreported ones that compute something
    rest_keys: dict[tuple[Any, ...], Dispatch] = {}
    for d in scorable:
        if d.index not in reported_f:
            rest_keys.setdefault(circuit_key(d.circuit), d)
    sampled = {
        circuit_key(d.circuit)
        for d in _spread(list(rest_keys.values()), MAX_IDEAL_CIRCUITS)  # simulated: bounded
    }
    core = [
        d
        for d in scorable
        if d.index in reported_f
        or (circuit_key(d.circuit) in sampled and not _deterministic(ideal_of(d)))  # type: ignore[arg-type]
    ]
    if not core:  # everything computes nothing (all calibration-like): score what there is
        core = [d for d in scorable if d.index in reported_f or circuit_key(d.circuit) in sampled]
    top = max(size_of(d) for d in core)
    carrying = [d for d in core if size_of(d) >= _CLASS_SHARE * top]

    # raw: the counts of every dispatch of one circuit pooled, against that circuit's ideal output
    by_key: dict[tuple[Any, ...], list[Dispatch]] = {}
    for d in carrying:
        by_key.setdefault(circuit_key(d.circuit), []).append(d)
    raw_of: dict[int, float] = {}
    floor_of: dict[int, float] = {}
    for group in by_key.values():
        ideal = ideal_of(group[0])
        assert ideal is not None
        f = _fidelity_to(_pooled(group), ideal)
        floor = _floor_of(ideal, sum(d.shots for d in group), group[0].index)
        for d in group:
            raw_of[d.index], floor_of[d.index] = f, floor
    weights = np.array([d.shots for d in carrying], dtype=float)

    def mean(values: dict[int, float]) -> float:
        return float(np.average([values[d.index] for d in carrying], weights=weights))

    raw, floor = mean(raw_of), mean(floor_of)
    if not used:
        return Mitigation(
            fidelity_raw=raw, fidelity_mitigated=raw, fidelity_floor=floor, postselection_rate=1.0,
            methods=[], source=source, coverage=coverage,
        )  # fmt: skip
    mitigated_of = {d.index: reported_f.get(d.index, raw_of[d.index]) for d in carrying}
    return Mitigation(
        fidelity_raw=raw, fidelity_mitigated=mean(mitigated_of), reported=True,
        fidelity_floor=floor, postselection_rate=float(np.mean(kept)), methods=methods,
        source=source, coverage=coverage, details={"steps": steps},
    )  # fmt: skip


def quality(rec: Recorder, method: str, source: str) -> Quality:
    """Fidelity and circuit size of one method's run.

    Args:
        rec: The run's recorder.
        method: Method id to measure.
        source: `hardware`, `noise_model` or `ideal_simulator`.

    Returns:
        A `Quality` with `mitigation` and `circuit` (each None when nothing was sent).
    """
    mine = rec.of(method)
    return Quality(mitigation=mitigation(rec, method, source), circuit=circuit_stats(mine))
