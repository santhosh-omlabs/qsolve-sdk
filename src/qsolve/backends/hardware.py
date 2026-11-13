"""What every gate-based hardware backend does the same way: transpile for the device keeping the
cheapest result, and report, per measured bit, the readout error of the qubit it came from (so
workflows can correct readout without knowing the layout)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

TRANSPILE_TRIES = 4  # layouts and routings differ by seed; the fewest two-qubit gates wins


def two_qubit_count(circuit: QuantumCircuit) -> int:
    """Number of two-qubit operations in a circuit, barriers excluded."""
    return sum(
        1
        for inst in circuit.data
        if inst.operation.num_qubits == 2 and inst.operation.name != "barrier"
    )


def transpile_best(
    circuits: Sequence[QuantumCircuit], backend: Any, seed: int, tries: int = TRANSPILE_TRIES
) -> list[QuantumCircuit]:
    """Transpile at optimisation level 3 (noise-aware layout) with `tries` seeds and keep, per
    circuit, the result with the fewest two-qubit gates, then the shallowest. Deterministic.

    Circuits that differ only in rotation angles are compiled once, as a template with the angles
    left as parameters, and then given their angles: they run the same physical gates (the
    optimiser cannot delete a gate whose angle it does not know), and copies sent to split shots
    share one result."""
    templates = [_template(c) for c in circuits]
    keys = [_content_key(t) for t, _ in templates]
    first: dict[tuple[Any, ...], int] = {}
    for i, k in enumerate(keys):
        first.setdefault(k, i)
    unique = [templates[i][0] for i in first.values()]
    best: list[QuantumCircuit | None] = [None] * len(unique)
    for attempt in range(tries):
        pm = generate_preset_pass_manager(
            backend=backend, optimization_level=3, seed_transpiler=seed + attempt
        )
        for i, t in enumerate(pm.run(unique)):
            b = best[i]
            if b is None or (two_qubit_count(t), t.depth()) < (two_qubit_count(b), b.depth()):
                best[i] = t
    slot = {k: j for j, k in enumerate(first)}
    bound: dict[tuple[int, tuple[float, ...]], QuantumCircuit] = {}
    out = []
    for k, (_t, values) in zip(keys, templates, strict=True):
        j = slot[k]
        if (j, tuple(values)) not in bound:
            compiled = best[j]
            assert compiled is not None
            names = templates[list(first.values())[j]][0].parameters  # the representative's
            bound[(j, tuple(values))] = (
                compiled.assign_parameters(_by_name(names, values), strict=False)
                if values
                else compiled
            )
        out.append(bound[(j, tuple(values))])
    return out


_ROTATIONS = {"rx", "ry", "rz", "p", "u1", "crx", "cry", "crz", "cp", "rxx", "ryy", "rzz"}


def _template(circuit: QuantumCircuit) -> tuple[QuantumCircuit, list[float]]:
    """The circuit with each numeric rotation angle replaced by a parameter a0, a1, ... in order,
    and those angles."""
    from qiskit.circuit import Parameter

    t = circuit.copy_empty_like()
    values: list[float] = []
    for inst in circuit.data:
        op = inst.operation
        if op.name in _ROTATIONS and all(isinstance(p, int | float) for p in op.params):
            op = op.copy()
            new = []
            for p in op.params:
                new.append(Parameter(f"a{len(values):06d}"))  # zero-padded: sorts in order
                values.append(float(p))
            op.params = new
        t.append(op, inst.qubits, inst.clbits)
    return t, values


def _by_name(params: Any, values: list[float]) -> dict[Any, float]:
    """Bind by the zero-padded names, which sort in creation order."""
    return {p: values[int(p.name[1:])] for p in params}


def _content_key(circuit: QuantumCircuit) -> tuple[Any, ...]:
    """What the circuit does: each instruction's name, parameters and bit indices."""
    instructions = (
        (
            inst.operation.name,
            tuple(
                float(p) if isinstance(p, (int, float)) else str(p) for p in inst.operation.params
            ),
            tuple(circuit.find_bit(q).index for q in inst.qubits),
            tuple(circuit.find_bit(c).index for c in inst.clbits),
        )
        for inst in circuit.data
    )
    return (*instructions, (circuit.num_qubits, circuit.num_clbits))


def measured_qubits(circuit: QuantumCircuit) -> dict[int, int]:
    """clbit index -> physical qubit index, from the measurements of a transpiled circuit."""
    out: dict[int, int] = {}
    for inst in circuit.data:
        if inst.operation.name == "measure":
            q = circuit.find_bit(inst.qubits[0]).index
            c = circuit.find_bit(inst.clbits[0]).index
            out[c] = q
    return out


def readout_errors(circuit: QuantumCircuit, properties: Any) -> list[list[float]] | None:
    """[[P(1|0), P(0|1)], ...] per clbit for a transpiled circuit, from device properties taken
    when the job ran; None when the device does not report them."""
    mapping = measured_qubits(circuit)
    if not mapping or properties is None:
        return None
    out: list[list[float]] = []
    try:
        for c in range(circuit.num_clbits):
            q = mapping[c]
            p10 = float(properties.qubit_property(q, "prob_meas1_prep0")[0])
            p01 = float(properties.qubit_property(q, "prob_meas0_prep1")[0])
            out.append([p10, p01])
    except Exception:
        return None
    return out


def two_qubit_counts(circuits: Sequence[QuantumCircuit]) -> list[int]:
    """`two_qubit_count` of each circuit in a sequence."""
    return [two_qubit_count(c) for c in circuits]


DEVICE_CIRCUITS = 4  # per batch: what the device ran, for the run page (the executor caps replies)
DEVICE_CIRCUIT_MAX_OPS = 5_000


def device_metadata(compiled: Sequence[QuantumCircuit]) -> list[dict[str, Any]]:
    """Per result: two-qubit gate count and physical qubits used, and for the first few distinct
    circuits the transpiled circuit itself, as drawable ops."""
    from ..circuit_record import active_qubits, circuit_ops

    out: list[dict[str, Any]] = []
    shown: set[int] = set()
    for c in compiled:
        meta: dict[str, Any] = {
            "two_qubit_gates": two_qubit_count(c),
            "physical_qubits": active_qubits(c),
        }
        if (
            id(c) not in shown
            and len(shown) < DEVICE_CIRCUITS
            and c.size() <= DEVICE_CIRCUIT_MAX_OPS
        ):
            shown.add(id(c))
            meta["device_circuit"] = circuit_ops(c, DEVICE_CIRCUIT_MAX_OPS)
        out.append(meta)
    return out
