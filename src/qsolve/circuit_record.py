"""Circuits as compact, drawable records: what the platform shows on a run page.

`circuit_ops` lowers a circuit to standard gates (library blocks such as Pauli evolutions are
expanded) and lists them as `[name, qubits, params, clbits]`, so any viewer can draw or export it
without Qiskit. `circuit_summary` is the cheap overview kept for every distinct circuit.
"""

from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit

# Gates a viewer can draw and OpenQASM 2 (qelib1.inc) can express, plus non-unitary operations.
DRAWABLE = {
    "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg", "rx", "ry", "rz", "p", "u",
    "u1", "u2", "u3", "cx", "cy", "cz", "ch", "swap", "crx", "cry", "crz", "cp", "rxx", "ryy",
    "rzz", "ccx", "cswap", "ecr", "measure", "barrier", "reset", "delay",
}  # fmt: skip
_LOWERING_ROUNDS = 8


def _param(p: Any) -> float | str:
    try:
        return round(float(p), 6)
    except (TypeError, ValueError):
        return str(p)  # an unbound parameter: shown by name


def lowered(circuit: QuantumCircuit) -> QuantumCircuit:
    """Expand every operation outside DRAWABLE until none is left (or a few rounds have passed:
    opaque operations without a definition stay as they are, drawn as labelled boxes)."""
    for _ in range(_LOWERING_ROUNDS):
        names = {inst.operation.name for inst in circuit.data} - DRAWABLE
        if not names:
            break
        circuit = circuit.decompose(gates_to_decompose=sorted(names))
    return circuit


def circuit_ops(circuit: QuantumCircuit, max_ops: int = 20_000) -> dict[str, Any]:
    """{"qubits", "clbits", "ops": [[name, [q...], [params...], [c...]], ...], "truncated"}.
    Qubit indices are positions in the circuit (physical qubits for a transpiled circuit)."""
    c = lowered(circuit)
    ops: list[list[Any]] = []
    for inst in c.data:
        if len(ops) >= max_ops:
            break
        op = inst.operation
        ops.append(
            [
                op.name,
                [c.find_bit(q).index for q in inst.qubits],
                [_param(p) for p in op.params] if op.name != "delay" else [],
                [c.find_bit(b).index for b in inst.clbits],
            ]
        )
    return {
        "qubits": c.num_qubits,
        "clbits": c.num_clbits,
        "ops": ops,
        "truncated": len(c.data) > max_ops,
    }


def circuit_summary(circuit: QuantumCircuit) -> dict[str, Any]:
    """Size and shape without lowering: cheap enough for every circuit a run sends."""
    counts = {str(k): int(v) for k, v in circuit.count_ops().items()}
    two = sum(
        1
        for inst in circuit.data
        if inst.operation.num_qubits == 2 and inst.operation.name != "barrier"
    )
    return {
        "qubits": circuit.num_qubits,
        "clbits": circuit.num_clbits,
        "size": int(circuit.size()),
        "depth": int(circuit.depth()),
        "two_qubit_gates": two,
        "ops": counts,
    }


def active_qubits(circuit: QuantumCircuit) -> list[int]:
    """Qubits some operation touches (for a transpiled circuit: the physical qubits used)."""
    used = {circuit.find_bit(q).index for inst in circuit.data for q in inst.qubits}
    return sorted(used)
