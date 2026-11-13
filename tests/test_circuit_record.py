"""Circuits as drawable records, and the solver details carried on the receipt."""

import json

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator, SparsePauliOp

from qsolve.backends.hardware import DEVICE_CIRCUITS, device_metadata, transpile_best
from qsolve.circuit_record import DRAWABLE, active_qubits, circuit_ops, circuit_summary, lowered
from qsolve.runner import _plain


def _evolution() -> QuantumCircuit:
    qc = QuantumCircuit(3, 3)
    qc.x(0)
    terms = SparsePauliOp(["XXY", "YYY"], [0.5, -0.25])  # commuting, so the expansion is exact
    qc.append(PauliEvolutionGate(terms, time=0.7), [0, 1, 2])
    qc.measure(range(3), range(3))
    return qc


def test_library_blocks_are_lowered_to_drawable_gates_without_changing_the_circuit() -> None:
    qc = _evolution()
    low = lowered(qc)
    assert {i.operation.name for i in low.data} <= DRAWABLE
    a, b = qc.remove_final_measurements(inplace=False), low.remove_final_measurements(inplace=False)
    assert Operator(a).equiv(Operator(b))


def test_ops_list_names_bits_and_rounded_parameters() -> None:
    rec = circuit_ops(_evolution())
    assert rec["qubits"] == 3 and rec["clbits"] == 3 and not rec["truncated"]
    names = [op[0] for op in rec["ops"]]
    assert names[0] == "x" and names[-3:] == ["measure"] * 3
    assert rec["ops"][-1][3] == [2]  # measure q2 -> c2
    assert all(isinstance(p, float) for op in rec["ops"] for p in op[2])
    json.dumps(rec)  # plain JSON
    assert circuit_ops(_evolution(), max_ops=2)["truncated"]


def test_summary_counts_without_lowering() -> None:
    s = circuit_summary(_evolution())
    assert s["qubits"] == 3 and s["ops"]["measure"] == 3 and s["two_qubit_gates"] == 0


def test_device_metadata_reports_physical_qubits_and_a_few_device_circuits() -> None:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    variants = []
    for k in range(DEVICE_CIRCUITS + 2):
        v = qc.copy()
        v.rz(0.1 * (k + 1), 0)
        variants.append(v)
    compiled = transpile_best(variants, GenericBackendV2(7, seed=2), seed=1, tries=1)
    meta = device_metadata(compiled)
    assert sum("device_circuit" in m for m in meta) == DEVICE_CIRCUITS
    for c, m in zip(compiled, meta, strict=True):
        assert m["physical_qubits"] == active_qubits(c) and len(m["physical_qubits"]) == 2
        assert m["two_qubit_gates"] >= 1


def test_solver_details_become_plain_json_or_a_size_note() -> None:
    d = _plain({"a": np.float64(1.5), "b": np.arange(3), "c": {"d": np.int64(2)}})
    assert d == {"a": 1.5, "b": [0, 1, 2], "c": {"d": 2}}
    big = _plain({"x": list(range(100_000))})
    assert "omitted" in big
