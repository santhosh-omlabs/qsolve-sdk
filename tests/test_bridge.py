"""The Qiskit-native bridge: a backend, a sampler and an estimator over a Qsolve target."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from qsolve.backends import AerBackend
from qsolve.bridge import QsolveBackend, QsolveEstimator, QsolveSampler
from qsolve.metering import Recorder
from qsolve.workflow import QuantumNotAllowed


def bell() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


def make(allow: bool = True) -> tuple[QsolveBackend, Recorder]:
    rec = Recorder()
    return QsolveBackend(AerBackend(), rec, seed=5, allow_quantum=allow), rec


def test_backend_run_returns_a_standard_qiskit_result_and_records_the_dispatch() -> None:
    backend, rec = make()
    counts = backend.run(bell(), shots=500).result().get_counts()
    assert sum(counts.values()) == 500 and set(counts) <= {"00", "11"}
    assert len(rec.dispatches) == 1 and rec.dispatches[0].shots == 500


def test_sampler_returns_bitarrays_and_cites_each_dispatch() -> None:
    backend, _ = make()
    result = QsolveSampler(backend).run([(bell(), None, 300)]).result()
    assert sum(result[0].data.meas.get_counts().values()) == 300
    assert [m["dispatch"] for m in result[0].metadata["qsolve"]] == [0]


def test_parameters_are_bound_per_pub_and_batched_into_one_dispatch_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, _ = make()
    calls: list[int] = []
    real = backend.inner.sample_batch

    def spy(circuits: Any, shots: int, seeds: Any) -> Any:
        calls.append(len(circuits))
        return real(circuits, shots, seeds)

    monkeypatch.setattr(backend.inner, "sample_batch", spy)
    theta = Parameter("t")
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)
    qc.measure_all()
    sampler = QsolveSampler(backend)
    result = sampler.run(
        [(qc, [0.0, np.pi]), (qc, [np.pi / 2]), (bell(), None)], shots=200
    ).result()
    assert calls == [4]  # two + one + one circuits, one batch: on a QPU, one job
    zero, flipped = result[0].data.meas.get_counts(0), result[0].data.meas.get_counts(1)
    assert zero == {"0": 200} and flipped == {"1": 200}


def test_the_same_seed_gives_the_same_counts() -> None:
    a = QsolveSampler(make()[0]).run([(bell(), None, 400)]).result()[0].data.meas.get_counts()
    b = QsolveSampler(make()[0]).run([(bell(), None, 400)]).result()[0].data.meas.get_counts()
    assert a == b


def test_a_classical_context_refuses_to_run_circuits() -> None:
    backend, rec = make(allow=False)
    with pytest.raises(QuantumNotAllowed):
        backend.run(bell(), shots=10)
    with pytest.raises(QuantumNotAllowed):
        QsolveSampler(backend).run([(bell(), None, 10)])
    assert rec.dispatches == []


def test_estimator_measures_each_pauli_group_in_its_own_basis() -> None:
    backend, _ = make()
    est = QsolveEstimator(backend, default_precision=0.02)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    from qiskit.quantum_info import SparsePauliOp

    ops = [SparsePauliOp("ZZ"), SparsePauliOp("XX"), SparsePauliOp("ZI"), SparsePauliOp("YY")]
    evs = est.run([(qc, ops)]).result()[0].data.evs
    assert np.allclose(evs, [1.0, 1.0, 0.0, -1.0], atol=0.12)  # a Bell state: <ZZ>=<XX>=1, <YY>=-1


def test_circuits_with_two_registers_keep_their_counts_per_register() -> None:
    from qiskit import ClassicalRegister, QuantumRegister

    q, a, b = QuantumRegister(2), ClassicalRegister(1, "a"), ClassicalRegister(1, "b")
    qc = QuantumCircuit(q, a, b)
    qc.x(1)
    qc.measure(0, a[0])
    qc.measure(1, b[0])
    data = QsolveSampler(make()[0]).run([(qc, None, 50)]).result()[0].data
    assert data.a.get_counts() == {"0": 50} and data.b.get_counts() == {"1": 50}


def test_what_is_recorded_is_what_was_sent_even_if_the_circuit_is_edited_afterwards() -> None:
    from qsolve.metering import Recorder

    rec = Recorder()
    backend = QsolveBackend(AerBackend(), rec)
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    backend.run(qc, shots=50)
    qc.data.clear()  # a workflow wiping its own object afterwards
    assert rec.dispatches[0].circuit.num_nonlocal_gates() == 2


def test_a_device_that_returns_more_shots_than_asked_still_gives_exactly_the_requested_shots() -> (
    None
):
    """Hardware can round the shot count up; a SamplerV2 result must still hold what was asked."""

    class Rounding(AerBackend):
        def sample(self, circuit: QuantumCircuit, shots: int, seed: int) -> Any:
            return super().sample(circuit, shots + 24, seed)

    backend = QsolveBackend(Rounding(), Recorder(), seed=5, allow_quantum=True)
    data = QsolveSampler(backend).run([(bell(), None, 500)]).result()[0].data.meas
    assert data.num_shots == 500
    assert sum(data.get_counts().values()) == 500


def test_seed_simulator_option_makes_a_call_reproducible() -> None:
    backend, _ = make()
    first = backend.run(bell(), shots=200, seed_simulator=3).result().get_counts()
    backend.run(bell(), shots=200)  # moves the running count; must not matter
    again = backend.run(bell(), shots=200, seed_simulator=3).result().get_counts()
    assert first == again
    backend.set_options(seed_simulator=3)
    assert backend.run(bell(), shots=200).result().get_counts() == first
