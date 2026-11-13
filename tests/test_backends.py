"""Provider adapters: readout metadata, IBM job caps, shared transpilation, threads, executor."""

from __future__ import annotations

import json
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qiskit import QuantumCircuit

from qsolve import run_workflow
from qsolve.backends import AerBackend, get_backend
from qsolve.backends.hardware import measured_qubits, readout_errors
from qsolve.backends.ibm import max_execution_seconds
from qsolve.backends.remote import RemoteBackend, serve_executor
from qsolve.runner import estimate_run
from qsolve.sim import RUN_OVERHEAD_S, statevector_seconds
from support import Q, write_workflow

# ---- hardware metadata --------------------------------------------------------------------------


class _Props:
    def qubit_property(self, q: int, name: str) -> tuple[float, None]:
        return ({"prob_meas1_prep0": 0.01 * q, "prob_meas0_prep1": 0.02 * q}[name], None)


def test_readout_errors_follow_the_physical_qubit_of_each_bit() -> None:
    qc = QuantumCircuit(5, 2)
    qc.measure(3, 0)
    qc.measure(1, 1)
    assert measured_qubits(qc) == {0: 3, 1: 1}
    assert readout_errors(qc, _Props()) == [[0.03, 0.06], [0.01, 0.02]]
    assert readout_errors(qc, None) is None


def test_ibm_jobs_are_capped_near_their_expected_bill() -> None:
    assert max_execution_seconds(1, 4000) == 20  # ceil(10 + 3 x (2.4 + 0.92))
    assert max_execution_seconds(20, 25_000) > 3 * 0.00023 * 20 * 25_000


def test_identical_circuits_are_transpiled_once_and_share_the_result() -> None:
    from qiskit.providers.fake_provider import GenericBackendV2

    from qsolve.backends.hardware import transpile_best

    a = QuantumCircuit(2, 2)
    a.h(0)
    a.cx(0, 1)
    a.measure([0, 1], [0, 1])
    b = a.copy()
    b.x(1)
    out = transpile_best([a, a.copy(), b, a], GenericBackendV2(5, seed=1), seed=3, tries=2)
    assert out[0] is out[1] is out[3]
    assert out[2] is not out[0]


# ---- simulator threads and the run-time model --------------------------------------------------


def test_counts_do_not_depend_on_simulator_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    qc = QuantumCircuit(16)  # parallel from 15 qubits
    qc.h(range(16))
    for i in range(15):
        qc.rzz(0.4 + 0.1 * i, i, i + 1)
    qc.measure_all()
    counts = []
    for threads in ("1", "2", "4"):
        monkeypatch.setenv("QSOLVE_SIM_THREADS", threads)
        counts.append(AerBackend().sample(qc, 2000, 7).counts)
    assert counts[0] == counts[1] == counts[2]


def test_native_circuits_are_not_transpiled() -> None:
    aer = AerBackend()
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.rzz(0.3, 0, 1)
    qc.measure_all()
    assert aer.runnable(qc) is qc
    sub = QuantumCircuit(2)
    sub.cx(0, 1)
    custom = QuantumCircuit(2)
    custom.append(sub.to_gate(label="pair"), [0, 1])
    assert aer.runnable(custom) is not custom  # an opaque gate is decomposed first


def test_threads_shorten_predictions_only_where_aer_parallelises() -> None:
    assert statevector_seconds(10, 100, threads=4) == statevector_seconds(10, 100, threads=1)
    one, two = statevector_seconds(24, 300, threads=1), statevector_seconds(24, 300, threads=2)
    assert 1.5 < one / two < 2.0


def test_prediction_grows_exponentially_with_qubits() -> None:
    t = {n: statevector_seconds(n, 200, 100) for n in (12, 20, 24)}
    assert t[12] < t[20] < t[24] and t[24] > 8 * t[20]


def test_estimate_run_adds_the_run_around_what_the_workflow_predicts(tmp_path: Path) -> None:
    wf = write_workflow(
        tmp_path / "wf",
        body="""
        def estimate(self, target):
            return Estimate(qubits=3, circuits_on_target=1, shots_on_target=10, runtime_s=2.0)
        """,
    )
    est = estimate_run(str(wf), {"Q": Q})
    assert est.runtime_s == round(2.0 + RUN_OVERHEAD_S, 1)


# ---- the executor (what a workflow's circuits cross on the platform) ---------------------------


@pytest.fixture()
def executor() -> Iterator[Any]:
    server, addr = serve_executor(AerBackend(), max_shots=5000, max_qubits=10)
    yield addr
    server.shutdown()
    server.server_close()


def _bell(n: int = 2) -> QuantumCircuit:
    qc = QuantumCircuit(n, n)
    qc.h(0)
    for i in range(1, n):
        qc.cx(0, i)
    qc.measure(range(n), range(n))
    return qc


def test_remote_matches_local_counts(executor: str) -> None:
    remote = RemoteBackend(executor)
    local = AerBackend()
    assert remote.sample(_bell(3), 1000, 5).counts == local.sample(_bell(3), 1000, 5).counts
    assert remote.device == local.device and remote.max_qubits == 10


def test_limits_enforced(executor: str) -> None:
    remote = RemoteBackend(executor)
    with pytest.raises(RuntimeError, match="shots"):
        remote.sample(_bell(), 10_000, 0)
    with pytest.raises(RuntimeError, match="qubit"):
        remote.sample(_bell(12), 100, 0)


def test_garbage_requests_do_not_kill_the_executor(executor: str) -> None:
    host, port = executor.removeprefix("tcp:").rsplit(":", 1)
    for payload in (
        b"not json\n",
        b'{"op": "sample", "qpy": "!!", "shots": 10, "seed": 0}\n',
        b'{"op": "rm -rf"}\n',
        b"\n",
    ):
        with socket.create_connection((host, int(port))) as s:
            s.sendall(payload)
            reply = json.loads(s.makefile().readline() or "{}")
        assert reply.get("ok") is False
    assert RemoteBackend(executor).sample(_bell(), 100, 0).shots == 100  # still alive


def test_executor_token_is_enforced() -> None:
    server, addr = serve_executor(AerBackend(), token="s3cret-token")
    try:
        with pytest.raises(RuntimeError, match="bad executor token"):
            RemoteBackend(addr)  # no token
        with pytest.raises(RuntimeError, match="bad executor token"):
            RemoteBackend(addr, token="wrong")
        assert RemoteBackend(addr, token="s3cret-token").sample(_bell(), 50, 1).shots == 50
    finally:
        server.shutdown()
        server.server_close()


def test_batches_cross_the_executor_with_the_counts_of_separate_calls(executor: str) -> None:
    remote, local = RemoteBackend(executor), AerBackend()
    circuits, seeds = [_bell(2), _bell(3), _bell(4)], [11, 12, 13]
    got = remote.sample_batch(circuits, 777, seeds)
    assert [r.counts for r in got] == [
        local.sample(c, 777, s).counts for c, s in zip(circuits, seeds, strict=True)
    ]
    with pytest.raises(RuntimeError, match="qubit"):
        remote.sample_batch([_bell(2), _bell(12)], 100, [0, 1])  # one bad circuit refuses all
    with pytest.raises(ValueError, match="one seed"):
        remote.sample_batch([_bell(2)], 100, [0, 1])


def test_a_workflow_runs_through_the_executor_as_it_does_on_the_platform(
    executor: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("QSOLVE_EXECUTOR_ADDR", executor)
    assert isinstance(get_backend("ibm:anything"), RemoteBackend)  # the executor decides the device
    wf = write_workflow(tmp_path / "wf")
    out = run_workflow(str(wf), {"Q": np.array(Q).tolist()}, target="ibm:anything")
    assert out.result.quality is not None and out.result.quality.circuit is not None
    assert out.result.quality.circuit.circuits == 1


def test_get_backend_rejects_unknown_keyword_arguments() -> None:
    with pytest.raises(TypeError):
        get_backend("sim-cpu", gpu=True)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="noise scale"):
        get_backend("fake:fez@fast")


def test_fake_hardware_batch_uses_every_circuits_seed() -> None:
    pytest.importorskip("qiskit_ibm_runtime")
    from qsolve.backends.fake import FakeHardwareBackend

    fake = FakeHardwareBackend("fez")
    circuits = [_bell(2), _bell(2)]
    a = fake.sample_batch(circuits, 300, [1, 2])
    assert [r.counts for r in fake.sample_batch(circuits, 300, [1, 2])] == [r.counts for r in a]
    b = fake.sample_batch(circuits, 300, [1, 99])  # only the second seed changed
    assert [r.counts for r in b] != [r.counts for r in a]
