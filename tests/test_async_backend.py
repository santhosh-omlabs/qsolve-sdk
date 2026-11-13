"""AsyncBackend: providers whose jobs can be collected later by id."""

from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit

from qsolve.backends import AerBackend, AsyncBackend, ProviderJobFailed, SampleResult


class Queued(AsyncBackend):
    target, provider, device = "qpu", "test", "queued"

    def __init__(self) -> None:
        self.sim = AerBackend()
        self.jobs: dict[str, tuple[QuantumCircuit, int, int]] = {}

    def submit(self, circuit: QuantumCircuit, shots: int, seed: int) -> str:
        jid = f"j{len(self.jobs)}"
        self.jobs[jid] = (circuit, shots, seed)
        return jid

    def collect(self, job_id: str) -> SampleResult:
        if job_id not in self.jobs:
            raise ProviderJobFailed(job_id)
        return self.sim.sample(*self.jobs[job_id])


def test_sample_is_submit_then_collect_and_jobs_can_be_collected_again() -> None:
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.measure([0, 1], [0, 1])
    b = Queued()
    first = b.sample(qc, 200, 7)
    assert first.counts == {"01": 200}
    again = b.collect("j0")  # a second process re-attaching by id gets the same result
    assert again.counts == first.counts
    try:
        b.collect("missing")
    except ProviderJobFailed:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ProviderJobFailed")


class Batched(Queued):
    """A provider that runs several circuits as one job, like IBM's Sampler."""

    batches = True

    def __init__(self) -> None:
        super().__init__()
        self.batch_jobs: dict[str, tuple[list[QuantumCircuit], int, list[int]]] = {}

    def submit_batch(self, circuits: Any, shots: int, seeds: Any) -> str:
        jid = f"b{len(self.batch_jobs)}"
        self.batch_jobs[jid] = (list(circuits), shots, list(seeds))
        return jid

    def collect_batch(self, job_id: str) -> list[SampleResult]:
        circuits, shots, seeds = self.batch_jobs[job_id]
        return [self.sim.sample(c, shots, s) for c, s in zip(circuits, seeds, strict=True)]


def test_one_sampler_run_is_one_hardware_job_with_the_counts_of_separate_calls() -> None:
    from qsolve.bridge import QsolveBackend, QsolveSampler
    from qsolve.metering import Recorder

    circuits = []
    for angle in (0.2, 0.9, 1.7):
        qc = QuantumCircuit(2)
        qc.ry(angle, 0)
        qc.cx(0, 1)
        qc.measure_all()
        circuits.append(qc)
    batched = Batched()
    backend = QsolveBackend(batched, Recorder(), seed=4)
    result = QsolveSampler(backend).run([(c, None, 2000) for c in circuits]).result()
    assert len(batched.batch_jobs) == 1 and not batched.jobs  # every circuit in one job
    separate = QsolveBackend(AerBackend(), Recorder(), seed=4)
    again = QsolveSampler(separate).run([(c, None, 2000) for c in circuits]).result()
    for got, want in zip(result, again, strict=True):
        assert got.data.meas.get_counts() == want.data.meas.get_counts()


def test_providers_without_batching_still_get_one_job_per_circuit() -> None:
    b = Queued()
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    assert [r.counts for r in b.sample_batch([qc, qc], 50, [1, 2])] == [{"0": 50}, {"0": 50}]
    assert len(b.jobs) == 2
