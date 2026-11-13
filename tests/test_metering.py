"""What the SDK measures at the backend boundary, without trusting the workflow."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from qsolve.backends import AerBackend
from qsolve.bridge import QsolveBackend, QsolveSampler
from qsolve.metering import (
    Recorder,
    circuit_stats,
    fidelity,
    ideal_distribution,
    mitigation,
    quality,
    shot_noise_floor,
)


def bell(n: int = 2) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(1, n):
        qc.cx(0, i)
    qc.measure_all()
    return qc


def run(circuits: list[QuantumCircuit], shots: int = 2000) -> Recorder:
    rec = Recorder()
    QsolveSampler(QsolveBackend(AerBackend(), rec, seed=1)).run(
        [(c, None, shots) for c in circuits]
    )
    return rec


def test_fidelity_of_identical_and_disjoint_distributions() -> None:
    assert fidelity({"00": 0.5, "11": 0.5}, {"00": 0.5, "11": 0.5}) == pytest.approx(1.0)
    assert fidelity({"00": 1.0}, {"11": 1.0}) == 0.0
    assert fidelity({"00 1": 3, "11 0": 1}, {"001": 3, "110": 1}) == pytest.approx(
        1.0
    )  # spaces ignored


def test_the_ideal_distribution_of_a_bell_state() -> None:
    ideal = ideal_distribution(bell())
    assert ideal == pytest.approx({"00": 0.5, "11": 0.5})


def test_circuits_that_cannot_be_simulated_exactly_have_no_ideal() -> None:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    qc.x(1)  # a gate after a measurement: mid-circuit
    qc.measure(1, 1)
    assert ideal_distribution(qc) is None
    assert ideal_distribution(QuantumCircuit(30, 30)) is None


def test_the_ideal_distribution_follows_the_classical_bit_each_qubit_is_measured_into() -> None:
    qc = QuantumCircuit(2, 2)
    qc.x(1)
    qc.measure(0, 1)  # qubit 0 (|0>) into clbit 1
    qc.measure(1, 0)  # qubit 1 (|1>) into clbit 0
    assert ideal_distribution(qc) == pytest.approx({"01": 1.0})  # clbit 1 on the left


def test_raw_fidelity_is_measured_from_the_recorded_counts_not_reported() -> None:
    m = mitigation(run([bell()]), "quantum", "ideal_simulator")
    assert m is not None and not m.reported
    assert m.fidelity_raw == pytest.approx(m.fidelity_mitigated)
    assert 0.99 < m.fidelity_raw <= 1.0 and m.fidelity_floor > 0.99 and m.postselection_rate == 1.0


def test_a_mitigation_report_is_recomputed_against_the_ideal_and_labelled_reported() -> None:
    rec = run([bell()])
    perfect = {"00": 0.5, "11": 0.5}
    rec.report([0], perfect, ["post-selection"], 0.9, [{"name": "raw", "fidelity": 0.9}])
    m = mitigation(rec, "quantum", "hardware")
    assert m is not None and m.reported and m.methods == ["post-selection"]
    assert m.fidelity_mitigated == pytest.approx(1.0) and m.postselection_rate == 0.9
    assert m.source == "hardware" and m.details["steps"][0]["name"] == "raw"
    # a report that is far from the ideal output scores low: the SDK does the arithmetic
    rec2 = run([bell()])
    rec2.report([0], {"01": 1.0}, ["made up"], None, [])
    assert mitigation(rec2, "quantum", "hardware").fidelity_mitigated == 0.0  # type: ignore[union-attr]


def test_circuit_stats_are_counted_from_what_was_sent() -> None:
    rec = run([bell(3), bell(3)], shots=100)
    s = circuit_stats(rec.dispatches)
    assert s is not None and s.qubits == 3 and s.circuits == 2 and s.shots == 200
    assert s.two_qubit_gates == 2 and s.depth >= 3


def test_nothing_sent_means_no_quality_numbers() -> None:
    q = quality(Recorder(), "quantum", "ideal_simulator")
    assert q.mitigation is None and q.circuit is None


def test_the_shot_noise_floor_falls_with_fewer_shots() -> None:
    ideal = {format(i, "04b"): 1 / 16 for i in range(16)}
    assert shot_noise_floor(ideal, 100_000, 0) > shot_noise_floor(ideal, 100, 0)


def test_circuit_key_tells_apart_circuits_that_differ_only_inside_an_evolution_gate() -> None:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp

    from qsolve.metering import circuit_key

    def build(label: str) -> QuantumCircuit:
        qc = QuantumCircuit(3)
        qc.append(PauliEvolutionGate(SparsePauliOp(label), time=0.3), [0, 1, 2])
        return qc

    assert circuit_key(build("XYZ")) != circuit_key(build("XZZ"))
    assert circuit_key(build("XYZ")) == circuit_key(build("XYZ"))


# ---- fidelity cannot be improved by what is sent next to the real circuit ---------------------


def _ghz(n: int = 4) -> QuantumCircuit:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


def _noisy_counts(n: int, shots: int, fidelity_like: float) -> dict[str, int]:
    """GHZ counts with a share of the shots spread over the other outcomes."""
    ok = int(shots * fidelity_like)
    rest = shots - ok
    others = [format(i, f"0{n}b") for i in range(1, 2**n - 1)]
    counts = {"0" * n: ok // 2, "1" * n: ok - ok // 2}
    for k in others:
        counts[k] = rest // len(others)
    return counts


def _recorder(*dispatches: tuple[QuantumCircuit, int, dict[str, int]]) -> Recorder:
    from qsolve.metering import Recorder

    rec = Recorder()
    for qc, shots, counts in dispatches:
        rec.add(qc, shots, counts, {}, 0.0)
    return rec


def test_trivial_decoy_circuits_do_not_raise_the_fidelity() -> None:
    from qiskit import QuantumCircuit

    from qsolve.metering import mitigation

    decoy = QuantumCircuit(1)
    decoy.measure_all()
    real = (_ghz(), 1000, _noisy_counts(4, 1000, 0.6))
    honest = mitigation(_recorder(real), "quantum", "noise_model")
    padded = mitigation(_recorder(real, (decoy, 200_000, {"0": 200_000})), "quantum", "noise_model")
    assert honest is not None and padded is not None
    assert padded.fidelity_raw == honest.fidelity_raw < 0.8  # the decoy carries no weight


def test_reporting_only_a_flattering_subset_changes_nothing_for_the_rest() -> None:
    from qsolve.metering import mitigation

    good = (_ghz(), 1000, {"0000": 500, "1111": 500})
    bad = (_ghz(5), 1000, _noisy_counts(5, 1000, 0.5))
    rec = _recorder(good, bad)
    rec.report([0], {"0000": 0.5, "1111": 0.5}, ["readout"], 1.0, [])  # reports only the good one
    m = mitigation(rec, "quantum", "noise_model")
    assert m is not None and m.fidelity_mitigated < 0.8  # the unreported circuit still counts, raw


def test_a_report_over_circuits_with_different_ideal_outputs_is_not_used() -> None:
    from qsolve.metering import mitigation

    a = (_ghz(4), 500, _noisy_counts(4, 500, 0.5))
    b = (_ghz(5), 500, _noisy_counts(5, 500, 0.5))
    rec = _recorder(a, b)
    rec.report([0, 1], {"0000": 0.5, "1111": 0.5}, ["x"], 1.0, [])
    m = mitigation(rec, "quantum", "noise_model")
    assert m is not None and not m.reported and m.fidelity_mitigated == m.fidelity_raw


def test_circuits_that_cannot_be_scored_are_counted_in_the_coverage() -> None:
    from qiskit import QuantumCircuit

    from qsolve.metering import mitigation

    big = QuantumCircuit(21)
    big.h(range(21))
    big.measure_all()
    rec = _recorder((_ghz(), 1000, {"0000": 500, "1111": 500}), (big, 3000, {"0" * 21: 3000}))
    m = mitigation(rec, "quantum", "noise_model")
    assert m is not None and m.coverage == pytest.approx(0.25)


def test_the_deepest_circuit_cannot_hide_among_many_small_ones() -> None:
    from qiskit import QuantumCircuit

    from qsolve.metering import MAX_STATS_CIRCUITS, circuit_stats

    small = []
    for k in range(3 * MAX_STATS_CIRCUITS):
        qc = QuantumCircuit(2)
        qc.rz(0.01 * (k + 1), 0)
        qc.measure_all()
        small.append(qc)
    deep = QuantumCircuit(2)
    for _ in range(30):
        deep.cx(0, 1)
        deep.rz(0.3, 1)
    deep.measure_all()
    order = [*small[:7], deep, *small[7:]]  # the deep circuit sits at an arbitrary position
    rec = _recorder(*((qc, 10, {"00": 10}) for qc in order))
    stats = circuit_stats(rec.dispatches)
    assert stats is not None and stats.two_qubit_gates >= 1 and stats.depth >= 30


def test_circuit_key_includes_the_register_sizes() -> None:
    from qiskit import QuantumCircuit

    from qsolve.metering import circuit_key

    a, b = QuantumCircuit(2, 2), QuantumCircuit(2, 1)
    for qc in (a, b):
        qc.h(0)
        qc.measure(0, 0)
    assert circuit_key(a) != circuit_key(b)


def test_the_fast_fidelity_agrees_with_the_plain_formula() -> None:
    from qsolve.metering import _fidelity_to, _floor_of, _ideal_probs, fidelity, ideal_distribution

    qc = _ghz(4)
    counts = _noisy_counts(4, 4000, 0.7)
    probs, ideal = _ideal_probs(qc), ideal_distribution(qc)
    assert probs is not None and ideal is not None
    assert _fidelity_to({f"{k[:2]} {k[2:]}": v for k, v in counts.items()}, probs) == pytest.approx(
        fidelity({k: float(v) for k, v in counts.items()}, ideal)
    )  # spaces in keys (several registers) are ignored
    assert _fidelity_to({"0": 5.0}, probs) == 0.0  # a key of the wrong width scores nothing
    assert 0.99 < _floor_of(probs, 100_000, 1) <= 1.0


def test_a_calibration_circuit_that_computes_nothing_is_not_the_answers_fidelity() -> None:
    """A mirror circuit (ideal output: all zeros) measured to fit a correction is bigger than the
    circuit that makes the answer and much noisier. It must not stand for the run."""
    from qiskit import QuantumCircuit

    from qsolve.metering import mitigation

    mirror = QuantumCircuit(4)
    for _ in range(3):
        for i in range(3):
            mirror.cx(i, i + 1)
        for i in reversed(range(3)):
            mirror.cx(i, i + 1)
    mirror.measure_all()
    probe = (_ghz(), 1000, _noisy_counts(4, 1000, 0.9))
    rec = _recorder(probe, (mirror, 1000, {"0000": 700, "0001": 300}))
    rec.report([0], {"0000": 0.5, "1111": 0.5}, ["rescale"], 1.0, [])  # the probe, corrected
    m = mitigation(rec, "quantum", "noise_model")
    assert m is not None and m.reported
    assert m.fidelity_mitigated > 0.99  # the probe's corrected fidelity, not the mirror's 0.7
    assert 0.85 < m.fidelity_raw < 0.95  # raw is the probe's own, measured the same way


def test_a_second_report_is_not_capped_by_circuits_it_does_not_cover() -> None:
    from qsolve.metering import mitigation

    a = (_ghz(), 1000, _noisy_counts(4, 1000, 0.5))
    b = (_ghz(5), 1000, _noisy_counts(5, 1000, 0.5))
    rec = _recorder(a, b)
    rec.report([0], {"0000": 0.5, "1111": 0.5}, ["x"], 1.0, [])
    rec.report([1], {"00000": 0.5, "11111": 0.5}, ["x"], 1.0, [])
    m = mitigation(rec, "quantum", "noise_model")
    assert m is not None and m.fidelity_mitigated > 0.99
