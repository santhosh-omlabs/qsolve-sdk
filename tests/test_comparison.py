"""How the headline numbers are computed from the methods the SDK measured."""

from __future__ import annotations

from qsolve.benchmark import Expect, check_estimate, check_quality
from qsolve.comparison import headline
from qsolve.manifest import MetricDecl
from qsolve.results import CircuitStats, Estimate, MethodResult, Result, Uncertainty
from qsolve.workflow import Outcome


def m(role: str, value: float, **kw: object) -> MethodResult:
    return MethodResult(id=role, role=role, title=role, value=value, **kw)  # type: ignore[arg-type]


def test_headline_formulas() -> None:
    q = Outcome(solution=None, objective=0.0, details={"n_test": 100})
    gap = headline(MetricDecl(kind="relative_gap"), [m("quantum", -9.0), m("reference", -10.0)], q)
    assert round(gap.accuracy, 6) == 0.9 and gap.accuracy_pm is None
    better = headline(
        MetricDecl(kind="relative_gap"), [m("quantum", -11.0), m("reference", -10.0)], q
    )
    assert better.accuracy == 1.0  # beating the reference is not "more than 100%"
    err = headline(
        MetricDecl(kind="relative_error"),
        [m("quantum", -1.1, stderr=0.01), m("reference", -1.0)],
        q,
    )
    assert round(err.accuracy, 6) == 0.9 and round(err.accuracy_pm or 0, 6) == 0.01
    rate = headline(MetricDecl(kind="error_rate"), [m("quantum", 0.1), m("reference", 0.05)], q)
    assert rate.accuracy == 0.9 and 0.03 < (rate.accuracy_pm or 0) < 0.08


def test_matching_means_equal_inside_a_declared_tolerance_or_inside_the_error_bar() -> None:
    q = Outcome(solution=None, objective=0.0)
    ref = m("reference", -1.0)

    def matches(kind: str, value: float, **kw: object) -> bool:
        decl = MetricDecl(kind=kind, **{k: v for k, v in kw.items() if k.startswith("tolerance")})  # type: ignore[arg-type]
        return headline(decl, [m("quantum", value, stderr=kw.get("stderr")), ref], q).matches  # type: ignore[arg-type]

    # energies are never exactly equal: 1.2 mHa off is not a match by default, but is inside
    # chemical accuracy when the workflow declares it, or inside the measurement's error bar
    assert not matches("relative_error", -1.0012)
    assert matches("relative_error", -1.0012, tolerance_abs=0.0016)
    assert matches("relative_error", -1.0012, stderr=0.001)
    assert not matches("relative_error", -1.0100, stderr=0.001)
    assert matches("relative_gap", -1.5) and not matches("relative_gap", -0.9)


def test_an_error_rate_matches_when_not_worse_or_not_distinguishable() -> None:
    q = Outcome(solution=None, objective=0.0, details={"n_test": 200})
    ref = m("reference", 0.10)
    decl = MetricDecl(kind="error_rate")
    assert headline(decl, [m("quantum", 0.08), ref], q).matches
    assert headline(
        decl, [m("quantum", 0.12), ref], q
    ).matches  # inside the 95% interval on 200 points
    assert not headline(decl, [m("quantum", 0.30), ref], q).matches


def result_with(**kw: object) -> Result:
    methods = [m("quantum", 1.0, dispatches=1, verified="recomputed"), m("reference", 1.0)]
    methods[1] = methods[1].model_copy(update=kw)
    return Result(
        answer={}, objective=1, baseline_objective=1, baseline_solver="r", gap=0,
        uncertainty=Uncertainty(kind="exact"), methods=methods,
        metrics=headline(MetricDecl(), methods, Outcome(solution=None, objective=1)),
    )  # fmt: skip


def test_validation_refuses_what_it_cannot_stand_behind() -> None:
    assert check_quality(Expect(), result_with()) == ""
    assert "claimed" in check_quality(Expect(), result_with(claimed=5.0))
    assert "failed" in check_quality(Expect(), result_with(unavailable="failed: boom"))
    assert check_quality(Expect(), result_with(verified="reported")) == ""  # reported is allowed
    silent = result_with()
    silent.methods[0] = silent.methods[0].model_copy(update={"dispatches": 0})
    assert "sent no circuit" in check_quality(Expect(), silent)


def test_the_workflows_own_estimate_is_held_to_what_was_sent() -> None:
    actual = CircuitStats(qubits=4, depth=5, two_qubit_gates=3, circuits=40, shots=120_000)
    ok, _ = check_estimate(
        Estimate(qubits=4, circuits_on_target=40, shots_on_target=100_000), actual
    )
    assert ok
    bad, why = check_estimate(
        Estimate(qubits=4, circuits_on_target=40, shots_on_target=20_000), actual
    )
    assert not bad and "shots" in why
    none, note = check_estimate(
        Estimate(qubits=4, circuits_on_target=1, shots_on_target=0, by_default=True), actual
    )
    assert none and "conservative" in note
    # a workflow cannot switch the check off by writing the words into its notes
    faked, _ = check_estimate(
        Estimate(qubits=4, circuits_on_target=1, shots_on_target=0, notes=["no estimate"]), actual
    )
    assert not faked


def test_reproducibility_tolerates_floating_point_noise_but_not_a_different_answer() -> None:
    from qsolve.benchmark import _first_difference

    base = {"value": -90.41482819117185, "ranking": [{"name": "EC", "e": -1.5}], "label": "x"}
    noisy = {
        "value": -90.41482819091993,
        "ranking": [{"name": "EC", "e": -1.5 + 1e-13}],
        "label": "x",
    }
    assert _first_difference(base, noisy) is None  # last-digit noise from threaded libraries
    moved = {**base, "value": -90.4148}
    assert _first_difference(base, moved) == "result.value"
    reordered = {**base, "ranking": [{"name": "FEC", "e": -1.5}]}
    assert _first_difference(base, reordered) == "result.ranking[0].name"
    assert _first_difference(base, {**base, "ranking": []}) == "result.ranking"
    assert _first_difference(base, {**base, "label": "y"}) == "result.label"
