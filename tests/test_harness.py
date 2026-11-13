"""The harness end to end, on a tiny workflow written to a temporary folder."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from qsolve import WorkflowError, estimate_run, run_workflow
from qsolve.benchmark import Expect, check_quality
from qsolve.runner import RunOutput
from support import MANIFEST, Q, write_workflow


def run(
    tmp_path: Path, body: str = "", manifest: dict[str, Any] | None = None, seed: int = 1
) -> RunOutput:
    wf = write_workflow(tmp_path / "wf", body=body, manifest=manifest)
    return run_workflow(str(wf), {"Q": Q}, seed=seed)


def test_a_run_measures_every_method_and_the_quantum_run(tmp_path: Path) -> None:
    out = run(tmp_path)
    r = out.result
    assert [m.id for m in r.methods] == ["random", "exact", "greedy"]
    assert all(m.verified == "recomputed" and m.claimed is None for m in r.methods)
    assert [m.dispatches for m in r.methods] == [1, 0, 0]  # only the quantum method sent circuits
    assert r.quality is not None and r.quality.circuit is not None
    assert (r.quality.circuit.circuits, r.quality.circuit.shots) == (1, 400)
    assert r.quality.mitigation is not None and r.quality.mitigation.source == "ideal_simulator"
    assert r.metrics is not None and r.baseline_solver == "Exhaustive search"
    assert r.methods[0].top_outcomes and r.methods[0].valid_fraction == pytest.approx(1.0)
    assert check_quality(Expect(), r) == ""


def test_the_same_seed_gives_the_same_result_hash(tmp_path: Path) -> None:
    a = run(tmp_path / "a", seed=4)
    b = run(tmp_path / "b", seed=4)
    assert a.provenance.result_hash == b.provenance.result_hash
    assert a.provenance.shots == 400 and a.provenance.target == "sim-cpu"


def test_a_method_that_lies_about_its_objective_is_caught(tmp_path: Path) -> None:
    out = run(
        tmp_path,
        body="""
        def reference(self, ctx):
            return Outcome(solution=[0, 0, 0], objective=-99.0)
        """,
    )
    ref = out.result.methods[1]
    assert ref.value == 0.0 and ref.claimed == -99.0  # the SDK's value, and what was claimed
    assert "claimed" in check_quality(Expect(), out.result)


def test_a_quantum_method_that_sends_no_circuit_fails_validation(tmp_path: Path) -> None:
    out = run(
        tmp_path,
        body="""
        def quantum(self, ctx):
            return self.reference(ctx)
        """,
    )
    assert out.result.methods[0].dispatches == 0
    assert "sent no circuit" in check_quality(Expect(), out.result)


def test_a_classical_method_cannot_use_the_backend(tmp_path: Path) -> None:
    out = run(
        tmp_path,
        body="""
        def classical_greedy(self, ctx):
            qc = QuantumCircuit(1)
            qc.measure_all()
            ctx.backend.run(qc, shots=10)
            return self.reference(ctx)
        """,
    )
    greedy = out.result.methods[2]
    assert greedy.value is None and greedy.unavailable.startswith("failed: QuantumNotAllowed")
    assert "failed" in check_quality(Expect(), out.result)


def test_a_method_can_say_it_does_not_apply(tmp_path: Path) -> None:
    out = run(
        tmp_path,
        body="""
        def classical_greedy(self, ctx):
            from qsolve import MethodUnavailable
            raise MethodUnavailable("needs a bigger problem")
        """,
    )
    assert out.result.methods[2].unavailable == "needs a bigger problem"
    assert check_quality(Expect(), out.result) == ""  # not applicable is not a failure


def test_a_solution_that_breaks_a_rule_is_unavailable_not_a_number(tmp_path: Path) -> None:
    out = run(
        tmp_path,
        body="""
        def classical_greedy(self, ctx):
            return Outcome(solution=[2, 2, 2], objective=-5.0)
        """,
    )
    assert out.result.methods[2].value is None and "one 0 or 1" in out.result.methods[2].unavailable


def test_a_declared_method_must_exist_on_the_workflow_class(tmp_path: Path) -> None:
    manifest = copy.deepcopy(MANIFEST)
    manifest["methods"].append({"id": "extra-one", "role": "classical", "title": "Extra"})
    with pytest.raises(WorkflowError, match="classical_extra_one"):
        run(tmp_path, manifest=manifest)


def test_a_failing_quantum_method_stays_a_crash_with_the_method_named(tmp_path: Path) -> None:
    """Only a deliberate WorkflowError means "bad input". Anything else the quantum method raises (a
    budget stop, a provider failure, a bug) must not be dressed up as one: the platform bills and
    reports the two differently."""
    with pytest.raises(ValueError, match="boom") as e:
        run(
            tmp_path,
            body="""
            def quantum(self, ctx):
                raise ValueError("boom")
            """,
        )
    assert any("quantum method 'random'" in n for n in e.value.__notes__)


def test_a_workflow_error_from_the_quantum_method_stays_one(tmp_path: Path) -> None:
    with pytest.raises(WorkflowError, match="no feasible"):
        run(
            tmp_path,
            body="""
            def quantum(self, ctx):
                raise WorkflowError("no feasible assignment")
            """,
        )


def test_user_mistakes_are_clear_errors(tmp_path: Path) -> None:
    wf = write_workflow(tmp_path / "wf")
    with pytest.raises(WorkflowError, match="square"):
        run_workflow(str(wf), {"Q": [[1, 2, 3], [4, 5, 6]]})
    with pytest.raises(WorkflowError, match="Invalid input"):
        run_workflow(str(wf), {"nope": 1})
    with pytest.raises(WorkflowError, match="supports at most"):
        run_workflow(str(wf), {"Q": [[0] * 8 for _ in range(8)]})
    with pytest.raises(WorkflowError, match="unknown parameters"):
        run_workflow(str(wf), {"Q": Q}, params={"nope": 1})


def test_the_estimate_comes_from_the_workflow_or_defaults_conservatively(tmp_path: Path) -> None:
    wf = write_workflow(tmp_path / "a")
    est = estimate_run(str(wf), {"Q": Q}, params={"shots": 123})
    assert (est.qubits, est.circuits_on_target, est.shots_on_target) == (3, 1, 123)
    bare = write_workflow(
        tmp_path / "b",
        body="""
        def estimate(self, target):
            return None
        """,
    )
    assert any("no estimate" in n for n in estimate_run(str(bare), {"Q": Q}).notes)


def test_progress_is_reported_as_stages(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = []
    wf = write_workflow(tmp_path / "wf")
    run_workflow(str(wf), {"Q": Q}, on_event=events.append)
    assert [e["stage"] for e in events if e["type"] == "stage"] == ["build", "run", "compare"]


def test_a_value_that_is_not_a_finite_number_is_unusable_not_perfect(tmp_path: Path) -> None:
    """NaN compared false against everything, so it used to read as accuracy 100%."""
    with pytest.raises(WorkflowError, match="not a finite number"):
        run(
            tmp_path,
            body="""
            def evaluate(self, solution):
                import math
                return math.nan
            """,
        )


def test_an_enormous_error_bar_does_not_make_a_wrong_answer_match(tmp_path: Path) -> None:
    manifest = copy.deepcopy(MANIFEST)
    manifest["metric"] = {"kind": "relative_error"}
    out = run(
        tmp_path,
        manifest=manifest,
        body="""
        def quantum(self, ctx, _orig=quantum):  # wraps the fixture's own method
            out = _orig(self, ctx)
            out.solution = None
            out.objective = 0.0   # the reference is far from 0
            out.stderr = 1e6
            return out

        def answer(self, outcome):
            return {"bits": [0, 0, 0], "objective": 0.0}
        """,
    )
    assert out.result.metrics.matches is False and out.result.metrics.accuracy < 0.5


def test_an_invalid_error_bar_is_refused(tmp_path: Path) -> None:
    with pytest.raises(WorkflowError, match="error bar"):
        run(
            tmp_path,
            body="""
            def quantum(self, ctx, _orig=quantum):
                out = _orig(self, ctx)
                out.stderr = float("nan")
                return out
            """,
        )


def test_sampled_probabilities_are_normalised_and_a_bad_sample_cannot_kill_the_run(
    tmp_path: Path,
) -> None:
    out = run(
        tmp_path,
        body="""
        def quantum(self, ctx, _orig=quantum):  # wraps the fixture's own method
            out = _orig(self, ctx)
            out.samples = [{"solution": out.solution, "p": 5.0}, {"solution": [0, 0, 0], "p": 9.0},
                           {"solution": "not a bit list", "p": 1.0}, {"p": 1.0}]
            return out
        """,
    )
    q = out.result.methods[0]
    assert q.valid_fraction is not None and q.valid_fraction <= 1.0 + 1e-9
    assert q.p_reference is not None and 0.0 <= q.p_reference <= 1.0


def test_the_wilson_interval_uses_the_side_that_matters() -> None:
    from qsolve.comparison import _wilson_interval

    lo, hi = _wilson_interval(0.02, 100)  # a 2% error rate on 100 cases
    assert 0.005 < lo < 0.01 and 0.06 < hi < 0.08
    assert _wilson_interval(0.0, 50)[0] == 0.0
    assert _wilson_interval(1.7, 50)[1] == pytest.approx(
        1.0
    )  # a rate above 1 is clamped, not a crash


def test_report_mitigation_refuses_circuits_that_were_never_sent(tmp_path: Path) -> None:
    with pytest.raises(WorkflowError, match="report_mitigation cites circuits"):
        run(
            tmp_path,
            body="""
            def quantum(self, ctx, _orig=quantum):
                out = _orig(self, ctx)
                ctx.report_mitigation(circuits=[999], distribution={"000": 1.0})
                return out
            """,
        )


@pytest.mark.parametrize(
    "call",
    [
        'distribution={"000": float("inf")}',
        'distribution={"000": 1.0}, kept=5',
        'distribution={"000": 1.0}, kept=float("nan")',
        "distribution={}",
    ],
)
def test_report_mitigation_refuses_nonsense_numbers(tmp_path: Path, call: str) -> None:
    with pytest.raises(WorkflowError, match="report_mitigation"):
        run(
            tmp_path,
            body=f"""
            def quantum(self, ctx, _orig=quantum):
                out = _orig(self, ctx)
                ctx.report_mitigation(circuits=[0], {call})
                return out
            """,
        )


def test_a_classical_method_that_sends_circuits_anyway_is_caught(tmp_path: Path) -> None:
    """Going round the backend it was given is recorded and refused."""
    out = run(
        tmp_path,
        body="""
        def classical_greedy(self, ctx):
            from qsolve.bridge import QsolveBackend
            qc = QuantumCircuit(1)
            qc.measure_all()
            QsolveBackend(ctx.backend.inner, ctx.backend.recorder).run(qc, shots=10)
            return self.reference(ctx)
        """,
    )
    greedy = next(m for m in out.result.methods if m.id == "greedy")
    assert greedy.value is None and "only the quantum method" in greedy.unavailable


def test_a_classical_answer_that_breaks_the_output_schema_fails_that_method_only(
    tmp_path: Path,
) -> None:
    out = run(
        tmp_path,
        body="""
        def classical_greedy(self, ctx):
            return Outcome(solution=[1, 1, 1], objective=self.evaluate([1, 1, 1]), details={"g": 1})

        def answer(self, outcome):
            good = {"bits": [int(b) for b in outcome.solution],
                    "objective": self.evaluate(outcome.solution)}
            return {"bits": "oops"} if outcome.details.get("g") else good
        """,
    )
    greedy = out.result.methods[2]
    assert greedy.value is None and "output schema" in greedy.unavailable
    assert out.result.methods[0].value is not None and out.result.methods[1].value is not None


def test_a_classical_answer_that_raises_fails_that_method_only(tmp_path: Path) -> None:
    out = run(
        tmp_path,
        body="""
        def classical_greedy(self, ctx):
            return Outcome(solution=[1, 1, 1], objective=self.evaluate([1, 1, 1]), details={"g": 1})

        def answer(self, outcome):
            if outcome.details.get("g"):
                raise KeyError("missing")
            return {"bits": [int(b) for b in outcome.solution],
                    "objective": self.evaluate(outcome.solution)}
        """,
    )
    assert "answer() failed" in out.result.methods[2].unavailable
