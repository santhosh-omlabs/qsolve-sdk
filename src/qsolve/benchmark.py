"""Benchmark and validation gates for a workflow. Used by `qsolve test` locally and by the
platform's validation pipeline, so a workflow passes locally iff it passes on the platform."""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .manifest import LoadedWorkflow, load_workflow
from .results import CircuitStats, Estimate, Quality
from .runner import RunOutput, WorkflowError, _stable, estimate_run, run_workflow, validate_input


class Expect(BaseModel):
    """What a benchmark case requires of its result (the `expect:` block in `cases.yaml`).

    Attributes:
        max_gap: Allowed relative gap to the reference; 0 means it must match.
        max_abs_error: If set, `|objective - reference|` must be at most this instead of `max_gap`.
        min_fidelity: Minimum mitigated fidelity (0-1) on the case's target.
        min_postselection: Minimum share of shots kept (0-1).
        max_two_qubit_gates: Ceiling on the circuit's two-qubit gates.
    """

    model_config = ConfigDict(extra="forbid")
    max_gap: float = 0.0  # allowed (objective - baseline) / |baseline|; 0 = must match optimum
    max_abs_error: float | None = None  # if set, |objective - baseline| must be <= this instead
    # error mitigation, checked on the target the case runs on (see Case.target)
    min_fidelity: float | None = None  # mitigated fidelity vs the ideal output, 0-1
    min_postselection: float | None = None  # fraction of shots that pass the checks, 0-1
    max_two_qubit_gates: int | None = None  # circuit size ceiling


class Case(BaseModel):
    """One benchmark case (an entry under `cases:` or `hardware_cases:` in `cases.yaml`).

    Give the input as `input_file` (a path inside the workflow) or inline as `input`.

    Attributes:
        name: Label shown in reports.
        input_file / input: Where the input comes from.
        params: Parameter overrides.
        seed: Seed for the run.
        target: Run on this target instead of the test's (for example `fake:fez@1.6`).
        expect: The requirements (`Expect`).
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    input_file: str | None = None
    input: dict[str, Any] | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    seed: int = 0
    # Run this case on another target, e.g. "fake:fez@1.6" (a device's noise model, free): the
    # mitigation numbers are only meaningful where there is noise to mitigate.
    target: str | None = None
    expect: Expect = Field(default_factory=Expect)


class CaseResult(BaseModel):
    """The outcome of one benchmark case: pass or fail, and the measured numbers."""

    name: str
    passed: bool
    target: str | None = None
    gap: float | None = None
    abs_error: float | None = None
    objective: float | None = None
    baseline_objective: float | None = None
    fidelity: float | None = None  # mitigated fidelity vs the ideal output
    postselection: float | None = None
    exec_time_s: float | None = None
    shots: int | None = None
    error: str | None = None


class Gate(BaseModel):
    """One validation gate: its name, whether it passed, and a one-line detail."""

    name: str
    passed: bool
    detail: str = ""


class BenchmarkReport(BaseModel):
    """The full report of `run_benchmarks`: every gate, every case, and an overall verdict."""

    workflow: str
    version: str
    target: str
    passed: bool
    gates: list[Gate]
    cases: list[CaseResult]
    total_time_s: float


HARDWARE_TARGET = re.compile(r"^ibm:[A-Za-z0-9_.-]{1,60}$")
MAX_HARDWARE_CASES = 3


def load_hardware_cases(root: str | Path) -> list[Case]:
    """The cases listed under `hardware_cases:` in benchmarks/cases.yaml. They are never run by
    `qsolve test` or by validation: the platform runs them on a real device when the workflow's
    developer asks, at the developer's own cost."""
    path = Path(root) / "benchmarks" / "cases.yaml"
    if not path.is_file():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [Case.model_validate(c) for c in raw.get("hardware_cases", [])]


def _load_cases(wf: LoadedWorkflow) -> list[Case]:
    path = wf.root / "benchmarks" / "cases.yaml"
    if not path.is_file():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [Case.model_validate(c) for c in raw.get("cases", [])]


def _case_input(wf: LoadedWorkflow, case: Case) -> dict[str, Any]:
    if case.input is not None:
        return case.input
    if case.input_file is None:
        raise WorkflowError(f"case {case.name!r} needs input or input_file")
    p = (wf.root / case.input_file).resolve()
    if wf.root not in p.parents:
        raise WorkflowError(f"case {case.name!r}: input_file escapes workflow directory")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise WorkflowError(f"case {case.name!r}: input must be a JSON object")
    return data


# share of the quantum method's shots that must be scorable for a fidelity bound to mean anything
MIN_COVERAGE = 0.9


def check_quality(expect: Expect, r: Any) -> str:
    """Why a result misses what the platform requires of every result, or the case's expectations;
    "" when it does not. Required: the SDK measured the quantum method and the reference, the
    quantum method really sent circuits to the backend, and no method's number contradicts its own
    solution."""
    if r.metrics is None:
        return "no headline metric: quantum and reference could not be compared"
    for meth in r.methods:
        if meth.unavailable.startswith("failed:"):
            return f"method {meth.id} {meth.unavailable}"
        if meth.claimed is not None:
            return f"method {meth.id} claimed {meth.claimed:.6g}, its solution gives {meth.value}"
        if meth.role == "quantum" and meth.dispatches == 0:
            return f"quantum method {meth.id} sent no circuit to the backend"
    q = r.quality
    if q is None:
        q = Quality()
    m = q.mitigation
    if expect.min_fidelity is not None:
        if m is None:
            return "no mitigation numbers reported"
        if m.coverage < MIN_COVERAGE:
            return (
                f"only {m.coverage:.0%} of the quantum method's shots could be scored against an "
                "ideal output (circuits over 20 qubits or with mid-circuit measurement cannot)"
            )
        if m.fidelity_mitigated < expect.min_fidelity:
            return f"fidelity {m.fidelity_mitigated:.3f} < {expect.min_fidelity}"
    if expect.min_postselection is not None:
        if m is None:
            return "no mitigation numbers reported"
        if m.postselection_rate < expect.min_postselection:
            return f"post-selection {m.postselection_rate:.3f} < {expect.min_postselection}"
    if expect.max_two_qubit_gates is not None:
        if q.circuit is None:
            return "no circuit size reported"
        if q.circuit.two_qubit_gates > expect.max_two_qubit_gates:
            return f"{q.circuit.two_qubit_gates} two-qubit gates > {expect.max_two_qubit_gates}"
    return ""


def check_estimate(est: Estimate, actual: CircuitStats | None) -> tuple[bool, str]:
    """Whether the workflow's own estimate of its run matches what the SDK metered."""
    if est.by_default:
        return True, "no estimate() given: quotes use a conservative default"
    if actual is None:
        return True, "no circuits were sent"
    for label, e, a in (
        ("circuits", est.circuits_on_target, actual.circuits),
        ("shots", est.shots_on_target, actual.shots),
    ):
        if a and not 0.5 <= e / max(a, 1) <= 2.0:
            return False, f"estimated {e} {label}, the run sent {a}"
    return True, "estimate within 2x of what was sent"


# Threaded numerical libraries (BLAS, PySCF) sum in a different order from run to run, so a result
# recomputed from scratch can differ in its last digits. The platform's runtime pins one thread and
# is exact; on a developer's multi-core machine a difference this small is not a different answer.
FLOAT_NOISE = 1e-8


def _first_difference(a: Any, b: Any, path: str = "result") -> str | None:
    """The path of the first value that differs by more than floating-point noise, else None."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return path
        for k in a:
            if (w := _first_difference(a[k], b[k], f"{path}.{k}")) is not None:
                return w
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            if (w := _first_difference(x, y, f"{path}[{i}]")) is not None:
                return w
        return None
    if isinstance(a, float) and isinstance(b, float):
        return None if math.isclose(a, b, rel_tol=FLOAT_NOISE, abs_tol=1e-12) else path
    return None if a == b else path


def run_benchmarks(workflow_dir: str | Path, target: str = "sim-cpu") -> BenchmarkReport:
    """Run the validation gates and every benchmark case of a workflow.

    This is what `qsolve test` and the platform's validation run.

    Args:
        workflow_dir: The workflow directory (holding `qsolve.yaml`).
        target: Target for cases that set no `target` of their own.

    Returns:
        A `BenchmarkReport`; `passed` is true when every gate passed.

    Raises:
        FileNotFoundError: no `qsolve.yaml`.
        ValueError, pydantic.ValidationError: the manifest or its schemas are invalid.
    """
    t0 = time.perf_counter()
    gates: list[Gate] = []
    wf = load_workflow(workflow_dir)  # raises on an invalid manifest or schema paths
    gates.append(
        Gate(name="manifest", passed=True, detail=f"{wf.manifest.name} v{wf.manifest.version}")
    )

    ex_errors: list[str] = []
    for ex in wf.manifest.inputs.examples:
        try:
            validate_input(wf, json.loads((wf.root / ex).read_text(encoding="utf-8")))
        except (WorkflowError, OSError, ValueError) as e:
            ex_errors.append(f"{ex}: {e}")
    gates.append(
        Gate(
            name="example_inputs",
            passed=not ex_errors and bool(wf.manifest.inputs.examples),
            detail="; ".join(ex_errors) or f"{len(wf.manifest.inputs.examples)} valid",
        )
    )

    cases = _load_cases(wf)
    gates.append(
        Gate(name="benchmark_cases", passed=bool(cases), detail=f"{len(cases)} cases defined")
    )

    results: list[CaseResult] = []
    first_run: RunOutput | None = None
    first_case: Case | None = None
    for case in cases:
        try:
            out = run_workflow(
                str(wf.root),
                _case_input(wf, case),
                target=case.target or target,
                params=case.params,
                seed=case.seed,
            )
        except Exception as e:
            results.append(
                CaseResult(name=case.name, passed=False, error=f"{type(e).__name__}: {e}")
            )
            continue
        r = out.result
        mit = r.quality.mitigation if r.quality else None
        abs_err = abs(r.objective - r.baseline_objective)
        if case.expect.max_abs_error is not None:
            ok = abs_err <= case.expect.max_abs_error
        else:
            ok = r.gap <= case.expect.max_gap + 1e-9
        why = check_quality(case.expect, r)
        ok = ok and not why
        results.append(
            CaseResult(
                name=case.name,
                passed=ok,
                target=case.target or target,
                error=why or None,
                gap=r.gap,
                abs_error=abs_err,
                objective=r.objective,
                baseline_objective=r.baseline_objective,
                fidelity=mit.fidelity_mitigated if mit else None,
                postselection=mit.postselection_rate if mit else None,
                exec_time_s=out.provenance.exec_time_s,
                shots=out.provenance.shots,
            )
        )
        if first_run is None:
            first_run, first_case = out, case

    declared = len(wf.manifest.methods)
    gates.append(
        Gate(
            name="methods_declared",
            passed=declared >= 2,
            detail=f"{declared} methods declared"
            if declared >= 2
            else "no methods declared: add `methods:` (a quantum, a reference, any classical)",
        )
    )
    noisy = [
        c
        for c in cases
        if c.expect.min_fidelity and (c.target or target).startswith(("fake:", "ibm:"))
    ]
    gates.append(
        Gate(
            name="fidelity_checked",
            passed=bool(noisy),
            detail="a case sets min_fidelity on a noisy target"
            if noisy
            else "no case sets expect.min_fidelity on a noisy target (`target: fake:fez@1.6`): "
            "mitigation is never validated where noise acts. No token needed: fake:fez@1.6 is "
            "Qiskit Aer on your CPU with IBM Fez's noise model",
        )
    )
    gates.append(
        Gate(
            name="cases_pass",
            passed=bool(results) and all(r.passed for r in results),
            detail=f"{sum(r.passed for r in results)}/{len(results)} passed",
        )
    )

    # Reproducibility gate: same seed + inputs must yield the same result.
    if first_run is not None and first_case is not None:
        try:
            again = run_workflow(
                str(wf.root),
                _case_input(wf, first_case),
                target=first_case.target or target,  # the same target the first run used
                params=first_case.params,
                seed=first_case.seed,
            )
        except Exception as e:
            gates.append(
                Gate(
                    name="reproducible",
                    passed=False,
                    detail=f"the rerun with the same seed failed: {type(e).__name__}: {e}",
                )
            )
        else:
            if again.provenance.result_hash == first_run.provenance.result_hash:
                same, detail = True, "identical result hash"
            else:
                where = _first_difference(_stable(first_run.result), _stable(again.result))
                same = where is None
                detail = (
                    f"identical up to floating-point noise (relative {FLOAT_NOISE:g})"
                    if same
                    else f"result differs under same seed at {where}"
                )
            gates.append(Gate(name="reproducible", passed=same, detail=detail))

        try:
            est = estimate_run(
                str(wf.root), _case_input(wf, first_case), target=first_case.target or target,
                params=first_case.params,
            )  # fmt: skip
        except Exception as e:
            gates.append(
                Gate(
                    name="estimate_honest",
                    passed=False,
                    detail=f"estimate() failed: {type(e).__name__}: {e}",
                )
            )
        else:
            ok, why = check_estimate(
                est, first_run.result.quality.circuit if first_run.result.quality else None
            )
            gates.append(Gate(name="estimate_honest", passed=ok, detail=why))

    return BenchmarkReport(
        workflow=wf.manifest.name,
        version=wf.manifest.version,
        target=target,
        passed=all(g.passed for g in gates),
        gates=gates,
        cases=results,
        total_time_s=time.perf_counter() - t0,
    )
