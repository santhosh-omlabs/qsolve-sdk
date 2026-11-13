"""Running every declared method and measuring the comparison.

A workflow declares *what* to compare (its methods, in its manifest) and implements each as a
method of its `Workflow` class. This module runs them all and computes every number itself:

* a method that returns a solution has its objective recomputed with the workflow's `evaluate`;
  a mismatch with what the method reported is recorded and fails validation;
* a method without a solution to re-evaluate is labelled `reported`;
* classical methods run with a backend that refuses circuits;
* the headline accuracy is one of a few generic formulas picked in the manifest.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

import jsonschema

from .manifest import MethodDecl, MetricDecl
from .metering import Recorder
from .results import MethodResult, Metrics
from .workflow import (
    Context,
    InvalidSolution,
    MethodUnavailable,
    Outcome,
    Workflow,
    WorkflowError,
)

_EPS = 1e-12
_MISMATCH = 1e-6


def implementation(wf: Workflow, decl: MethodDecl) -> Any:
    """The method of the workflow class that implements `decl`."""
    name = {"quantum": "quantum", "reference": "reference"}.get(
        decl.role, "classical_" + decl.id.replace("-", "_")
    )
    fn = getattr(wf, name, None)
    if not callable(fn):
        raise WorkflowError(
            f"method {decl.id!r} is declared in the manifest "
            f"but the workflow has no `{name}` method"
        )
    return fn


def _verify(wf: Workflow, out: Outcome) -> tuple[float, str, float | None, str]:
    """(value, how it was obtained, what the method claimed if that differed, why unusable)."""
    claimed = float(out.objective)
    if not math.isfinite(claimed):
        return math.nan, "reported", None, "it reported a value that is not a finite number"
    if out.stderr is not None and not (math.isfinite(out.stderr) and out.stderr >= 0):
        return (
            claimed,
            "reported",
            None,
            "it reported an error bar that is not a finite, non-negative number",
        )
    if out.solution is None:
        return claimed, "reported", None, ""
    try:
        value = float(wf.evaluate(out.solution))
    except InvalidSolution as e:
        return (
            math.nan,
            "recomputed",
            claimed,
            str(e) or "its solution breaks a rule of the problem",
        )
    except Exception as e:
        raise WorkflowError(
            f"evaluate() failed on a method's solution: {type(e).__name__}: {e}"
        ) from e
    if not math.isfinite(value):
        return (
            math.nan,
            "recomputed",
            claimed,
            "evaluate() returned a value that is not a finite number",
        )
    off = abs(value - claimed) > _MISMATCH * max(1.0, abs(value))
    return value, "recomputed", (claimed if off else None), ""


def _clean_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Samples with a solution and a finite, non-negative probability, summing to 1."""
    good = []
    for s in samples:
        try:
            p = float(s["p"])
            s["solution"]
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(p) and p >= 0:
            good.append({**s, "p": p})
    total = sum(s["p"] for s in good)
    return [{**s, "p": s["p"] / total} for s in good] if total > 0 else []


def _sampled(
    wf: Workflow, out: Outcome, ref: float | None
) -> tuple[float | None, float | None, list[dict[str, Any]]]:
    """Share of sampled probability on the reference answer and on valid answers, and the most
    frequent outcomes, each evaluated here."""
    samples = _clean_samples(out.samples or [])
    if not samples:
        return None, None, []
    on_ref = valid = 0.0
    for s in samples:
        try:
            v = float(wf.evaluate(s["solution"]))
        except Exception:
            continue
        if not math.isfinite(v):
            continue
        valid += s["p"]
        if ref is not None and v <= ref + 1e-9 * max(1.0, abs(ref)):
            on_ref += s["p"]
    top = sorted(samples, key=lambda s: -s["p"])[:5]
    shown = [
        {
            "outcome": str(s.get("label") or json.dumps(s["solution"], default=str))[:60],
            "p": float(s["p"]),
        }
        for s in top
    ]
    return (on_ref if ref is not None else None), valid, shown


def run_methods(
    wf: Workflow,
    decls: list[MethodDecl],
    *,
    quantum_ctx: Context,
    classical_ctx: Context,
    recorder: Recorder,
    output_schema: dict[str, Any] | None = None,
) -> tuple[list[MethodResult], dict[str, Outcome]]:
    """Every declared method, run and measured. The quantum method runs last.

    With `output_schema`, a reference or classical method whose `answer()` fails or breaks the
    schema is reported as unavailable (with the reason); the quantum answer is checked by the
    runner."""
    ordered = [d for d in decls if d.role != "quantum"] + [d for d in decls if d.role == "quantum"]
    results: dict[str, MethodResult] = {}
    outcomes: dict[str, Outcome] = {}
    for d in ordered:
        meta = {"id": d.id, "role": d.role, "title": d.title, "what": d.what,
                "strength": d.strength, "limit": d.limit}  # fmt: skip
        fn = implementation(wf, d)
        ctx = quantum_ctx if d.role == "quantum" else classical_ctx
        recorder.method = d.id
        t0 = time.perf_counter()
        try:
            out = fn(ctx)
        except MethodUnavailable as e:
            results[d.id] = MethodResult(
                **meta, unavailable=str(e) or "not available for this input"
            )
            continue
        except Exception as e:
            if d.role != "classical" and isinstance(e, WorkflowError):
                raise  # the workflow's own, deliberate "this input cannot be solved"
            if d.role != "classical":
                # Not a statement about the input: a budget stop, a provider failure or a bug. It
                # stays a crash, so the platform bills what was used and reports the right cause.
                e.add_note(f"raised in the {d.role} method {d.id!r}")
                raise
            results[d.id] = MethodResult(**meta, unavailable=f"failed: {type(e).__name__}: {e}")
            continue
        secs = time.perf_counter() - t0
        if d.role != "quantum" and recorder.of(d.id):
            results[d.id] = MethodResult(
                **meta,
                unavailable=(
                    "it sent circuits to the quantum backend, which only the quantum method may do"
                ),
                dispatches=len(recorder.of(d.id)),
            )
            continue
        value, verified, claimed, why = _verify(wf, out)
        if why:
            results[d.id] = MethodResult(
                **meta, unavailable=why, verified=verified, claimed=claimed
            )
            continue
        answer = None
        try:
            answer = wf.answer(out)
        except Exception as e:
            if d.role != "quantum":  # the runner reports the quantum method's own failure
                results[d.id] = MethodResult(
                    **meta, unavailable=f"answer() failed: {type(e).__name__}: {e}"[:300]
                )
                continue
        if answer is not None and d.role != "quantum" and output_schema is not None:
            try:  # every method's answer must fit the output schema, not only the quantum one
                jsonschema.validate(answer, output_schema)
            except jsonschema.ValidationError as e:
                results[d.id] = MethodResult(
                    **meta,
                    unavailable=f"its answer does not match the output schema: {e.message}"[:300],
                )
                continue
        results[d.id] = MethodResult(
            **meta, value=value, stderr=out.stderr, seconds=round(secs, 3), verified=verified,
            claimed=claimed, answer=answer, dispatches=len(recorder.of(d.id)),
        )  # fmt: skip
        outcomes[d.id] = out
    ref_decl = next(d for d in decls if d.role == "reference")
    ref = results[ref_decl.id].value
    for d in decls:
        m = results[d.id]
        if ref is not None and m.value is not None:
            m.vs_reference = (m.value - ref) / max(abs(ref), _EPS)
            if d.id in outcomes and outcomes[d.id].samples:
                m.p_reference, m.valid_fraction, m.top_outcomes = _sampled(wf, outcomes[d.id], ref)
    return [results[d.id] for d in decls], outcomes


def _wilson_interval(p: float, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval of a rate p measured on n cases."""
    p = min(max(p, 0.0), 1.0)
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(centre - half, 0.0), min(centre + half, 1.0)


# An error bar can only excuse a gap while it is small next to the answer: a bar as wide as the
# answer itself says nothing, and a workflow that reports an enormous one must not "match" by it.
_MAX_BAR = 0.25


def headline(decl: MetricDecl, methods: list[MethodResult], quantum: Outcome) -> Metrics:
    """Accuracy against the reference by one of the generic formulas (see MetricDecl)."""
    q = next(m for m in methods if m.role == "quantum")
    ref = next(m for m in methods if m.role == "reference")
    if q.value is None or ref.value is None:
        raise ValueError("the quantum method and the reference must both produce a value")
    scale = max(abs(ref.value), _EPS)
    pm: float | None = None
    bar = q.stderr if q.stderr is not None and q.stderr <= _MAX_BAR * scale else None
    if decl.kind == "relative_gap":
        acc = 1.0 - max((q.value - ref.value) / scale, 0.0)
    elif decl.kind == "relative_error":
        acc = 1.0 - abs(q.value - ref.value) / scale
        pm = q.stderr / scale if q.stderr is not None else None
    else:  # error_rate: the value is a share of mistakes
        acc = 1.0 - q.value
        n = quantum.details.get("n_test")
        n = int(n) if isinstance(n, int | float) and math.isfinite(n) and n >= 1 else 0
        interval = _wilson_interval(q.value, n) if n else None
        if interval:
            pm = max(interval[1] - q.value, q.value - interval[0])
    diff = abs(q.value - ref.value)
    if decl.kind == "error_rate":
        # no more mistakes than the reference (within the declared slack), or a reference rate that
        # the quantum rate's 95% interval cannot rule out
        slack = decl.tolerance + (decl.tolerance_abs or 0.0)
        matches = q.value <= ref.value + slack + 1e-12 or (
            interval is not None and interval[0] <= ref.value + slack
        )
    else:
        matches = (
            acc >= 1.0 - decl.tolerance - 1e-9
            or (decl.tolerance_abs is not None and diff <= decl.tolerance_abs)
            or (bar is not None and diff <= 2.0 * bar)  # inside its own 2-sigma error bar
        )
        if decl.kind == "relative_gap":  # beating the reference is also matching it
            matches = matches or q.value <= ref.value
    return Metrics(kind=decl.kind, accuracy=max(0.0, min(1.0, acc)), accuracy_pm=pm,
                   note=decl.note, tolerance=decl.tolerance, matches=matches)  # fmt: skip
