"""The harness: runs a workflow's methods through the SDK and measures everything.

The platform executor runs this same function inside the sandbox, so what a developer sees locally
is what the platform runs. The workflow is untrusted Qiskit code: it gets a Qiskit backend and
primitives, and everything that matters is measured here (see `comparison` and `metering`).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
from pydantic import BaseModel
from threadpoolctl import threadpool_limits

from . import metering
from ._version import __version__
from .backends import Backend, get_backend
from .bridge import QsolveBackend, QsolveEstimator, QsolveSampler
from .comparison import headline, run_methods
from .events import Sink, capture, emit
from .manifest import LoadedWorkflow, load_workflow
from .results import Estimate, Provenance, Result, Uncertainty
from .sim import RUN_OVERHEAD_S
from .workflow import Context, Outcome, Workflow, WorkflowError

__all__ = ["RunOutput", "WorkflowError", "estimate_run", "run_workflow", "validate_input"]


class RunOutput(BaseModel):
    """What `run_workflow` returns: the user-facing `result` and the immutable `provenance`."""

    result: Result
    provenance: Provenance


# Floats are hashed on an absolute grid of 1e-6: far finer than any shot-noise or physical
# difference worth detecting, yet well above library-level numerical noise (about 1e-14 to 1e-10,
# which derived ratios can amplify). A finer grid makes the reproducibility gate flaky.
HASH_DECIMALS = 6


def _round_floats(obj: Any, decimals: int = HASH_DECIMALS) -> Any:
    """Round floats to an absolute grid so numerical noise does not change hashes."""
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, dict):
        return {k: _round_floats(v, decimals) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_round_floats(v, decimals) for v in obj]
    return obj


def _hash(obj: Any) -> str:
    blob = json.dumps(
        _round_floats(obj), sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(blob).hexdigest()


def _stable(result: Result) -> dict[str, Any]:
    """The result as hashed for reproducibility: only wall-clock times differ run to run."""
    d: dict[str, Any] = result.model_dump(mode="json")
    for m in d.get("methods", []):
        m["seconds"] = None
    return d


def validate_input(wf: LoadedWorkflow, input: dict[str, Any]) -> None:
    """Check an input against the workflow's input schema.

    Raises:
        WorkflowError: the input breaks the schema; the message names the path of the first problem.
    """
    try:
        jsonschema.validate(input, wf.input_schema)
    except jsonschema.ValidationError as e:
        path = ".".join(str(p) for p in e.absolute_path) or "<root>"
        raise WorkflowError(f"Invalid input at {path}: {e.message}") from e


def _make_importable(root: Path, top: str) -> None:
    """Make `import <top>` find this workflow's code, even when another workflow with the same
    package name was imported earlier in the process (every workflow calls its package `src`)."""
    if sys.path[:1] != [str(root)]:
        if str(root) in sys.path:
            sys.path.remove(str(root))
        sys.path.insert(0, str(root))
    for name in [n for n in sys.modules if n == top or n.startswith(top + ".")]:
        file = getattr(sys.modules[name], "__file__", None)
        if file and not Path(file).resolve().is_relative_to(root.resolve()):
            del sys.modules[name]


def _instantiate(wf: LoadedWorkflow, input: dict[str, Any], params: dict[str, Any]) -> Workflow:
    """Import the workflow class and build it from the input; enforce the envelope."""
    m = wf.manifest
    module, attr = m.entrypoint.split(":", 1)
    _make_importable(wf.root, module.split(".")[0])
    try:
        cls = getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError) as e:
        raise WorkflowError(f"Cannot load the workflow {m.entrypoint!r}: {e}") from e
    if not (isinstance(cls, type) and issubclass(cls, Workflow)):
        raise WorkflowError(f"{m.entrypoint!r} must be a subclass of qsolve.Workflow")
    try:
        instance: Workflow = cls(input, params)
    except WorkflowError:
        raise
    except (ValueError, KeyError, TypeError, IndexError) as e:
        # the constructor sees user data; malformed data must surface as a clear user error
        raise WorkflowError(f"Could not set up the problem from this input: {e}") from e
    n = instance.size()
    if n > m.envelope.max_variables or n > m.envelope.max_qubits:
        raise WorkflowError(
            f"Problem has {n} variables; this workflow supports at most "
            f"{min(m.envelope.max_variables, m.envelope.max_qubits)} ({m.envelope.notes})"
        )
    return instance


def estimate_run(
    workflow_dir: str,
    input: dict[str, Any],
    *,
    target: str = "sim-cpu",
    params: dict[str, Any] | None = None,
) -> Estimate:
    """Validate and size a run without executing it (used for quotes and credit holds)."""
    wf = load_workflow(workflow_dir)
    try:
        resolved = wf.manifest.resolve_params(params)
    except ValueError as e:
        raise WorkflowError(str(e)) from e
    validate_input(wf, input)
    instance = _instantiate(wf, input, resolved)
    est = instance.estimate(target)
    if est is None:
        return Estimate(
            qubits=instance.size(),
            circuits_on_target=1,
            shots_on_target=int(resolved.get("shots", 0)),
            notes=["the workflow gives no estimate; using a conservative default"],
            by_default=True,
        )
    est.by_default = False  # whatever the workflow set: only the SDK says an estimate is a default
    if est.runtime_s is not None:  # the workflow predicts its own work; add the run around it
        est.runtime_s = round(est.runtime_s + RUN_OVERHEAD_S, 1)
    return est


def _source(inner: Backend) -> str:
    if inner.target != "qpu":
        return "ideal_simulator"
    return "noise_model" if str(inner.device).startswith("fake") else "hardware"


def _context(
    inner: Backend, rec: metering.Recorder, seed: int, target: str, *, quantum: bool
) -> Context:
    backend = QsolveBackend(inner, rec, seed=seed, allow_quantum=quantum)

    def report(
        circuits: list[int],
        distribution: dict[str, float],
        methods: list[str] | tuple[str, ...] = (),
        kept: float | None = None,
        steps: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    ) -> None:
        ids = [int(i) for i in circuits]
        sent = len(rec.dispatches)
        if not ids or any(i < 0 or i >= sent for i in ids):
            raise WorkflowError(
                f"report_mitigation cites circuits {ids}, but {sent} were sent so far "
                "(a dispatch id is in pub_result.metadata['qsolve'][k]['dispatch'])"
            )
        dist = {str(k): float(v) for k, v in dict(distribution).items()}
        if (
            not dist
            or any(not math.isfinite(v) or v < 0 for v in dist.values())
            or not sum(dist.values()) > 0
        ):
            raise WorkflowError(
                "report_mitigation: the distribution must hold finite, non-negative weights"
            )
        if kept is not None and not (math.isfinite(kept) and 0.0 <= kept <= 1.0):
            raise WorkflowError("report_mitigation: `kept` is a share of shots between 0 and 1")
        rec.report(ids, dist, list(methods), kept, list(steps))

    return Context(
        seed=seed, target=target, backend=backend, sampler=QsolveSampler(backend),
        estimator=QsolveEstimator(backend), emit=emit, report_mitigation=report,
    )  # fmt: skip


def run_workflow(
    workflow_dir: str,
    input: dict[str, Any],
    *,
    target: str = "sim-cpu",
    params: dict[str, Any] | None = None,
    seed: int = 0,
    backend: Backend | None = None,
    commit_sha: str | None = None,
    image_digest: str | None = None,
    on_event: Sink | None = None,
) -> RunOutput:
    """Run every declared method on one input, measure everything and build the result.

    The platform runs this same function in its sandbox.

    Args:
        workflow_dir: Directory holding `qsolve.yaml`.
        input: The input, validated against the input schema.
        target: `sim-cpu`, `sim-gpu`, `fake:<device>[@scale]` or `ibm:<device>`.
        params: Parameter overrides; defaults fill the rest.
        seed: Seed for the run.
        backend: Use this backend instead of resolving `target`.
        commit_sha, image_digest: Recorded in the provenance.
        on_event: Receives progress events.

    Returns:
        A `RunOutput`.

    Raises:
        WorkflowError: bad input, parameters or target; a failed constructor; no usable quantum
            or reference value; a failed `answer()`; or an answer that breaks the output schema.
    """
    wf = load_workflow(workflow_dir)
    m = wf.manifest
    try:
        resolved = m.resolve_params(params)
    except ValueError as e:
        raise WorkflowError(str(e)) from e
    validate_input(wf, input)
    try:
        inner = backend or get_backend(target)
    except (ValueError, RuntimeError, ImportError) as e:  # bad target, no IBM token, no extra
        raise WorkflowError(str(e)) from e

    started = datetime.now(UTC)
    t0 = time.perf_counter()
    rec = metering.Recorder()
    # One BLAS thread: summation order is then fixed, so the same seed gives bit-identical results
    # on a laptop and in the sandbox. These workloads are small; reproducibility matters more.
    with threadpool_limits(limits=1), capture(on_event):
        emit("stage", stage="build")
        instance = _instantiate(wf, input, resolved)
        emit("stage", stage="run", device=inner.device, target=inner.target)
        methods, outcomes = run_methods(
            instance, m.methods,
            quantum_ctx=_context(inner, rec, seed, target, quantum=True),
            classical_ctx=_context(inner, rec, seed, target, quantum=False),
            recorder=rec, output_schema=wf.output_schema,
        )  # fmt: skip
        emit("stage", stage="compare")
        q_decl = next(d for d in m.methods if d.role == "quantum")
        r_decl = next(d for d in m.methods if d.role == "reference")
        q = next(x for x in methods if x.id == q_decl.id)
        ref = next(x for x in methods if x.id == r_decl.id)
        if q.value is None or ref.value is None:
            why = q.unavailable or ref.unavailable or "no value"
            raise WorkflowError(f"the quantum method or the reference produced no result: {why}")
        q_out, r_out = outcomes[q_decl.id], outcomes[r_decl.id]
        metrics = headline(m.metric, methods, q_out)
        try:
            answer = instance.answer(q_out)
        except Exception as e:
            raise WorkflowError(
                f"answer() failed on the quantum result: {type(e).__name__}: {e}"
            ) from e
        quality = metering.quality(rec, q_decl.id, _source(inner))

    # The workflow's last chance to run code before the answer is checked; nothing it does after
    # this point can change what was validated.
    gap = float((q.value - ref.value) / max(abs(ref.value), 1e-12))
    default = (
        f"{q.title}: {q.value:.6g}; {ref.title}: {ref.value:.6g} (accuracy {metrics.accuracy:.2%})."
    )
    summary = str(instance.summary(q_out, r_out) or default)[:500]
    try:
        jsonschema.validate(answer, wf.output_schema)
    except jsonschema.ValidationError as e:
        raise WorkflowError(f"Workflow produced an invalid answer: {e.message}") from e
    result = Result(
        answer=answer, objective=q.value, baseline_objective=ref.value, baseline_solver=ref.title,
        gap=gap,
        uncertainty=Uncertainty(
            kind="sampling" if q.stderr is not None else "exact", stderr=q.stderr,
            note="one standard deviation on the quantum value" if q.stderr is not None else "",
        ),
        summary=summary, quality=quality, methods=methods, metrics=metrics,
    )  # fmt: skip
    prov = Provenance(
        run_id=str(uuid.uuid4()), workflow=m.name, version=m.version, commit_sha=commit_sha,
        image_digest=image_digest, sdk_version=__version__, input_hash=_hash(input),
        params=resolved, seed=seed, target=inner.target, provider=inner.provider,
        device=inner.device, calibration_snapshot=inner.calibration_snapshot(),
        shots=quality.circuit.shots if quality.circuit else 0,
        exec_time_s=time.perf_counter() - t0, started_at=started.isoformat(),
        finished_at=datetime.now(UTC).isoformat(), result_hash=_hash(_stable(result)),
        solver={"name": q_decl.id, "details": _plain(_outcome_details(q_out))},
    )  # fmt: skip
    return RunOutput(result=result, provenance=prov)


def _outcome_details(out: Outcome) -> dict[str, Any]:
    return {**out.details}


SOLVER_DETAILS_MAX_BYTES = 200_000


def _plain(details: dict[str, Any]) -> dict[str, Any]:
    """The method's details as plain JSON (numpy scalars and arrays converted), or a note when
    they are too large to keep with the receipt."""
    text = json.dumps(details, default=_to_plain)
    if len(text) > SOLVER_DETAILS_MAX_BYTES:
        return {"omitted": f"{len(text)} bytes, over the {SOLVER_DETAILS_MAX_BYTES} kept"}
    plain: dict[str, Any] = json.loads(text)
    return plain


def _to_plain(o: Any) -> Any:
    if hasattr(o, "tolist"):  # numpy arrays and scalars
        return o.tolist()
    return str(o)
