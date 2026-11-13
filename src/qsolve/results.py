"""Result and provenance objects."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Uncertainty(BaseModel):
    """How sure the headline number is: `kind`, `stderr` and a `note`."""

    kind: str  # "sampling" | "exact" | "heuristic"
    stderr: float | None = None
    note: str = ""


class Mitigation(BaseModel):
    """Fidelity of the quantum run. `source` says where the noise came from: `hardware`,
    `noise_model` (a device's noise model) or `ideal_simulator`. The raw figure is measured by the
    SDK from the recorded circuits and counts. The mitigated figure is what the workflow's own
    post-processing reports (`reported` is true), recomputed by the SDK against the ideal output;
    with no report it equals the raw figure."""

    fidelity_raw: float  # measured vs ideal distribution, before any mitigation
    fidelity_mitigated: float  # after the listed methods
    reported: bool = False  # the mitigated figure comes from the workflow's own report
    fidelity_floor: float  # best a noise-free device scores at this shot count
    postselection_rate: float  # fraction of shots that pass the checks (1.0 = nothing discarded)
    methods: list[str] = Field(default_factory=list)
    source: str
    # share of the quantum method's shots whose circuits could be scored against an ideal output
    # (not circuits over 20 qubits, or with mid-circuit measurement): the rest is not in the figures
    coverage: float = 1.0
    details: dict[str, Any] = Field(default_factory=dict)


class CircuitStats(BaseModel):
    """Size of what the quantum method actually sent to the backend, recorded by the SDK."""

    qubits: int
    depth: int
    two_qubit_gates: int
    circuits: int = 0  # circuits sent in total
    shots: int = 0  # shots over all of them


class Quality(BaseModel):
    """Fidelity and circuit size of the quantum run, measured from what crossed the backend
    boundary (see `qsolve.metering`)."""

    mitigation: Mitigation | None = None
    circuit: CircuitStats | None = None


class MethodResult(BaseModel):
    """One method, quantum or classical, run on this input. Everything numeric is measured by the
    platform's harness; the text (title, what, strength, limit) is declared by the workflow."""

    id: str
    role: str  # quantum | reference | classical
    title: str
    what: str = ""
    strength: str = ""
    limit: str = ""
    value: float | None = None  # the objective this method reached (lower is better)
    stderr: float | None = None
    vs_reference: float | None = None  # (value - reference) / |reference|; 0 means the same
    seconds: float | None = None
    # how the number was obtained: `recomputed` (the SDK evaluated the method's own solution with
    # the workflow's `evaluate`) or `reported` (the workflow's claim, no solution to re-evaluate)
    verified: str = "reported"
    dispatches: int = 0  # circuits this method sent to the quantum backend (quantum: at least 1)
    claimed: float | None = None  # what the method itself reported, when it disagreed
    unavailable: str = ""  # why the method could not run on this input
    # methods that sample: share of samples on the reference answer, and on any valid answer
    p_reference: float | None = None
    valid_fraction: float | None = None
    top_outcomes: list[dict[str, Any]] = Field(default_factory=list)  # most frequent outcomes
    answer: dict[str, Any] | None = None


class Metrics(BaseModel):
    """The headline numbers, computed by the platform from the raw method results."""

    kind: str  # relative_gap | relative_error | error_rate
    accuracy: float  # 0-1: how close the quantum result is to the reference
    accuracy_pm: float | None = None  # its +/- (1 sigma or a 95% interval half-width)
    note: str = ""  # how it was computed, in words (declared by the workflow)
    tolerance: float = 0.0  # declared: accuracy within this of 1 counts as matching
    # the platform's judgement: equal to the reference, inside a tolerance the workflow declared, or
    # statistically indistinguishable from it (inside the measurement's own error bar)
    matches: bool = False


class Result(BaseModel):
    """The final, user-facing result every workflow must produce."""

    answer: dict[str, Any]
    objective: float
    baseline_objective: float
    baseline_solver: str
    gap: float  # (objective - baseline) / max(|baseline|, eps); 0 means matched the baseline
    uncertainty: Uncertainty
    summary: str = ""
    artifacts: dict[str, Any] = Field(default_factory=dict)
    quality: Quality | None = None
    methods: list[MethodResult] = Field(default_factory=list)
    metrics: Metrics | None = None


class Provenance(BaseModel):
    """Immutable record of how a result was produced."""

    run_id: str
    workflow: str
    version: str
    commit_sha: str | None = None
    image_digest: str | None = None
    sdk_version: str
    input_hash: str
    params: dict[str, Any]
    seed: int
    target: str
    provider: str
    device: str
    calibration_snapshot: dict[str, Any] = Field(default_factory=dict)
    shots: int = 0
    exec_time_s: float = 0.0
    started_at: str
    finished_at: str
    result_hash: str
    # how the solver produced its answer (circuit, mitigation, per-system numbers); not hashed,
    # because it carries diagnostics beyond the answer
    solver: dict[str, Any] = Field(default_factory=dict)


class Estimate(BaseModel):
    """Pre-run resource estimate. The platform turns this into a credit quote and hold."""

    qubits: int
    circuits_on_target: int  # circuits submitted to the chosen backend
    shots_on_target: int  # total shots across those circuits
    classical_cost_class: str = "low"  # low | medium | high (local CPU work in the sandbox)
    # Predicted seconds for the whole run on one reference core per simulator thread (see
    # qsolve.sim); excludes time waiting in a hardware provider's queue. None: not predicted.
    runtime_s: float | None = Field(default=None, ge=0)
    notes: list[str] = Field(default_factory=list)
    # set by the SDK, never by the workflow: this is the conservative default, not its own estimate
    by_default: bool = False
