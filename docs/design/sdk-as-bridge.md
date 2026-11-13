# SDK as the bridge (v1.0): workflows are Qiskit programs

Design rationale for SDK 1.0. It replaces the earlier model in which the SDK shipped its own solvers.

## Principle
- A **workflow** is a Qiskit program in its own repo. Its quantum algorithm, circuits, error
  mitigation and classical references are its own code, built on Qiskit (and numpy, scipy, pyscf,
  scikit-learn as it needs).
- The **SDK** is the bridge to Qsolve. It contains no quantum algorithm. It provides:
  1. the contract (manifest + the `Workflow` class),
  2. Qiskit-native execution: a `BackendV2` and Sampler/Estimator primitives that run the workflow's
     circuits on CPU, GPU or QPU through Qsolve,
  3. boundary measurement: every circuit sent and every count returned is recorded by the SDK,
  4. the harness: it runs every declared method, re-evaluates what it can, computes the numbers,
  5. validation, testing and scaffolding (`qsolve init|validate|lint|estimate|run|test`).
- The **platform** (separate repo) runs the harness in its sandbox and owns billing, queues, QPU
  access and the UI. It knows no workflow, field or algorithm.
- Qsolve does not compete with Qiskit. Nothing algorithmic ships in the SDK.

## Trust model (what Qsolve can and cannot guarantee)
All workflow code is untrusted. Qsolve guarantees:
1. **Provenance**: what ran, in a sandbox, reproducibly (seed, inputs, parameters, versions, hashes).
2. **Boundary measurement**: circuits actually sent and counts returned, recorded by an SDK-owned
   backend wrapper (a snapshot of each circuit, taken before it is sent). From these the SDK
   computes qubits, depth, two-qubit gates, shots, and the raw fidelity against the circuit's ideal
   output (noise-free simulation, up to 20 qubits). The fidelity is taken over the circuits a
   mitigation report covers and every other circuit whose ideal output is not a single bit string
   (a mirror or calibration circuit says nothing about the answer; one that computes something and
   is left unreported still counts, at its raw fidelity). Of those, only the circuits that carry the
   run's noise (at least half the largest one's two-qubit operations) are averaged, weighted by
   shots, so a trivial circuit sent alongside carries no weight. Circuits that cannot be scored are
   reported as `coverage`, and a fidelity requirement needs 90% coverage.
3. **Re-evaluation**: a method that returns a solution has its objective recomputed with the
   workflow's `evaluate(solution)`. A method that claims a different number is flagged and fails
   validation. A method without a re-evaluable solution is labelled `reported`. A value or error bar
   that is not a finite number is refused, and an error bar only excuses a gap while it is small.
4. **The quantum method really is quantum**: a method with role `quantum` that sent no circuit to
   the backend fails validation; a classical method that sent one is refused.
5. **Known-answer benchmark cases** and a noise-model case with minimum fidelity, run by the platform.
6. **Honest cost**: the workflow's `estimate()` is compared with the metered actual circuits and shots.
7. **Cross-check against the executor**: the platform's executor runs outside the sandbox and meters
   every circuit that really ran. A run whose result accounts for far fewer shots than the executor
   metered is failed as `integrity_mismatch` and billed for what ran.

It cannot prove that a declared classical baseline is the best classical method, or that a
workflow's error mitigation is legitimate. Mitigated figures are labelled `reported`, next to the
measured raw figures. "Validated" means 1 to 7, not more.

### Known limit: the harness and the workflow share a process
The SDK measures from inside the sandbox, in the same Python process as the workflow code. Code
determined to tamper can find the recorder, the protocol descriptor or the harness's modules and lie
to the SDK (an independent review demonstrated this). What stops it from mattering today:
- the executor, outside the sandbox, meters and bills every circuit, and point 7 catches circuits
  hidden from the SDK's record;
- result frames cannot be forged by writing to file descriptor 1 (it leads to the log stream);
- `evaluate()` is the workflow's own code, so "re-evaluated" shows a method's value is consistent
  with its solution, not that the evaluator is honest. First-party workflows are reviewed by hand.

Closing it for third-party workflows means running the workflow in a separate process that holds no
reference to the harness and reaches the backend only through a pipe the harness owns. That is the
next security step, to take before untrusted workflows are published.

## Contract
`qsolve.yaml` (schema_version 1): as in the earlier format, except `components` is replaced by
`entrypoint: package.module:ClassName`, and each entry of `methods` is implemented as a method of
that class (no `runner`).

```python
from qsolve import Workflow, Outcome, Context, WorkflowError, InvalidSolution

class MyWorkflow(Workflow):
    def __init__(self, input: dict, params: dict) -> None:   # parse and validate; raise WorkflowError
        super().__init__(input, params)
    def size(self) -> int: ...                  # qubits / variables, checked against the envelope
    def quantum(self, ctx: Context) -> Outcome: ...         # role quantum   (Qiskit)
    def reference(self, ctx: Context) -> Outcome: ...       # role reference (classical)
    def classical_<id>(self, ctx: Context) -> Outcome: ...  # one per role classical (id: dashes -> _)
    def evaluate(self, solution) -> float: ...  # objective of a solution; raise InvalidSolution
    def answer(self, solution) -> dict: ...     # the domain answer, validated by outputs.schema
    def estimate(self, target: str) -> Estimate: ...        # optional: circuits and shots to expect
```
- `Outcome(solution, objective, stderr=None, samples=None, details=None)`. `solution` is JSON data.
  `samples` is an optional list of `{"solution": ..., "p": ...}` (top outcomes of a sampling method).
- `Context`: `seed`, `target`, `backend` (Qiskit `BackendV2`), `sampler` (`SamplerV2`),
  `estimator` (`EstimatorV2`), `emit(kind, **data)`, `report_mitigation(...)`. Classical methods
  receive a context whose backend raises: classical means no quantum hardware.
- Circuits sent through `ctx.backend` / `ctx.sampler` / `ctx.estimator` are logical Qiskit circuits;
  the platform transpiles for the device (as before). Readout calibration is available as
  `ctx.readout_errors(circuit)` when the device reports it.
- `ctx.report_mitigation(circuits=[...], distribution=..., methods=[...], kept=..., steps=[...])`
  declares the mitigated output of the given dispatches. The SDK recomputes its fidelity against the
  ideal output and shows it next to the measured raw fidelity, labelled `reported`.

## What is measured, by whom
| Number | Computed by | Label |
|---|---|---|
| each method's objective | `evaluate(solution)` run by the harness | `recomputed` (else `reported`) |
| accuracy, matches, +/- | harness, from the method values | measured |
| qubits, depth, 2q gates, shots, circuits | SDK backend wrapper | measured |
| raw fidelity | SDK, counts vs ideal of the same circuit | measured |
| mitigated fidelity, kept share, steps | workflow reports, SDK recomputes fidelity | `reported` |

## Migration
- SDK 1.0.0 removes `solvers/`, `chem/`, `qml/`, `circuits.py`, `problems.py`, `helpers.py`,
  `interfaces.py`, `quality.py`, `methods`/`components` handling. Its public API is the contract above.
- The seven example workflows become self-contained Qiskit programs: each carries its own algorithm
  code in its `src/` (no shared library), so a newcomer can read one folder and see everything.
- Platform backend: only the SDK boundary changes (`results`, `benchmark`, `runner` entry points).
  `Result`, `MethodResult`, `Metrics`, `Quality` keep their shape, so the API and UI are unchanged
  apart from the label `reported`.
