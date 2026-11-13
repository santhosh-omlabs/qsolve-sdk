# Workflow contract (schema_version 1)

The reference for everything Qsolve requires of a workflow. For a guided introduction see
[writing-a-workflow.md](writing-a-workflow.md); for the reasoning, the
[design note](design/sdk-as-bridge.md).

## Repository layout

```
qsolve.yaml            required
schemas/*.json         required: input and output JSON Schemas (draft-07)
src/                   your code (an importable package; keep an __init__.py)
benchmarks/cases.yaml  required to publish: the cases the platform runs and records; an optional
                       `hardware_cases:` list (max 3, `target: ibm:<device>`) is run on a device on request
examples/              example inputs referenced by the manifest
requirements.lock      optional: the dependencies your code needs, pinned (see Platform rules, 2)
```

## `qsolve.yaml`

| Field | Type | Notes |
|---|---|---|
| `schema_version` | `1` | |
| `name`, `version`, `family`, `summary`, `license`, `tags` | | `name`: lowercase letters, digits, dashes; `version`: semver |
| `category` | string | the field it belongs to, as a person would say it ("Finance"); shown in the UI |
| `inputs` | `{schema, examples[]}` | example inputs are validated |
| `parameters[]` | `{name, label?, type, default, min?, max?, choices?, description?}` | `type`: `int`, `float`, `bool`, `str`, `enum` |
| `outputs` | `{schema}` | validates the answer. Give properties a `title` and, optionally, `x-ui: {unit, format: percent}`: the platform renders the answer from them |
| `entrypoint` | `package.module:Class` | your `qsolve.Workflow` subclass |
| `objective` | `{label, unit, decimals}` | what the number every method reaches means |
| `metric` | `{kind, note, tolerance, tolerance_abs}` | how accuracy is computed: `relative_gap` (lower-is-better objective), `relative_error` (distance to the reference, with its error bar), `error_rate` (a share of mistakes, with a 95% interval). `note` explains it in words. A result *matches* when it is equal, within `tolerance`, within `tolerance_abs` (in the objective's units, e.g. 0.0016 Ha), or statistically indistinguishable (inside twice its own error bar, which counts only while it is at most a quarter of the reference value; for an error rate, no worse or when the reference rate lies inside the 95% Wilson interval of yours, with the case count `n_test` from `details`) |
| `methods[]` | `{id, role, title, what, strength, limit}` | exactly one `quantum`, exactly one `reference`, any number of `classical`. The text is for people who do not know the method |
| `envelope` | `{max_qubits, max_variables, notes}` | `size()` is checked against it before any credits are spent |
| `targets` | `{simulators: [cpu, gpu], qpu: [ibm], min_qubits_qpu}` | |
| `resources` | `{timeout_s, memory_mb, network: none}` | enforced by the sandbox |
| `runtime` | `{python, dependencies}` | |

Unknown fields are rejected. `qsolve validate .` checks everything.

## The `Workflow` class

```python
class MyWorkflow(Workflow):
    def __init__(self, input, params)         # parse and validate; raise WorkflowError
    def size(self) -> int
    def quantum(self, ctx) -> Outcome          # role quantum
    def reference(self, ctx) -> Outcome        # role reference
    def classical_<id>(self, ctx) -> Outcome   # role classical (id: dashes become underscores)
    def evaluate(self, solution) -> float      # lower is better; raise InvalidSolution
    def answer(self, outcome) -> dict          # validated by outputs.schema
    def estimate(self, target) -> Estimate     # optional
    def summary(self, quantum, reference) -> str   # optional
```

`Outcome(solution, objective, stderr=None, samples=None, details=None)`: `solution` is JSON data (or
`None` for a result that cannot be re-evaluated, such as a measured energy: it is labelled
`reported`); `samples` lists all outcomes of a sampling method as `{"solution", "p"}`.
Raise `MethodUnavailable("why")` from a classical method that does not apply to an input.

`Context`: `seed`, `target`, `backend` (Qiskit `BackendV2`), `sampler` (`SamplerV2`), `estimator`
(`EstimatorV2`), `emit(kind, **data)` for progress events, `report_mitigation(...)`.

- Send circuits only through `ctx`. Everything sent is recorded. All circuits of one `sampler.run` with
  the same shot count go to the device as one batch (on a QPU, one job).
- Classical methods get a context whose backend refuses circuits.
- Per-circuit device metadata (readout errors per measured bit, `two_qubit_gates`, and `dispatch`, the
  index to cite in a mitigation report) is in `pub_result.metadata["qsolve"][k]`.
- `ctx.report_mitigation(circuits=[dispatch ids], distribution={bitstring: p}, methods=[...],
  kept=<share of shots kept>, steps=[...])` declares the mitigated output of those dispatches.
- The same input, parameters and seed must give the same result.

## Who measures what

| Number | Computed by | Label |
|---|---|---|
| each method's objective | `evaluate(solution)`, run by the SDK | `recomputed` (else `reported`) |
| accuracy, matches, +/- | the SDK, from the method values | measured |
| qubits, depth, two-qubit gates, circuits, shots | the SDK's backend wrapper | measured |
| raw fidelity (and its shot-noise floor) | the SDK: counts against the ideal output of the same circuit, simulated noise-free up to 20 qubits, over the circuits a mitigation report covers and the other circuits that compute something (not mirror or calibration circuits), those that carry the run's noise, weighted by shots; `coverage` is the share of shots that could be scored | measured |
| mitigated fidelity, share kept, steps | the workflow reports; the SDK recomputes the fidelity | `reported` |

## Benchmark cases

```yaml
cases:
  - name: unique-name
    input_file: examples/x.json     # or inline `input:`
    params: {shots: 100000}         # optional parameter overrides
    seed: 1
    target: fake:fez@1.6            # optional: a device's noise model (free)
    expect:
      max_gap: 0.0                  # allowed relative gap to the reference
      max_abs_error: 0.003          # alternative: absolute |objective - reference|
      min_fidelity: 0.97            # measured fidelity vs the ideal output (needs a noisy target)
      min_postselection: 0.9        # share of shots kept, as reported
      max_two_qubit_gates: 40       # circuit size ceiling
```

## Validation gates (`qsolve test`)

| Gate | Passes when |
|---|---|
| `manifest` | manifest and schemas load and validate |
| `example_inputs` | at least one example exists and all validate |
| `benchmark_cases` | at least one case is defined |
| `methods_declared` | the manifest declares the quantum method and the reference |
| `fidelity_checked` | a case sets `min_fidelity`, so the quantum run is validated where noise acts |
| `cases_pass` | every case meets its `expect`; no method contradicts its own solution or crashes; the quantum method sent circuits |
| `reproducible` | re-running the first case with the same seed gives the same result hash |
| `estimate_honest` | the workflow's `estimate()` is within 2x of the circuits and shots actually sent |

The platform runs these gates, plus dependency and security scans (`qsolve lint`) and a sandboxed
re-run, before a workflow is published.

## Backends

| Target | Locally | On the platform |
|---|---|---|
| `sim-cpu`, `sim-gpu` | Aer simulator | executor chooses |
| `ibm:<device>` | your own `QSOLVE_IBM_TOKEN` | executor chooses |
| `fake:<device>[@scale]` | Aer with the device's noise model, free (`fake:fez@1.6`) | benchmarks only |

## Platform rules

1. A reference method is mandatory.
2. Workflows run in the platform's runtime image, not in one built for you (per-workflow images are
   planned). It carries Python 3.12, numpy, scipy, scikit-learn, Qiskit, Qiskit Aer, PySCF. Importing anything else fails in the sandbox gate, so a
   workflow that needs more cannot be published yet. Your own Dockerfile is ignored.
3. No network, no environment secrets, no files outside the repo and scratch directory.
4. Quantum jobs go through the backend you are given; provider credentials never enter the sandbox.
5. Every run is seeded. Use `ctx.seed` for all randomness.
6. The platform, never the workflow, writes metrics and provenance.
