# Changelog

## 1.1.1 - 2026-11-20

- The scaffold README linked the schema rules at a page that does not exist; it now links
  [Input and output schemas](https://qsolve.omlabs.co/docs/build/input-and-output-schemas).
- Every version tag now gets a GitHub Release, with its changelog section as the notes.

## 1.1.0 - 2026-11-20

- `qsolve check`: validate, lint and test in the order the platform runs them, stopping at the
  first failing stage. `qsolve test` alone does not run lint, which the platform does.
- `qsolve init`: the scaffold no longer writes the informational `runtime` block (`python`,
  `dependencies`: both defaults, neither read), prints the documentation and `llms.txt` links, and
  its README points to the schema rules.
- The `fidelity_checked` gate failure says `fake:fez@1.6` needs no token and runs on the CPU.

## 1.0.0 - 2026-11-20

First public release. The SDK is the bridge between a Qiskit workflow and the Qsolve platform. It
contains no quantum algorithm: a workflow brings its own circuits, error mitigation and classical
references.

- `Workflow` contract: `quantum`, `reference`, `classical_<id>`, `evaluate`, `answer`, optional
  `estimate` and `summary`; every method is declared in `qsolve.yaml`. Every method's answer, not
  only the quantum one, is checked against the output schema; a bad classical answer marks that
  method unavailable.
- Qiskit-native execution: `QsolveBackend` (a `BackendV2`) and `QsolveSampler` / `QsolveEstimator`
  (V2 primitives) over CPU and GPU simulators, noise-model rehearsal (`fake:<device>[@scale]`) and
  IBM Quantum hardware (`ibm:<device>`). A sampler result always holds exactly the requested number
  of shots, even when a device rounds the count up.
- Measurement at the boundary: circuits, shots, depth, two-qubit gates and raw fidelity against the
  ideal output, computed from what crossed the backend. Error mitigation is workflow-reported and
  labelled as such.
- Re-evaluation of every method's answer with the workflow's own `evaluate`, and a platform-style
  comparison against the declared reference.
- `qsolve` command line: `init`, `validate`, `lint`, `run`, `estimate`, `test`. `init` scaffolds a
  working QAOA workflow that passes every gate, including an Apache-2.0 `LICENSE`. Errors print a
  message, not a traceback.
- Lint codes for secrets, sandbox-forbidden imports, manifest and schema mistakes, `shots` above
  `MAX_SHOTS_PER_CIRCUIT` (100000) and `cases.yaml` content.
- Manifest checks reject a parameter with `min > max`, an `enum` without `choices` and NaN or
  infinite bounds.
- Public API and compatibility policy: see the README. `qsolve.__all__` is stable under semantic
  versioning.
