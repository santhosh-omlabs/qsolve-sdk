# Changelog

## Unreleased

- Lint: `E025` (a `shots` parameter above `MAX_SHOTS_PER_CIRCUIT`, 100000), `E055` to `E059`
  (`cases.yaml` content: input present, `input_file` inside the repository, unique names, no `ibm:`
  target under `cases`, input valid for the schema), `W052` (`min_qubits_qpu` is ignored by the
  platform). `E021` flags only schema keywords, not a property named `pattern`. `.env.example`,
  `.env.sample` and `.env.template` are allowed. A non-UTF-8 entrypoint module is a finding, not a
  crash.
- Manifest: a parameter with `min > max`, an `enum` without `choices`, or a NaN/infinite bound is
  rejected; NaN and infinity never pass a value check; `True` is not the enum choice `1`.
- Every method's answer, not only the quantum one, is checked against the output schema; a bad
  classical answer marks that method unavailable.
- CLI: `run` and `estimate` print a message, not a traceback, for a missing or invalid manifest;
  square brackets survive in output (`fake:<device>[@scale]`, `qsolve-sdk[ibm]`); `init` writes
  the name and summary as quoted YAML.
- `get_backend` rejects unknown keyword arguments; `QsolveBackend` honours `seed_simulator`;
  `FakeHardwareBackend.sample_batch` mixes every circuit's seed into the job seed.
- `qsolve.__all__` also exports `MethodResult`, `Metrics`, `CircuitStats`, `RunOutput` and
  `MAX_SHOTS_PER_CIRCUIT`. The compatibility policy is in the README.

## 1.0.0

The SDK is the bridge between a Qiskit workflow and the Qsolve platform. It contains no quantum
algorithm: a workflow brings its own circuits, error mitigation and classical references.

- `Workflow` contract: `quantum`, `reference`, `classical_<id>`, `evaluate`, `answer`, optional
  `estimate` and `summary`; every method is declared in `qsolve.yaml`.
- Qiskit-native execution: `QsolveBackend` (a `BackendV2`) and `QsolveSampler` / `QsolveEstimator`
  (V2 primitives) over CPU and GPU simulators, noise-model rehearsal (`fake:<device>[@scale]`) and
  IBM Quantum hardware (`ibm:<device>`).
- Measurement at the boundary: circuits, shots, depth, two-qubit gates and raw fidelity against the
  ideal output, computed from what crossed the backend. Error mitigation is workflow-reported and
  labelled as such.
- Re-evaluation of every method's answer with the workflow's own `evaluate`, and a platform-style
  comparison against the declared reference.
- A sampler result always holds exactly the requested number of shots, even when a device rounds
  the count up.
- `qsolve` command line: `init`, `validate`, `lint`, `run`, `estimate`, `test`.
