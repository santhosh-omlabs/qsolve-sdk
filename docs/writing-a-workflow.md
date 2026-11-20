# Writing a workflow

A workflow is a Qiskit program with a manifest. This guide builds one from the scaffold; the
[contract](workflow-contract.md) is the reference.

## 1. Start from the scaffold

```bash
qsolve init warehouse-picker --summary "Pick which depots to open"
cd warehouse-picker
qsolve test .          # it already passes every gate
```

The scaffold solves a small QUBO with QAOA in about 120 lines of Qiskit. Read `src/workflow.py` first.

## 2. Model your problem (`__init__`, `size`, `evaluate`)

Parse the input once in `__init__` and raise `WorkflowError` with a message for the person who
submitted it. `size()` is the number of qubits or variables; the manifest's `envelope` rejects larger
inputs before any credits are spent. `evaluate(solution)` returns the objective of a solution (lower is
better) and raises `InvalidSolution` for one that breaks a rule. Qsolve calls it on every method's
answer, so keep it exact and fast.

## 3. The quantum method

Write it in Qiskit: build the circuit, train the parameters however you like (a local statevector
simulation is your own business and is not metered), then run what you want measured through `ctx`:

```python
counts = ctx.sampler.run([(circuit, parameter_values, shots)]).result()[0].data.meas.get_counts()
```

Return an `Outcome` with the best solution and its objective, all sampled outcomes as `samples`, and
`stderr` if the value is a measurement. Qsolve records what you send, so the circuit's size and
fidelity are measured for you.

If you mitigate errors (readout correction, post-selection, zero-noise extrapolation, anything), say
so: `ctx.report_mitigation(circuits=[...], distribution=..., methods=[...], kept=..., steps=[...])`.
The figures are labelled *reported*, next to the raw ones Qsolve measured.

## 4. The classical methods

`reference` is what you are judged against: use the strongest standard method that is practical (an
exact solver where the size allows it). Add any number of `classical_<id>` methods. They receive a
context that refuses circuits. Raise `MethodUnavailable("why")` when one does not apply to an input.

## 5. Declare and describe the methods (`qsolve.yaml`)

Every method needs an `id`, a `role` and a plain-language `title`, `what`, `strength` and `limit`.
Qsolve shows that text and never writes its own about your workflow. Pick the `metric` kind that fits
your objective, say in `note` what accuracy means, and set a tolerance if exact equality is not the right
standard (chemical accuracy, say).

## 6. Answer, estimate, test

- `answer(outcome)` returns the domain answer people read, validated by `schemas/output.json`. Give
  its properties `title`s and units: the platform renders the answer from them.
- `estimate(target)` returns the circuits and shots you expect on a target. It is held to what was sent.
- `benchmarks/cases.yaml`: a few cases with known answers, and one on `fake:fez@1.6` with a
  `min_fidelity` so error mitigation is validated where noise acts. Optionally, up to three
  `hardware_cases:` (same fields, with `target: ibm:<device>`) for a real device: `qsolve test` and
  validation ignore them; once published, you can run them from the workflow page at your own cost and
  the results join the track record. Needs `qpu: [ibm]` under `targets`.

```bash
qsolve validate . && qsolve lint . && qsolve test .
```

Push to a public repository and submit it on the [Qsolve platform](https://qsolve.omlabs.co).

## Hardware realities

Today's devices are noisy: fidelity falls with circuit depth and two-qubit gate count, so keep circuits
shallow. One batched job beats many (every `sampler.run` call is one batch). Adding shots does not fix
bias. Rehearse on `fake:<device>@1.6` (an empirical factor: real chips have tended to be noisier than their
published models) before spending hardware time.
