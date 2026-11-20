# qsolve-sdk

The bridge between a Qiskit workflow and the [Qsolve](https://qsolve.omlabs.co) platform. Qsolve is
a platform where researchers find, benchmark and run quantum and hybrid workflows, judged against
the best classical method. [Documentation](https://qsolve.omlabs.co/docs/build/overview).

You write a workflow in plain Qiskit: your circuits, your algorithm, your error mitigation, your
classical references. This SDK runs it on Qsolve's backends (CPU, GPU, QPU), records every circuit it
sends and every count it gets back, runs every method you declare, re-evaluates the answers, measures
the quantum run and validates the lot. If a workflow passes `qsolve test` on your laptop, it passes
validation on the platform, because the platform runs this same code in its sandbox. (The sandbox
has no network and a fixed set of importable modules; `qsolve lint` tells you what it will refuse,
and [the contract](docs/workflow-contract.md) lists the platform rules.)

The SDK contains no quantum algorithm. Qsolve does not compete with Qiskit: use Qiskit (and its
ecosystem) for the quantum part.

## Install

```bash
pip install qsolve-sdk
pip install "qsolve-sdk[ibm]"    # + the IBM runtime: device noise models and IBM hardware
```

Python 3.12 or newer (the version the Qsolve sandbox runs, so what passes here runs there), Qiskit
2.x. Install the `ibm` extra even if you never touch hardware: the
`fake:<device>` noise-model targets, and the noise-model case in the scaffold's benchmarks, need it.
Running on a real IBM device additionally needs your own token in the `QSOLVE_IBM_TOKEN`
environment variable (optional `QSOLVE_IBM_INSTANCE`). `qsolve --version` prints the installed
version.

## Five-minute tour

```bash
qsolve init my-problem --summary "Pick the best depots"   # a complete Qiskit workflow to start from
cd my-problem
qsolve validate .                          # manifest, schemas, example inputs
qsolve lint .                              # secrets and imports the sandbox would reject
qsolve run . -i examples/qubo_5.json       # run on the CPU simulator
qsolve estimate . -i examples/qubo_5.json  # circuits and shots to expect, no execution
qsolve test .                              # the validation gates and benchmark cases
```

Every command exits non-zero on failure, so `qsolve validate . && qsolve lint . && qsolve test .`
is a complete pre-submission check. The scaffold is a working QAOA workflow that passes every gate;
change it step by step and re-run `qsolve test .` after each step.

`src/workflow.py` is the whole workflow:

```python
from qsolve import Context, Outcome, Workflow


class Qubo(Workflow):
    def __init__(self, input, params): ...  # parse the input; raise WorkflowError for mistakes
    def size(self) -> int: ...  # qubits or variables
    def quantum(self, ctx: Context) -> Outcome:  # plain Qiskit: build circuits, run them via ctx
        counts = ctx.sampler.run([(circuit, params, shots)]).result()[0].data.meas.get_counts()
        ...

    def reference(self, ctx) -> Outcome: ...  # the classical answer you are judged against
    def classical_annealing(self, ctx) -> Outcome: ...  # any other classical method you declare
    def evaluate(
        self, solution
    ) -> float: ...  # the objective of a solution: Qsolve re-checks every method with it
    def answer(self, outcome) -> dict: ...  # the domain answer people read
```

`qsolve.yaml` declares the methods and says in plain words what each one is. Qsolve shows that text
and measures every number itself.

## Targets

| Target | Runs on |
|---|---|
| `sim-cpu` | Qiskit Aer on the CPU (the default) |
| `sim-gpu` | Qiskit Aer on a GPU; needs `qiskit-aer-gpu` |
| `fake:<device>[@scale]` | Aer with an IBM device's published noise model, optionally with every error rate multiplied by `scale`, from 1 to 10 (for example `fake:fez@1.6`); free; needs the `ibm` extra |
| `ibm:<device>` | IBM Quantum hardware, with your own token in `QSOLVE_IBM_TOKEN` (needs the `ibm` extra); spends your own IBM quota |

## What Qsolve measures, and what it does not

All workflow code is untrusted. Qsolve guarantees: provenance (what ran, reproducibly, in a sandbox);
boundary measurement (circuits, shots, depth, two-qubit gates and raw fidelity, computed from what
actually crossed the backend); re-evaluation (each method's objective recomputed with your
`evaluate`; a contradiction fails validation); that the quantum method really sent circuits; and honest
cost (your `estimate()` is compared with what was sent). It cannot prove that your classical baseline
is the best classical method, or that your error mitigation is legitimate: mitigated figures are
labelled *reported*, next to the measured raw figures.

## Public API

Everything a workflow imports is exported from `qsolve`: `Workflow`, `Outcome`, `Context`,
`WorkflowError`, `InvalidSolution`, `MethodUnavailable`, `QuantumNotAllowed`, `Estimate`, `Result`,
`Provenance`, `Quality`, `Mitigation`, `Uncertainty`, `MethodResult`, `Metrics`, `CircuitStats`,
`RunOutput`, `Manifest`, `load_workflow`, `run_workflow`, `estimate_run`, `MAX_SHOTS_PER_CIRCUIT`
(the most shots the platform runs per circuit, 100000) and `__version__`.

**Compatibility.** The SDK follows semantic versioning. Names in `qsolve.__all__`, the `qsolve`
commands and the lint codes are public and stable: they change incompatibly only in a new major
version, after a deprecating minor release, and every change is listed in `CHANGELOG.md`. Other
modules (`qsolve.backends`, `qsolve.metering`, `qsolve.benchmark`, `qsolve.lint`, ...) are
documented and importable but may change in a minor version. The executor protocol
(`qsolve.backends.remote`), `qsolve.sandbox_entry` and every underscore-prefixed name are internal
and may change in any release. Every public name,
with signatures, parameters and examples, is in the SDK reference of the
[Qsolve documentation](https://qsolve.omlabs.co/docs/build/overview) (Build a workflow, "SDK
reference"), and every `qsolve` command and flag is in "The qsolve command line" there.

## Documentation

- [The Qsolve documentation](https://qsolve.omlabs.co/docs/build/overview), section "Build a
  workflow": the guided tutorial, the SDK reference and the CLI reference
- [Writing a workflow](docs/writing-a-workflow.md)
- [The contract](docs/workflow-contract.md)
- [Design: the SDK as the bridge](docs/design/sdk-as-bridge.md)
- [Changelog](CHANGELOG.md) and [contributing](CONTRIBUTING.md)

## Development

```bash
uv sync && uv run pytest && uv run ruff check src tests && uv run mypy src
```
