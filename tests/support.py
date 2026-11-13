"""A tiny but complete workflow written to a temporary folder, for testing the SDK itself.

`write_workflow(tmp, body="...")` appends `body` inside the class, so a test can override one method
(a lying reference, a classical method that touches the backend, a missing evaluate...).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import yaml

SOURCE = """
import numpy as np
from qiskit import QuantumCircuit
from qsolve import Estimate, InvalidSolution, Outcome, Workflow, WorkflowError


class Tiny(Workflow):
    def __init__(self, input, params):
        super().__init__(input, params)
        self.Q = np.array(input["Q"], dtype=float)
        if self.Q.ndim != 2 or self.Q.shape[0] != self.Q.shape[1]:
            raise WorkflowError("Q must be square")
        self.n = len(self.Q)

    def size(self):
        return self.n

    def evaluate(self, solution):
        x = np.asarray(solution, dtype=float)
        if x.shape != (self.n,) or not np.isin(x, (0, 1)).all():
            raise InvalidSolution("one 0 or 1 per variable")
        return float(x @ self.Q @ x)

    def answer(self, outcome):
        bits = [int(b) for b in outcome.solution]
        return {"bits": bits, "objective": self.evaluate(outcome.solution)}

    def estimate(self, target):
        return Estimate(
            qubits=self.n, circuits_on_target=1, shots_on_target=int(self.params["shots"])
        )

    def _states(self):
        return (np.arange(2**self.n)[:, None] >> np.arange(self.n)) & 1

    def reference(self, ctx):
        s = self._states()
        e = np.einsum("si,ij,sj->s", s, self.Q, s)
        return Outcome(solution=s[int(np.argmin(e))].tolist(), objective=float(e.min()))

    def classical_greedy(self, ctx):
        return self.reference(ctx)

    def quantum(self, ctx):
        shots = int(self.params["shots"])
        qc = QuantumCircuit(self.n)
        qc.h(range(self.n))
        qc.measure_all()
        counts = ctx.sampler.run([(qc, None, shots)]).result()[0].data.meas.get_counts()
        samples = [
            {"solution": [int(c) for c in reversed(b)], "p": n / shots}
            for b, n in counts.items()
        ]
        best = min(samples, key=lambda s: self.evaluate(s["solution"]))
        return Outcome(solution=best["solution"], objective=self.evaluate(best["solution"]),
                       samples=samples)
"""

MANIFEST = {
    "schema_version": 1,
    "name": "tiny-flow",
    "version": "0.1.0",
    "family": "demo",
    "summary": "A test workflow",
    "license": "MIT",
    "inputs": {"schema": "schemas/input.json", "examples": ["examples/q.json"]},
    "parameters": [{"name": "shots", "type": "int", "default": 400, "min": 10, "max": 5000}],
    "outputs": {"schema": "schemas/output.json"},
    "entrypoint": "src.workflow:Tiny",
    "metric": {"kind": "relative_gap"},
    "methods": [
        {"id": "random", "role": "quantum", "title": "Random sampling circuit"},
        {"id": "exact", "role": "reference", "title": "Exhaustive search"},
        {"id": "greedy", "role": "classical", "title": "Greedy"},
    ],
    "envelope": {"max_qubits": 6, "max_variables": 6},
}
Q = [[-2, 1, 0], [1, -1, 2], [0, 2, -3]]


def write_workflow(root: Path, *, body: str = "", manifest: dict | None = None) -> Path:
    (root / "src").mkdir(parents=True)
    (root / "schemas").mkdir()
    (root / "examples").mkdir()
    (root / "src" / "__init__.py").write_text("")
    (root / "src" / "workflow.py").write_text(
        SOURCE + textwrap.indent(textwrap.dedent(body), "    ")
    )
    (root / "schemas" / "input.json").write_text(
        json.dumps(
            {
                "type": "object",
                "required": ["Q"],
                "properties": {"Q": {"type": "array", "items": {"type": "array"}}},
            }
        )
    )
    (root / "schemas" / "output.json").write_text(
        json.dumps(
            {
                "type": "object",
                "required": ["bits", "objective"],
                "properties": {"bits": {"type": "array"}, "objective": {"type": "number"}},
            }
        )
    )
    (root / "examples" / "q.json").write_text(json.dumps({"Q": Q}))
    (root / "qsolve.yaml").write_text(yaml.safe_dump(manifest or MANIFEST, sort_keys=False))
    return root
