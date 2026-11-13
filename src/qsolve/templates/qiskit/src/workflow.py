"""A QUBO solved with QAOA, in plain Qiskit.

This file is the whole workflow: the problem, the quantum method, the classical methods and how a
solution is scored. Qsolve runs every method declared in qsolve.yaml, re-evaluates each answer with
`evaluate`, and measures the quantum run from the circuits it sends through `ctx`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from qsolve import Context, Estimate, InvalidSolution, Outcome, Workflow, WorkflowError


class Qubo(Workflow):
    """Minimise x'Qx over 0/1 variables."""

    def __init__(self, input: dict[str, Any], params: dict[str, Any]) -> None:
        """Parse and check the input: a square matrix `Q` and optional variable `labels`."""
        super().__init__(input, params)
        q = np.array(input["Q"], dtype=float)
        if q.ndim != 2 or q.shape[0] != q.shape[1]:
            raise WorkflowError("Q must be a square matrix")
        self.n = len(q)
        self.Q = (q + q.T) / 2  # x'Qx only sees the symmetric part
        labels = input.get("labels") or [f"x{i}" for i in range(self.n)]
        if len(labels) != self.n:
            raise WorkflowError(f"{len(labels)} variable names for {self.n} variables")
        self.labels = labels
        # every state as a row of bits: bit i of the index is variable i (Qiskit's qubit order)
        self._states = (np.arange(2**self.n)[:, None] >> np.arange(self.n)) & 1
        self._energies = np.einsum("si,ij,sj->s", self._states, self.Q, self._states)

    # -- the contract -----------------------------------------------------------------------------

    def size(self) -> int:
        """The number of variables (one qubit each)."""
        return self.n

    def evaluate(self, solution: list[int]) -> float:
        """The objective `x'Qx` of a 0/1 assignment; raises `InvalidSolution` for anything else."""
        x = np.asarray(solution, dtype=float)
        if x.shape != (self.n,) or not np.isin(x, (0, 1)).all():
            raise InvalidSolution("a solution is one 0 or 1 per variable")
        return float(x @ self.Q @ x)

    def answer(self, outcome: Outcome) -> dict[str, Any]:
        """The answer people read: the assignment, the variables set to 1 and the objective."""
        x = [int(b) for b in outcome.solution]
        return {
            "assignment": dict(zip(self.labels, x, strict=True)),
            "ones": [name for name, b in zip(self.labels, x, strict=True) if b],
            "objective": self.evaluate(x),
        }

    def estimate(self, target: str) -> Estimate:
        """One circuit with `shots` shots on the target."""
        return Estimate(
            qubits=self.n, circuits_on_target=1, shots_on_target=int(self.params["shots"])
        )

    # -- the quantum method: QAOA, trained noise-free, sampled on the target ----------------------

    def _ising(self) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
        """E(x) with x_i = (1 - z_i) / 2 is  const + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j."""
        q = self.Q
        h = -np.diag(q) / 2 - (q.sum(axis=1) - np.diag(q)) / 2
        j = {(a, b): float(q[a, b]) for a in range(self.n) for b in range(a + 1, self.n) if q[a, b]}
        return h, j

    def _circuit(self, p: int) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
        h, j = self._ising()
        scale = max(
            np.abs(h).max(initial=0.0), max((abs(v) for v in j.values()), default=0.0), 1e-9
        )
        gamma, beta = ParameterVector("g", p), ParameterVector("b", p)
        qc = QuantumCircuit(self.n)
        qc.h(range(self.n))
        for layer in range(p):
            for i in range(self.n):
                qc.rz(2 * gamma[layer] * float(h[i]) / scale, i)
            for (a, b), w in j.items():
                qc.rzz(2 * gamma[layer] * w / scale, a, b)
            qc.rx(2 * beta[layer], range(self.n))
        return qc, gamma, beta

    def quantum(self, ctx: Context) -> Outcome:
        """Train QAOA angles on a noise-free statevector, then sample via `ctx.sampler`."""
        p, shots = int(self.params["layers"]), int(self.params["shots"])
        qc, _, _ = self._circuit(p)
        order = np.argsort(self._energies)

        def cost(theta: np.ndarray) -> float:  # mean energy of the best quarter (CVaR), exact
            probs = Statevector(qc.assign_parameters(theta)).probabilities()
            cum = np.cumsum(probs[order])
            keep = np.clip(0.25 - (cum - probs[order]), 0.0, probs[order])
            return float((self._energies[order] * keep).sum() / keep.sum())

        ramp = (np.arange(1, p + 1) - 0.5) / p
        x0 = np.concatenate([0.7 * ramp, 0.7 * (1 - ramp)])
        theta = minimize(
            cost, x0, method="COBYLA", options={"maxiter": int(self.params["optimizer_iters"])}
        ).x

        measured = qc.copy()
        measured.measure_all()
        counts = ctx.sampler.run([(measured, theta, shots)]).result()[0].data.meas.get_counts()
        samples = [
            {"solution": [int(c) for c in reversed(bits)], "p": n / shots}
            for bits, n in counts.items()
        ]
        best = min(samples, key=lambda s: self.evaluate(s["solution"]))
        return Outcome(
            solution=best["solution"],
            objective=self.evaluate(best["solution"]),
            samples=sorted(samples, key=lambda s: -s["p"])[:10],
            details={"theta": theta.tolist(), "layers": p},
        )

    # -- classical methods ------------------------------------------------------------------------

    def reference(self, ctx: Context) -> Outcome:
        """Exhaustive search over every assignment: what the quantum result is judged against."""
        best = int(np.argmin(self._energies))
        return Outcome(solution=self._states[best].tolist(), objective=float(self._energies[best]))

    def classical_annealing(self, ctx: Context) -> Outcome:
        """Simulated annealing with restarts (the `annealing` method in `qsolve.yaml`)."""
        rng = np.random.default_rng(ctx.seed)
        best_x, best_e = None, np.inf
        for _ in range(5):
            x = rng.integers(0, 2, self.n)
            e = float(x @ self.Q @ x)
            for sweep in range(300):
                temp = max(1e-3, 2.0 * (1 - sweep / 300)) * max(np.abs(self.Q).max(), 1e-9)
                for _ in range(self.n):
                    i = int(rng.integers(self.n))
                    y = x.copy()
                    y[i] = 1 - y[i]
                    ey = float(y @ self.Q @ y)
                    if ey <= e or rng.random() < np.exp(-(ey - e) / temp):
                        x, e = y, ey
                        if e < best_e:
                            best_x, best_e = x.copy(), e
        assert best_x is not None
        return Outcome(solution=best_x.tolist(), objective=best_e)
