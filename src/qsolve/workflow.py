"""The workflow contract: what a workflow author writes, and nothing algorithmic beyond that.

A workflow is a Qiskit program. It subclasses `Workflow`, builds circuits with Qiskit, and runs them
through the `Context` it is given (a Qiskit `BackendV2` and Sampler/Estimator primitives that run on
CPU, GPU or QPU through Qsolve). The SDK records every circuit sent and every count returned, runs
every method the manifest declares, re-evaluates what it can, and computes the numbers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .results import Estimate

if TYPE_CHECKING:
    from qiskit.primitives import BaseEstimatorV2, BaseSamplerV2
    from qiskit.providers import BackendV2


class WorkflowError(Exception):
    """A problem with the user's input or the workflow itself, worded for the person who submitted
    it. Raised from `__init__` for bad input; the platform shows the message as it is."""


class InvalidSolution(Exception):
    """Raised by `Workflow.evaluate` for a solution that breaks a rule of the problem."""


class MethodUnavailable(Exception):
    """Raised by a classical method that does not apply to this input; say why in the message."""


class QuantumNotAllowed(WorkflowError):
    """A classical method (role `reference` or `classical`) tried to use the quantum backend."""


@dataclass
class Outcome:
    """What a method returns.

    `solution` is JSON data in whatever form `Workflow.evaluate` accepts; `None` for a result that
    cannot be re-evaluated (a measured energy), which is then labelled `reported`. `objective` is
    the value the method reports for it (lower is better); the SDK recomputes it with `evaluate`
    and flags a difference. `stderr` is one standard deviation when the method measures it.
    `samples` lists the outcomes of a sampling method as `{"solution": ..., "p": probability}`:
    give all of them, the SDK evaluates each to report the share that is valid and the share on
    the reference answer, and shows the top few. `details` is free-form and kept with the receipt.
    """

    solution: Any
    objective: float
    stderr: float | None = None
    samples: list[dict[str, Any]] | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    """What a method runs with. Quantum methods get a live backend; classical methods get one that
    refuses to run circuits, so "classical" is enforced, not promised."""

    seed: int
    target: str
    backend: BackendV2
    sampler: BaseSamplerV2
    estimator: BaseEstimatorV2
    emit: Callable[..., None]
    report_mitigation: Callable[..., None]


class Workflow:
    """Base class of every workflow. Override what the contract asks for (see the design note in
    docs/design). `__init__` receives the validated input and parameters: parse them here and raise
    `WorkflowError` for anything the user got wrong."""

    def __init__(self, input: dict[str, Any], params: dict[str, Any]) -> None:
        """Store the validated input and parameters as `self.input` and `self.params`.

        Subclasses call `super().__init__(input, params)`, parse the input and raise
        `WorkflowError` for anything the user got wrong.

        Args:
            input: The user's input, already validated against the input schema.
            params: Every manifest parameter, defaults applied and checked.
        """
        self.input = input
        self.params = params

    def size(self) -> int:
        """Number of qubits or variables, checked against the manifest's envelope."""
        raise NotImplementedError

    def quantum(self, ctx: Context) -> Outcome:
        """The quantum method (manifest role `quantum`)."""
        raise NotImplementedError

    def reference(self, ctx: Context) -> Outcome:
        """The classical reference the quantum result is judged against (role `reference`)."""
        raise NotImplementedError

    def evaluate(self, solution: Any) -> float:
        """The objective of `solution` (lower is better). Raise `InvalidSolution` if it breaks a
        rule. The SDK calls this to recompute every method's number."""
        raise NotImplementedError

    def answer(self, outcome: Outcome) -> dict[str, Any]:
        """The domain answer for a method's outcome, as the output schema describes it. Called for
        the quantum result (required) and for every other method (shown beside it)."""
        raise NotImplementedError

    def estimate(self, target: str) -> Estimate | None:
        """Optional: circuits and shots to expect on `target`, for the price quote. The SDK
        compares it with what was metered and fails validation if it is far off."""
        return None

    def summary(self, quantum: Outcome, reference: Outcome) -> str | None:
        """Optional: one sentence for the result page."""
        return None
