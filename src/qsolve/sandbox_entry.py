"""Entrypoint the platform runs inside the sandbox container:

    python -m qsolve.sandbox_entry < job.json

Reads one JSON job from stdin: {"mode": "estimate"|"run"|"validate", "workflow_dir",
"input", "params", "target", "seed"}. Writes framed lines to the real stdout, one per line:

    @@QS:event@@ {...}     progress events (capped by the platform)
    @@QS:result@@ {...}    the final result (Estimate, or RunOutput)
    @@QS:error@@ {...}     {"kind": "workflow"|"crash", "message": str}

Everything the workflow itself prints, from Python or written straight to file descriptor 1 (a
subprocess, a native library), goes to stderr, so it cannot be mistaken for protocol frames. The
protocol stream is a private duplicate of the original stdout. This stops accidents and casual
forging; code running in this same process can still find that descriptor, which is why the platform
also checks the result against what its host-side executor metered. Exit codes: 0 ok, 2 workflow
error (user input), 1 crash.
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback
from typing import Any, TextIO

FRAME = "@@QS:{kind}@@ {payload}\n"


def _finite(x: Any) -> Any:
    """JSON has no NaN or infinity: they become null, so strict parsers accept every frame."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_finite(v) for v in x]
    return x


def _frame(out: TextIO, kind: str, payload: Any) -> None:
    text = json.dumps(_finite(payload), default=str, allow_nan=False)
    out.write(FRAME.format(kind=kind, payload=text))
    out.flush()


def main() -> int:
    """Read one job from stdin, run it and write framed result lines; see the module docstring.

    Returns:
        The process exit code: 0 ok, 2 workflow error, 1 crash.
    """
    protocol = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)  # fd 1 now leads to the log stream, whoever writes to it
    sys.stdout = sys.stderr  # workflow prints go to the log stream
    try:
        job = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        _frame(protocol, "error", {"kind": "crash", "message": f"bad job spec: {e}"})
        return 1

    from qsolve import WorkflowError, estimate_run, run_workflow

    try:
        if job["mode"] == "validate":
            from qsolve.benchmark import run_benchmarks

            report = run_benchmarks(job["workflow_dir"], "sim-cpu")
            _frame(protocol, "result", report.model_dump(mode="json"))
            return 0
        if job["mode"] == "estimate":
            est = estimate_run(
                job["workflow_dir"], job["input"], target=job["target"], params=job.get("params")
            )
            _frame(protocol, "result", est.model_dump(mode="json"))
        else:
            out = run_workflow(
                job["workflow_dir"],
                job["input"],
                target=job["target"],
                params=job.get("params"),
                seed=int(job.get("seed", 0)),
                commit_sha=job.get("commit_sha"),
                image_digest=job.get("image_digest"),
                on_event=lambda e: _frame(protocol, "event", e),
            )
            _frame(protocol, "result", out.model_dump(mode="json"))
        return 0
    except WorkflowError as e:
        _frame(protocol, "error", {"kind": "workflow", "message": str(e)})
        return 2
    except BaseException as e:
        traceback.print_exc()
        _frame(protocol, "error", {"kind": "crash", "message": f"{type(e).__name__}: {e}"[:500]})
        return 1


if __name__ == "__main__":
    sys.exit(main())
