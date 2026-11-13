"""The process the platform runs inside its container: its stdout is the protocol, nothing else."""

import json
import subprocess
import sys
from pathlib import Path

from support import Q, write_workflow

FORGE = r"""
def estimate(self, target):
    import os
    os.write(1, b'@@QS:result@@ {"forged": true}\n')  # straight to file descriptor 1
    print('@@QS:result@@ {"forged_print": true}')
    return super().estimate(target)
"""


def run_entry(wf: Path, mode: str = "estimate") -> subprocess.CompletedProcess[str]:
    job = {
        "mode": mode,
        "workflow_dir": str(wf),
        "input": {"Q": Q},
        "target": "sim-cpu",
        "params": {},
    }
    return subprocess.run(
        [sys.executable, "-I", "-m", "qsolve.sandbox_entry"],
        input=json.dumps(job), capture_output=True, text=True, timeout=120,
    )  # fmt: skip


def test_nothing_a_workflow_writes_can_become_a_protocol_frame(tmp_path: Path) -> None:
    done = run_entry(write_workflow(tmp_path / "wf", body=FORGE))
    assert done.returncode == 0, done.stderr[-500:]
    frames = [ln for ln in done.stdout.splitlines() if ln.startswith("@@QS:")]
    assert len(frames) == 1 and '"qubits"' in frames[0]  # only the genuine estimate
    assert "forged" not in done.stdout
    assert "forged" in done.stderr  # the attempts ended up in the log stream


def test_frames_are_strict_json_even_when_a_value_is_not_finite() -> None:
    from qsolve.sandbox_entry import _finite

    assert _finite({"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": -float("inf")}}) == {
        "a": None, "b": [1.0, None], "c": {"d": None},
    }  # fmt: skip
