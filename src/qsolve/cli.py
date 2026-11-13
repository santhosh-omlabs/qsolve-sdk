"""qsolve CLI: validate, test and run workflows locally."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .benchmark import run_benchmarks
from .lint import has_errors, lint_workflow
from .manifest import load_workflow
from .runner import WorkflowError, estimate_run, run_workflow, validate_input

app = typer.Typer(help="Qsolve workflow SDK", no_args_is_help=True, rich_markup_mode=None)
console = Console()


def _version(show: bool) -> None:
    if show:
        from . import __version__

        console.print(__version__)
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show the version")
    ] = False,
) -> None:
    pass


# every error line is printed through `_fail`, which escapes square brackets (rich would read
# `[ibm]` or `[@scale]` as markup and drop them)
_TARGET_HELP = "sim-cpu | sim-gpu | fake:<device>[@scale] | ibm:<device>"


# what a bad workflow directory, manifest, input file or target raises: shown, never a traceback
_USER_ERRORS = (WorkflowError, FileNotFoundError, ValidationError, ValueError)


def _fail(label: str, detail: object) -> None:
    console.print(f"[red]{label}[/red] {escape(str(detail))}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:  # missing file, bad encoding, invalid JSON
        raise WorkflowError(f"cannot read {path}: {e}") from e
    if not isinstance(data, dict):
        raise WorkflowError(f"{path} must contain a JSON object")
    return data


@app.command()
def init(
    name: Annotated[str, typer.Argument(help="Workflow name: lowercase letters, digits, dashes")],
    summary: Annotated[str, typer.Option(help="One-line description")] = "My Qsolve workflow",
    directory: Annotated[Path | None, typer.Option(help="Parent directory")] = None,
) -> None:
    """Scaffold a working Qiskit workflow (a QUBO solved with QAOA) to start from."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,62}", name):
        _fail("Name must be lowercase letters, digits and dashes (2-63 chars).", "")
        raise typer.Exit(1)
    dest = (directory or Path.cwd()) / name
    if dest.exists():
        _fail(f"{dest} already exists.", "")
        raise typer.Exit(1)
    template = Path(__file__).parent / "templates" / "qiskit"
    shutil.copytree(template, dest, ignore=shutil.ignore_patterns("__pycache__"))
    (dest / "src" / "__init__.py").touch()
    (dest / "tests").mkdir()
    (dest / "tests" / "test_workflow.py").write_text(_TEST_TEMPLATE, encoding="utf-8")
    manifest = dest / "qsolve.yaml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8")
        .replace("__NAME__", json.dumps(name))
        .replace("__SUMMARY__", json.dumps(" ".join(summary.split()), ensure_ascii=False)),
        encoding="utf-8",
    )
    readme = f"# {name}\n\n{summary}\n\n{_README_TAIL}"
    (dest / "README.md").write_text(readme, encoding="utf-8")
    console.print(f"[green]Created[/green] {escape(str(dest))}")
    console.print(
        f"Next: cd {escape(name)} && qsolve test .   (edit src/workflow.py: it is plain Qiskit)"
    )


_TEST_TEMPLATE = """import json
from pathlib import Path

from qsolve import run_workflow

ROOT = Path(__file__).resolve().parents[1]


def test_example_matches_the_best_classical_answer() -> None:
    inp = json.loads((ROOT / "examples" / "qubo_5.json").read_text())
    out = run_workflow(str(ROOT), inp, params={"layers": 2, "optimizer_iters": 30}, seed=1)
    assert out.result.metrics is not None and out.result.metrics.matches
"""

_README_TAIL = """## Develop

```
qsolve validate .   # check the manifest, schemas and example inputs
qsolve run . -i examples/qubo_5.json
qsolve test .       # the same gates the Qsolve platform runs
```

`src/workflow.py` is plain Qiskit: the problem, the quantum method, the classical methods and how a
solution is scored. `qsolve.yaml` declares the methods and says in plain words what each one is.
Qsolve runs every declared method, re-evaluates each answer, and measures the quantum run from the
circuits it sends. Edit `benchmarks/cases.yaml` to define the cases the platform will benchmark.
"""


@app.command()
def validate(
    workflow: Annotated[Path, typer.Argument(help="Workflow directory")] = Path("."),
) -> None:
    """Check the manifest, schemas and example inputs."""
    try:
        wf = load_workflow(workflow)
    except (FileNotFoundError, ValidationError, ValueError) as e:
        _fail("Invalid workflow:", e)
        raise typer.Exit(1) from e
    for ex in wf.manifest.inputs.examples:
        try:
            validate_input(wf, _load_json(wf.root / ex))
        except WorkflowError as e:
            _fail(f"Example {escape(ex)}:", e)
            raise typer.Exit(1) from e
    console.print(f"[green]OK[/green] {escape(wf.manifest.name)} v{wf.manifest.version}")


def _json_or_text(value: str) -> Any:
    """`--param shots=500` is a number, `--param mode=fast` stays text."""
    try:
        return json.loads(value)
    except ValueError:
        return value


@app.command()
def run(
    workflow: Annotated[Path, typer.Argument(help="Workflow directory")],
    input_file: Annotated[Path, typer.Option("--input", "-i", help="Input JSON")],
    target: Annotated[str, typer.Option(help=_TARGET_HELP)] = "sim-cpu",
    seed: Annotated[int, typer.Option()] = 0,
    param: Annotated[list[str] | None, typer.Option("--param", "-p", help="key=value")] = None,
    out: Annotated[Path | None, typer.Option(help="Write full JSON output here")] = None,
) -> None:
    """Run a workflow locally."""
    overrides: dict[str, Any] = {}
    for kv in param or []:
        k, sep, v = kv.partition("=")
        if not sep or not k:
            raise typer.BadParameter(f"--param expects key=value, got {kv!r}")
        overrides[k] = _json_or_text(v)
    try:
        output = run_workflow(
            str(workflow), _load_json(input_file), target=target, params=overrides, seed=seed
        )
    except _USER_ERRORS as e:
        _fail("Error:", e)
        raise typer.Exit(1) from e
    r, p = output.result, output.provenance
    t = Table(title=escape(f"{p.workflow} v{p.version} on {p.device}"))
    t.add_column("Metric")
    t.add_column("Value")
    t.add_row("Objective (quantum)", f"{r.objective:.6g}")
    t.add_row(f"Objective ({r.baseline_solver})", f"{r.baseline_objective:.6g}")
    t.add_row("Gap vs baseline", f"{r.gap:.3%}")
    t.add_row("Uncertainty", f"{r.uncertainty.kind} (stderr {r.uncertainty.stderr})")
    t.add_row("Shots", str(p.shots))
    t.add_row("Run time (s)", f"{p.exec_time_s:.2f}")
    console.print(t)
    console.print(escape(r.summary))
    if out:
        out.write_text(output.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"Wrote {escape(str(out))}")


@app.command()
def lint(
    workflow: Annotated[Path, typer.Argument(help="Workflow directory")] = Path("."),
) -> None:
    """Static checks the platform applies before running anything (no code is executed)."""
    findings = lint_workflow(workflow)
    for f in findings:
        colour = "red" if f.severity == "error" else "yellow"
        where = f" [dim]{escape(f.path)}[/dim]" if f.path else ""
        console.print(
            f"[{colour}]{f.severity.upper():7}[/{colour}] {f.code} {escape(f.message)}{where}"
        )
    errors = sum(f.severity == "error" for f in findings)
    console.print(
        "[green]No problems found[/green]" if not findings
        else f"{errors} error(s), {len(findings) - errors} warning(s)"
    )  # fmt: skip
    raise typer.Exit(1 if has_errors(findings) else 0)


@app.command()
def estimate(
    workflow: Annotated[Path, typer.Argument(help="Workflow directory")],
    input_file: Annotated[Path, typer.Option("--input", "-i", help="Input JSON")],
    target: Annotated[str, typer.Option(help=_TARGET_HELP)] = "sim-cpu",
) -> None:
    """Size a run (qubits, circuits, shots) without executing it."""
    try:
        est = estimate_run(str(workflow), _load_json(input_file), target=target)
    except _USER_ERRORS as e:
        _fail("Error:", e)
        raise typer.Exit(1) from e
    console.print_json(est.model_dump_json())


@app.command()
def test(
    workflow: Annotated[Path, typer.Argument(help="Workflow directory")] = Path("."),
    target: Annotated[str, typer.Option(help=_TARGET_HELP)] = "sim-cpu",
    out: Annotated[Path | None, typer.Option(help="Write the JSON report here")] = None,
) -> None:
    """Run the validation gates and benchmark cases (the same ones the platform runs)."""
    try:
        report = run_benchmarks(workflow, target)
    except (FileNotFoundError, ValidationError, ValueError) as e:
        _fail("Invalid workflow:", e)
        raise typer.Exit(1) from e
    g = Table(title="Gates")
    g.add_column("Gate")
    g.add_column("Result")
    g.add_column("Detail")
    for gate in report.gates:
        g.add_row(
            escape(gate.name),
            "[green]pass[/green]" if gate.passed else "[red]FAIL[/red]",
            escape(gate.detail),
        )
    console.print(g)
    t = Table(title=f"Cases on {report.target}")
    for col in ("Case", "Result", "Gap", "Fidelity", "Time (s)"):
        t.add_column(col)
    for c in report.cases:
        status = "[green]pass[/green]" if c.passed else f"[red]FAIL[/red] {escape(c.error or '')}"
        t.add_row(
            escape(c.name),
            status,
            "-" if c.gap is None else f"{c.gap:.2%}",
            "-" if c.fidelity is None else f"{c.fidelity:.3f}",
            "-" if c.exec_time_s is None else f"{c.exec_time_s:.1f}",
        )
    console.print(t)
    if out:
        out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    console.print("[green]ALL GATES PASSED[/green]" if report.passed else "[red]FAILED[/red]")
    raise typer.Exit(0 if report.passed else 1)


if __name__ == "__main__":
    app()
