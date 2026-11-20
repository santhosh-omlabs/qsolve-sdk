"""The command line a new developer touches first: scaffold, validate, run, lint, estimate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from qsolve.cli import _json_or_text, app

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scaffold(tmp_path_factory: pytest.TempPathFactory) -> Path:
    parent = tmp_path_factory.mktemp("scaffold")
    result = runner.invoke(app, ["init", "my-problem", "--directory", str(parent)])
    assert result.exit_code == 0, result.output
    return parent / "my-problem"


def test_init_writes_a_complete_workflow(scaffold: Path) -> None:
    for name in ("qsolve.yaml", "README.md", "src/workflow.py", "benchmarks/cases.yaml"):
        assert (scaffold / name).is_file(), name
    assert "my-problem" in (scaffold / "qsolve.yaml").read_text(encoding="utf-8")
    assert "__NAME__" not in (scaffold / "qsolve.yaml").read_text(encoding="utf-8")
    # runtime.python and runtime.dependencies are informational defaults: not written
    assert "runtime:" not in (scaffold / "qsolve.yaml").read_text(encoding="utf-8")


def test_scaffold_lints_clean_and_ships_the_sdk_license(scaffold: Path) -> None:
    # a new author's first `qsolve lint .` must not warn about a missing LICENSE
    # read_text normalises line endings: a Windows checkout may rewrite one copy as CRLF
    assert (scaffold / "LICENSE").read_text(encoding="utf-8") == (ROOT / "LICENSE").read_text(
        encoding="utf-8"
    )
    result = runner.invoke(app, ["lint", str(scaffold)])
    assert result.exit_code == 0 and "No problems found" in result.output, result.output


@pytest.mark.parametrize("bad", ["A", "Upper-Case", "has space", "-leading-dash", "x"])
def test_init_rejects_names_that_are_not_lowercase_dashes(tmp_path: Path, bad: str) -> None:
    result = runner.invoke(app, ["init", bad, "--directory", str(tmp_path)])
    assert result.exit_code != 0
    assert not any(tmp_path.iterdir())


def test_init_never_overwrites_an_existing_folder(tmp_path: Path) -> None:
    (tmp_path / "taken").mkdir()
    result = runner.invoke(app, ["init", "taken", "--directory", str(tmp_path)])
    # Rich wraps long paths at the terminal width, so compare with whitespace collapsed
    assert result.exit_code == 1 and "already exists" in " ".join(result.output.split())


def test_check_runs_the_stages_in_order_and_stops_at_the_first_failure(
    scaffold: Path, tmp_path: Path
) -> None:
    broken = tmp_path / "broken"
    broken.mkdir()
    result = runner.invoke(app, ["check", str(broken)])
    out = " ".join(result.output.split())
    assert result.exit_code == 1 and "== qsolve validate" in out
    assert "== qsolve lint" not in out and "qsolve validate failed" in out
    ok = runner.invoke(app, ["check", str(scaffold)])
    text = " ".join(ok.output.split())
    assert ok.exit_code == 0, ok.output
    assert (
        text.index("== qsolve validate")
        < text.index("== qsolve lint")
        < text.index("== qsolve test")
    )
    assert "ready to submit" in text


def test_the_scaffold_validates_and_lints_clean(scaffold: Path) -> None:
    assert runner.invoke(app, ["validate", str(scaffold)]).exit_code == 0
    assert runner.invoke(app, ["lint", str(scaffold)]).exit_code == 0


def test_run_prints_an_answer_and_writes_the_full_output(scaffold: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    result = runner.invoke(
        app,
        [
            "run",
            str(scaffold),
            "-i",
            str(scaffold / "examples/qubo_5.json"),
            "--out",
            str(out),
            "--param",
            "layers=1",
            "--param",
            "optimizer_iters=10",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "result" in json.loads(out.read_text(encoding="utf-8"))


def test_estimate_reports_the_cost_without_running(scaffold: Path) -> None:
    result = runner.invoke(
        app, ["estimate", str(scaffold), "-i", str(scaffold / "examples/qubo_5.json")]
    )
    assert result.exit_code == 0, result.output


def test_a_missing_input_file_is_a_clean_error_not_a_traceback(scaffold: Path) -> None:
    result = runner.invoke(app, ["run", str(scaffold), "-i", str(scaffold / "nope.json")])
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_input_must_be_a_json_object(scaffold: Path, tmp_path: Path) -> None:
    bad = tmp_path / "list.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    result = runner.invoke(app, ["run", str(scaffold), "-i", str(bad)])
    assert result.exit_code != 0


@pytest.mark.parametrize(
    ("raw", "parsed"),
    [("500", 500), ("0.5", 0.5), ("true", True), ("[1, 2]", [1, 2]), ("fast", "fast"),
     ("tabu", "tabu"), ("", ""), ("-3", -3)],
)  # fmt: skip
def test_param_values_are_json_when_they_can_be_and_text_otherwise(
    raw: str, parsed: object
) -> None:
    assert _json_or_text(raw) == parsed


def test_a_new_developers_mistakes_print_a_message_not_a_traceback(
    scaffold: Path, tmp_path: Path
) -> None:
    ex = scaffold / "examples" / "small.json"
    good = next(scaffold.glob("examples/*.json"), ex)
    cases = [
        ["run", str(scaffold), "-i", str(tmp_path / "missing.json")],  # no such file
        ["run", str(scaffold), "-i", str(good), "--target", "nonsense"],  # unknown target
        ["run", str(scaffold), "-i", str(good), "--target", "fake:"],  # no device
    ]
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json", encoding="utf-8")
    cases.append(["run", str(scaffold), "-i", str(bad_json)])
    for args in cases:
        result = runner.invoke(app, args)
        assert result.exit_code == 1, (args, result.output)
        assert "Error" in result.output and "Traceback" not in result.output, (args, result.output)


def test_broken_yaml_and_a_missing_schema_are_reported_by_lint_and_validate(
    scaffold: Path, tmp_path: Path
) -> None:
    import shutil

    broken = tmp_path / "broken"
    shutil.copytree(scaffold, broken)
    (broken / "qsolve.yaml").write_text("name: [unclosed\n  - x: : :", encoding="utf-8")
    for cmd in ("lint", "validate"):
        r = runner.invoke(app, [cmd, str(broken)])
        assert r.exit_code == 1 and "Traceback" not in r.output, (cmd, r.output)

    missing = tmp_path / "missing"
    shutil.copytree(scaffold, missing)
    next((missing / "schemas").glob("output*.json")).unlink()
    r = runner.invoke(app, ["lint", str(missing)])
    assert r.exit_code == 1 and "E002" in r.output and "E001" not in r.output, r.output


def test_param_without_equals_is_an_error_and_version_prints(scaffold: Path) -> None:
    r = runner.invoke(app, ["run", str(scaffold), "-i", "x.json", "-p", "foo"])
    assert r.exit_code == 2
    v = runner.invoke(app, ["--version"])
    assert v.exit_code == 0 and v.output.strip()


def test_hardware_cases_are_linted_but_never_run_by_test(scaffold: Path, tmp_path: Path) -> None:
    import shutil

    import yaml

    from qsolve.benchmark import load_hardware_cases

    wf = tmp_path / "hw"
    shutil.copytree(scaffold, wf)
    cases_path = wf / "benchmarks" / "cases.yaml"
    cases = yaml.safe_load(cases_path.read_text(encoding="utf-8"))
    base = cases["cases"][0]
    cases["hardware_cases"] = [{**base, "name": "on-chip", "target": "ibm:ibm_fez"}]
    cases_path.write_text(yaml.safe_dump(cases), encoding="utf-8")
    m = yaml.safe_load((wf / "qsolve.yaml").read_text(encoding="utf-8"))
    m.setdefault("targets", {})["qpu"] = ["ibm"]
    (wf / "qsolve.yaml").write_text(yaml.safe_dump(m), encoding="utf-8")
    assert [c.name for c in load_hardware_cases(wf)] == ["on-chip"]
    assert "E05" not in runner.invoke(app, ["lint", str(wf)]).output

    def lint_codes(change: object) -> str:
        c = yaml.safe_load(cases_path.read_text(encoding="utf-8"))
        change(c)  # type: ignore[operator]
        cases_path.write_text(yaml.safe_dump(c), encoding="utf-8")
        return runner.invoke(app, ["lint", str(wf)]).output

    assert "E052" in lint_codes(lambda c: c["hardware_cases"][0].update(target="sim-cpu"))
    assert "E053" in lint_codes(
        lambda c: c["hardware_cases"].extend(
            [{**c["hardware_cases"][0], "name": f"x{i}", "target": "ibm:ibm_fez"} for i in range(3)]
        )
    )
    m["targets"]["qpu"] = []
    (wf / "qsolve.yaml").write_text(yaml.safe_dump(m), encoding="utf-8")
    assert "E054" in runner.invoke(app, ["lint", str(wf)]).output


def test_run_and_estimate_report_a_missing_or_broken_manifest_without_a_traceback(
    tmp_path: Path,
) -> None:
    inp = tmp_path / "in.json"
    inp.write_text("{}")
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "qsolve.yaml").write_text("name: [unclosed\n")
    for directory in (tmp_path / "nothing", broken):
        for cmd in ("run", "estimate"):
            result = runner.invoke(app, [cmd, str(directory), "-i", str(inp)])
            assert result.exit_code == 1, (cmd, result.output)
            assert "Traceback" not in result.output and isinstance(result.exception, SystemExit)
            assert "Error:" in result.output


def test_square_brackets_survive_in_help_and_errors(scaffold: Path, tmp_path: Path) -> None:
    help_text = runner.invoke(app, ["run", "--help"]).output
    assert "fake:<device>[@scale]" in help_text.replace("\n", "").replace("│", "").replace(" ", "")
    good = scaffold / "examples" / "qubo_5.json"
    bad = runner.invoke(app, ["run", str(scaffold), "-i", str(good), "--target", "nope"])
    assert bad.exit_code == 1 and "fake:<device>[@scale]" in bad.output.replace("\n", "")


def test_the_ibm_extra_hint_keeps_its_brackets(
    scaffold: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import qsolve.cli as cli

    def boom(*a: object, **k: object) -> None:
        raise RuntimeError("Install the ibm extra: pip install 'qsolve-sdk[ibm]'")

    monkeypatch.setattr(cli, "run_workflow", boom)
    # RuntimeError is not a user error; the hint must still print correctly where it is shown
    from qsolve.cli import _fail

    with cli.console.capture() as cap:
        _fail("Error:", "pip install 'qsolve-sdk[ibm]'")
    assert "qsolve-sdk[ibm]" in cap.get()


@pytest.mark.parametrize(
    "summary", ["a: b", "uses #hash # here", "- dash", "quote ' and \" both", "x\ny"]
)
def test_init_summary_is_written_as_safe_yaml(tmp_path: Path, summary: str) -> None:
    import yaml

    result = runner.invoke(
        app, ["init", "wf-one", "--summary", summary, "--directory", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    manifest = yaml.safe_load((tmp_path / "wf-one" / "qsolve.yaml").read_text(encoding="utf-8"))
    assert manifest["summary"] == " ".join(summary.split())
    assert runner.invoke(app, ["validate", str(tmp_path / "wf-one")]).exit_code == 0
