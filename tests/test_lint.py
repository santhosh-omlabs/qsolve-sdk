import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from qsolve.lint import has_errors, lint_workflow

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


TEMPLATE = Path(__file__).resolve().parents[1] / "src" / "qsolve" / "templates" / "qiskit"


def clone(tmp_path: Path) -> Path:
    """A fresh scaffold: the workflow `qsolve init` writes."""
    dst = tmp_path / "wf"
    shutil.copytree(TEMPLATE, dst, ignore=shutil.ignore_patterns("__pycache__"))
    manifest = dst / "qsolve.yaml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8")
        .replace("__NAME__", "scaffold")
        .replace("__SUMMARY__", "s"),
        encoding="utf-8",
    )
    (dst / "src" / "__init__.py").touch()
    (dst / "LICENSE").write_text("Apache-2.0")
    (dst / "README.md").write_text("# scaffold")
    return dst


def codes(root: Path) -> set[str]:
    return {f.code for f in lint_workflow(root)}


@pytest.mark.skipif(not EXAMPLES.is_dir(), reason="the examples repository is not next to this one")
@pytest.mark.parametrize(
    "name",
    ["portfolio-optimization", "max-cut", "generic-qubo", "molecular-ground-state",
     "interaction-energy", "electrolyte-solvent-screen", "quantum-kernel-classifier"],
)  # fmt: skip
def test_first_party_workflows_are_clean(name: str) -> None:
    # W052 (min_qubits_qpu is ignored by the platform) is expected until the examples drop it
    findings = [f for f in lint_workflow(EXAMPLES / name) if f.code != "W052"]
    assert findings == [], [f.as_dict() for f in findings]


def test_missing_manifest(tmp_path: Path) -> None:
    assert "E001" in codes(tmp_path)


def test_unavailable_imports_are_errors_but_stdlib_and_sdk_are_fine(tmp_path: Path) -> None:
    wf = clone(tmp_path)
    (wf / "src" / "extra.py").write_text(
        "import json, math\nimport numpy\nimport requests\nimport torch.nn\n"
    )
    found = [f for f in lint_workflow(wf) if f.code == "E040"]
    assert {f.message.split("'")[1] for f in found} == {"requests", "torch"}
    assert has_errors(found)


def test_tests_directory_is_not_checked_for_imports(tmp_path: Path) -> None:
    wf = clone(tmp_path)
    (wf / "tests").mkdir(exist_ok=True)
    (wf / "tests" / "test_x.py").write_text("import pytest\n")
    assert "E040" not in codes(wf)


def test_risky_calls_are_warnings_not_errors(tmp_path: Path) -> None:
    wf = clone(tmp_path)
    (wf / "src" / "risky.py").write_text("import subprocess\neval('1')\n")
    risky = [f for f in lint_workflow(wf) if f.code == "W041"]
    assert len(risky) == 2 and not has_errors(risky)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string", "pattern": "^(a+)+$"},
        {"type": "object", "patternProperties": {"^x": {"type": "string"}}},
        {"$ref": "https://evil.example/schema.json"},
        {"type": "string", "format": "regex"},
    ],
)
def test_schema_regexes_and_remote_refs_are_rejected(tmp_path: Path, schema: dict) -> None:
    wf = clone(tmp_path)
    s = json.loads((wf / "schemas" / "input.json").read_text())
    s["properties"]["bad"] = schema
    (wf / "schemas" / "input.json").write_text(json.dumps(s))
    assert "E021" in codes(wf)


def test_invalid_json_schema_and_bad_example(tmp_path: Path) -> None:
    wf = clone(tmp_path)
    s = json.loads((wf / "schemas" / "input.json").read_text())
    s["properties"]["Q"]["type"] = "not-a-type"
    (wf / "schemas" / "input.json").write_text(json.dumps(s))
    assert "E020" in codes(wf)
    wf2 = clone(tmp_path / "second")
    (wf2 / "examples" / "qubo_5.json").write_text(json.dumps({"nope": 1}))
    assert "E023" in codes(wf2)


def test_an_unresolvable_entrypoint_and_syntax_errors(tmp_path: Path) -> None:
    wf = clone(tmp_path)
    m = yaml.safe_load((wf / "qsolve.yaml").read_text())
    m["entrypoint"] = "src.workflow:DoesNotExist"
    (wf / "qsolve.yaml").write_text(yaml.safe_dump(m))
    assert "E030" in codes(wf)
    m["entrypoint"] = "src.missing_module:X"
    (wf / "qsolve.yaml").write_text(yaml.safe_dump(m))
    assert "E030" in codes(wf)
    (wf / "src" / "workflow.py").write_text("def broken(:" + chr(10))
    assert "E031" in codes(wf)


def test_secrets_and_credential_files_are_caught(tmp_path: Path) -> None:
    wf = clone(tmp_path)
    (wf / ".env").write_text("X=1\n")
    (wf / "src" / "cfg.py").write_text(
        'KEY = "AKIAABCDEFGHIJKLMNOP"\nGH = "ghp_' + "a" * 36 + '"\n'
    )
    (wf / "deploy.pem").write_text("-----BEGIN PRIVATE KEY-----\nabc\n")
    found = lint_workflow(wf)
    assert sum(f.code == "E012" for f in found) == 2  # .env and .pem
    assert sum(f.code == "E013" for f in found) == 2  # AWS key + GitHub token (the .pem is E012)


def test_missing_benchmarks_license_and_examples(tmp_path: Path) -> None:
    wf = clone(tmp_path)
    (wf / "benchmarks" / "cases.yaml").unlink()
    m = yaml.safe_load((wf / "qsolve.yaml").read_text())
    m["license"] = "  "
    m["inputs"]["examples"] = []
    (wf / "qsolve.yaml").write_text(yaml.safe_dump(m))
    assert {"E050", "E051", "E024"} <= codes(wf)


def test_symlinks_are_rejected(tmp_path: Path) -> None:
    wf = clone(tmp_path)
    try:
        (wf / "src" / "link.py").symlink_to(wf / "src" / "workflow.py")
    except OSError:
        pytest.skip("symlinks unavailable")
    assert "E010" in codes(wf)


def test_a_package_pinned_in_the_lock_is_a_warning_not_an_error(tmp_path: Path) -> None:
    """The runtime image is shared today: the pin is acknowledged, but the sandbox rejects it."""
    wf = clone(tmp_path)
    (wf / "src" / "extra.py").write_text("import networkx\nimport yaml\n")
    assert "E040" in codes(wf)
    lock = "# pinned\nnetworkx==3.3 --hash=sha256:abc\nPyYAML==6.0\n"
    (wf / "requirements.lock").write_text(lock)
    found = codes(wf)
    assert "E040" not in found and "W042" in found


def test_the_reproducibility_rerun_uses_the_first_cases_own_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rerun must run where the first run did: on another target the result differs and an
    honest workflow would be failed for it."""
    from qsolve import benchmark

    wf = clone(tmp_path)
    cases = yaml.safe_load((wf / "benchmarks" / "cases.yaml").read_text(encoding="utf-8"))
    first = {**cases["cases"][0], "target": "sim-cpu"}
    (wf / "benchmarks" / "cases.yaml").write_text(
        yaml.safe_dump({"cases": [first]}), encoding="utf-8"
    )
    targets: list[str] = []
    real = benchmark.run_workflow

    def spy(*a: Any, **kw: Any) -> Any:
        targets.append(kw["target"])
        return real(*a, **kw)

    monkeypatch.setattr(benchmark, "run_workflow", spy)
    benchmark.run_benchmarks(wf, target="fake:fez@1.6")  # a default the case overrides
    assert targets == ["sim-cpu", "sim-cpu"]  # the case, then its reproducibility rerun


def _scaffold_with_cases(tmp_path: Path, cases: list[dict[str, Any]]) -> Path:
    wf = clone(tmp_path)
    (wf / "benchmarks" / "cases.yaml").write_text(
        yaml.safe_dump({"cases": cases}), encoding="utf-8"
    )
    return wf


def test_a_fidelity_bound_on_an_ideal_simulator_does_not_validate_mitigation(
    tmp_path: Path,
) -> None:
    from qsolve.benchmark import run_benchmarks

    case = {"name": "x", "input_file": "examples/qubo_5.json", "seed": 1,
            "expect": {"max_gap": 0.0, "min_fidelity": 0.1}}  # fmt: skip
    gates = {g.name: g for g in run_benchmarks(_scaffold_with_cases(tmp_path, [case])).gates}
    assert (
        not gates["fidelity_checked"].passed and "noisy target" in gates["fidelity_checked"].detail
    )


def test_a_failing_rerun_is_a_failed_gate_not_a_crashed_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qsolve import benchmark

    case = {
        "name": "x",
        "input_file": "examples/qubo_5.json",
        "seed": 1,
        "expect": {"max_gap": 0.0},
    }
    wf = _scaffold_with_cases(tmp_path, [case])
    real, calls = benchmark.run_workflow, []

    def flaky(*a: Any, **kw: Any) -> Any:
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("second run fails")
        return real(*a, **kw)

    monkeypatch.setattr(benchmark, "run_workflow", flaky)
    gates = {g.name: g for g in benchmark.run_benchmarks(wf).gates}
    assert not gates["reproducible"].passed and "second run fails" in gates["reproducible"].detail


def test_a_dynamic_import_is_flagged_because_the_static_check_cannot_see_it(tmp_path: Path) -> None:
    wf = clone(tmp_path)
    source = "import importlib\nm = importlib.import_module('requests')\n"
    (wf / "src" / "dyn.py").write_text(source)
    assert "W043" in codes(wf)


def set_params(wf: Path, params: list[dict[str, Any]], **extra: Any) -> None:
    manifest = yaml.safe_load((wf / "qsolve.yaml").read_text(encoding="utf-8"))
    manifest["parameters"] = params
    manifest.update(extra)
    (wf / "qsolve.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def test_shots_above_the_platform_cap_are_an_error(tmp_path: Path) -> None:
    from qsolve import MAX_SHOTS_PER_CIRCUIT

    assert MAX_SHOTS_PER_CIRCUIT == 100_000
    wf = clone(tmp_path)
    ok = {"name": "shots", "type": "int", "default": 1000, "min": 100, "max": MAX_SHOTS_PER_CIRCUIT}
    set_params(wf, [ok])
    assert "E025" not in codes(wf)
    for bad in (
        {**ok, "max": MAX_SHOTS_PER_CIRCUIT + 1},
        {**ok, "max": None},  # no ceiling
        {**ok, "default": MAX_SHOTS_PER_CIRCUIT + 1, "max": 1_000_000},
    ):
        set_params(wf, [bad])
        assert "E025" in codes(wf), bad


def test_min_qubits_qpu_is_accepted_with_a_warning(tmp_path: Path) -> None:
    wf = clone(tmp_path)
    set_params(wf, [], targets={"simulators": ["cpu"], "min_qubits_qpu": 8})
    found = {f.code: f.severity for f in lint_workflow(wf)}
    assert found.get("W052") == "warning" and "E002" not in found


def rewrite_schema(wf: Path, edit: Any) -> None:
    path = wf / "schemas" / "input.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    edit(schema)
    path.write_text(json.dumps(schema), encoding="utf-8")


def test_a_property_named_pattern_is_not_the_pattern_keyword(tmp_path: Path) -> None:
    wf = clone(tmp_path)

    def edit(schema: dict[str, Any]) -> None:
        schema["properties"]["pattern"] = {"type": "string", "maxLength": 5}
        schema["properties"]["patternProperties"] = {"type": "integer"}
        schema.setdefault("required", []).extend(["pattern", "patternProperties"])
        schema["examples"] = [{"pattern": "x", "pattern2": {"pattern": 1}}]
        schema["definitions"] = {"pattern": {"type": "string"}}

    rewrite_schema(wf, edit)
    assert "E021" not in codes(wf)


def test_pattern_as_a_keyword_is_still_an_error_at_any_depth(tmp_path: Path) -> None:
    wf = clone(tmp_path)

    def edit(schema: dict[str, Any]) -> None:
        schema["properties"]["name"] = {"type": "string", "pattern": "^a+$"}
        schema["properties"]["deep"] = {
            "type": "array",
            "items": {"anyOf": [{"patternProperties": {"a": {}}}]},
        }

    rewrite_schema(wf, edit)
    found = [f for f in lint_workflow(wf) if f.code == "E021"]
    assert len(found) == 2


def write_cases(wf: Path, cases: str) -> None:
    (wf / "benchmarks" / "cases.yaml").write_text(cases, encoding="utf-8")


GOOD_INPUT = "input_file: examples/qubo_5.json"


def test_cases_need_an_input(tmp_path: Path) -> None:
    wf = clone(tmp_path)
    write_cases(wf, "cases:\n  - {name: a}\n")
    assert "E055" in codes(wf)
    write_cases(wf, f"cases:\n  - {{name: a, {GOOD_INPUT}}}\n")
    assert codes(wf) == set()


def test_case_input_files_must_exist_and_stay_inside_the_repository(tmp_path: Path) -> None:
    wf = clone(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    for ref in ("../outside.json", str(outside), "examples/missing.json"):
        write_cases(wf, f"cases:\n  - {{name: a, input_file: '{ref}'}}\n")
        assert "E056" in codes(wf), ref
        write_cases(
            wf,
            "hardware_cases:\n"
            f"  - {{name: a, target: 'ibm:ibm_fez', input_file: '{ref}'}}\n"
            "cases: []\n",
        )
        assert "E056" in codes(wf), ref


def test_case_input_is_checked_against_the_input_schema(tmp_path: Path) -> None:
    wf = clone(tmp_path)
    write_cases(wf, "cases:\n  - {name: a, input: {not_a_qubo: 1}}\n")
    assert "E059" in codes(wf)


def test_case_names_are_unique_and_real_devices_stay_out_of_cases(tmp_path: Path) -> None:
    wf = clone(tmp_path)
    write_cases(wf, f"cases:\n  - {{name: a, {GOOD_INPUT}}}\n  - {{name: a, {GOOD_INPUT}}}\n")
    assert "E057" in codes(wf)
    write_cases(wf, f"cases:\n  - {{name: a, {GOOD_INPUT}, target: 'ibm:ibm_fez'}}\n")
    assert "E058" in codes(wf)
    write_cases(wf, f"cases:\n  - {{name: a, {GOOD_INPUT}, target: 'fake:fez'}}\n")
    assert "E058" not in codes(wf)


def test_unparseable_entrypoint_module_is_a_finding_not_a_crash(tmp_path: Path) -> None:
    wf = clone(tmp_path)
    (wf / "src" / "workflow.py").write_bytes(b"x = 1\n\xff\xfe\x00bad")
    found = [f for f in lint_workflow(wf) if f.code == "E031"]
    assert found and all(f.path.replace("\\", "/") == "src/workflow.py" for f in found)


def test_env_templates_are_allowed_but_real_env_files_are_not(tmp_path: Path) -> None:
    wf = clone(tmp_path)
    for name in (".env.example", ".env.sample", ".env.template"):
        (wf / name).write_text("TOKEN=\n")
    assert "E012" not in codes(wf)
    for name in (".env", ".env.local", ".env.production"):
        (wf / name).write_text("TOKEN=x\n")
    assert len([f for f in lint_workflow(wf) if f.code == "E012"]) == 3
