"""Static checks on a workflow repository: no code is executed.

The same checks run locally (`qsolve lint`) and on the platform before anything is built or run.
They catch hygiene and policy problems early with actionable messages. They are defense in depth;
the sandbox, not this linter, is the security boundary.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import jsonschema
import yaml
from pydantic import ValidationError

from .manifest import MAX_SHOTS_PER_CIRCUIT, load_workflow

Severity = Literal["error", "warning"]

MAX_FILES = 500
MAX_TOTAL_BYTES = 20 * 1024 * 1024
MAX_SCHEMA_BYTES = 256 * 1024
MAX_TEXT_SCAN_BYTES = 1024 * 1024

SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "node_modules",
}

# Third-party modules available in the platform runtime image. Workflows may import these and the
# standard library; anything else cannot be installed at run time (no network in the sandbox).
ALLOWED_MODULES = frozenset(
    {
        "qsolve", "numpy", "scipy", "sklearn", "qiskit", "qiskit_aer", "pyscf",
        "pydantic", "jsonschema", "yaml", "typer", "rich",
    }
)  # fmt: skip

# Import names that differ from the distribution name a lock file lists.
_DIST_TO_MODULE = {
    "scikit_learn": "sklearn",
    "pyyaml": "yaml",
    "pillow": "PIL",
    "opencv_python": "cv2",
}


def locked_modules(root: Path) -> frozenset[str]:
    """Import names of the packages a workflow pins in its own requirements.lock."""
    lock = root / "requirements.lock"
    if not lock.is_file():
        return frozenset()
    names: set[str] = set()
    for line in lock.read_text(encoding="utf-8").splitlines():
        m = re.match(
            r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(?:==|>=|~=|<=|$)", line
        )
        if m and not line.lstrip().startswith("#"):
            dist = re.sub(r"[-.]+", "_", m.group(1)).lower()
            names.add(_DIST_TO_MODULE.get(dist, dist))
    return frozenset(names)


RISKY_IMPORTS = {"subprocess", "socket", "ctypes", "pickle", "marshal", "shelve", "multiprocessing"}
RISKY_CALLS = {"eval", "exec", "compile", "__import__"}

_FORBIDDEN_FILES = re.compile(
    r"(^|/)(\.env(?!\.(?:example|sample|template)$)(\..*)?|id_rsa.*|id_ed25519.*|.*\.pem|.*\.p12|.*\.pfx|.*\.key|credentials.*\.json|\.netrc)$"
)
_SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "Supabase/Stripe-style secret": re.compile(
        r"\b(?:sb_secret|sk_live|sk_test)_[A-Za-z0-9]{16,}\b"
    ),
    "IBM Quantum token": re.compile(
        r"(?i)\b(?:ibm|qiskit)[\w-]*token\s*[:=]\s*['\"][A-Za-z0-9_-]{30,}['\"]"
    ),
}  # fmt: skip


@dataclass(frozen=True)
class Finding:
    """One lint finding.

    Attributes:
        severity: `error` (fails the check) or `warning`.
        code: `E###` or `W###`.
        message: What is wrong and how to fix it.
        path: File the finding is about, relative to the workflow root; may be empty.
    """

    severity: Severity
    code: str
    message: str
    path: str = ""

    def as_dict(self) -> dict[str, str]:
        """The finding as a plain dict with keys `severity`, `code`, `message`, `path`."""
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }


def _iter_files(root: Path) -> tuple[list[Path], list[Finding]]:
    findings: list[Finding] = []
    files: list[Path] = []
    total = 0
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if p.is_symlink():
            findings.append(Finding("error", "E010", "symbolic links are not allowed", str(rel)))
            continue
        if p.is_file():
            files.append(p)
            total += p.stat().st_size
    if len(files) > MAX_FILES:
        findings.append(Finding("error", "E011", f"too many files ({len(files)} > {MAX_FILES})"))
    if total > MAX_TOTAL_BYTES:
        findings.append(
            Finding("error", "E011", f"repository too large ({total} > {MAX_TOTAL_BYTES} bytes)")
        )
    return files, findings


# Keys whose value is data, not a schema; and keys whose value maps free names to schemas.
_SCHEMA_DATA_KEYS = frozenset({"required", "enum", "const", "default", "examples"})
_SCHEMA_MAPS = frozenset({"properties", "definitions", "$defs", "dependencies"})


def _check_schema(rel: str, schema: Any) -> list[Finding]:
    out: list[Finding] = []
    try:
        jsonschema.Draft7Validator.check_schema(schema)
    except jsonschema.SchemaError as e:
        return [Finding("error", "E020", f"invalid JSON Schema: {e.message}", rel)]

    def walk(node: Any, trail: str) -> None:
        """Visit a schema. Only keys in keyword position are checked: a property that is merely
        *named* `pattern` (a key under `properties`) and plain data (`required`, `enum`, `const`,
        `default`, `examples`) are not keywords."""
        if isinstance(node, list):  # allOf / anyOf / oneOf / items as a list
            for i, v in enumerate(node):
                walk(v, f"{trail}/{i}")
            return
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            here = f"{trail}/{key}"
            if key in ("pattern", "patternProperties"):
                out.append(
                    Finding(
                        "error", "E021",
                        f"'{key}' is not allowed in input/output schemas (regular "
                        f"expressions can be abused to hang validation); "
                        f"use enum, minLength or maxLength instead",
                        f"{rel}#{here}",
                    )
                )  # fmt: skip
            elif key == "$ref" and isinstance(value, str) and not value.startswith("#"):
                out.append(
                    Finding(
                        "error",
                        "E021",
                        "only local '#/...' $ref values are allowed",
                        f"{rel}#{here}",
                    )
                )
            elif key == "format" and value == "regex":
                out.append(
                    Finding("error", "E021", "format 'regex' is not allowed", f"{rel}#{here}")
                )
            if key in _SCHEMA_DATA_KEYS:
                continue
            if key in _SCHEMA_MAPS and isinstance(value, dict):  # names are free, values schemas
                for name, sub_schema in value.items():
                    walk(sub_schema, f"{here}/{name}")
            else:
                walk(value, here)

    walk(schema, "")
    return out


def _module_file(root: Path, module: str) -> Path | None:
    parts = module.split(".")
    for base in (root, root / "src"):
        for cand in (
            base.joinpath(*parts).with_suffix(".py"),
            base.joinpath(*parts, "__init__.py"),
        ):
            if cand.is_file():
                return cand
    return None


def _check_components(root: Path, components: dict[str, str]) -> list[Finding]:
    out: list[Finding] = []
    for role, ref in components.items():
        module, cls = ref.split(":", 1)
        if module == "qsolve" or module.startswith("qsolve."):
            continue  # SDK component; verified at run time
        f = _module_file(root, module)
        if f is None:
            out.append(
                Finding(
                    "error",
                    "E030",
                    f"{role}: module '{module}' not found in the repository",
                    "qsolve.yaml",
                )
            )
            continue
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as e:
            out.append(
                Finding(
                    "error",
                    "E031",
                    f"{f.name}: syntax error line {e.lineno}: {e.msg}",
                    str(f.relative_to(root)),
                )
            )
            continue
        except (ValueError, OSError) as e:  # not UTF-8 (UnicodeDecodeError), null bytes, unreadable
            out.append(
                Finding(
                    "error", "E031", f"{f.name}: cannot parse Python: {e}", str(f.relative_to(root))
                )
            )
            continue
        names = {n.name for n in tree.body if isinstance(n, ast.ClassDef | ast.FunctionDef)}
        names |= {
            t.id
            for n in tree.body
            if isinstance(n, ast.Assign)
            for t in n.targets
            if isinstance(t, ast.Name)
        }
        if cls not in names:
            out.append(
                Finding(
                    "error", "E030", f"{role}: '{cls}' is not defined in {f.name}", "qsolve.yaml"
                )
            )
    return out


def _top_level_modules(root: Path) -> set[str]:
    names: set[str] = set()
    for base in (root, root / "src"):
        if not base.is_dir():
            continue
        for p in base.iterdir():
            if p.is_dir() and (p / "__init__.py").is_file():
                names.add(p.name)
            elif p.suffix == ".py":
                names.add(p.stem)
    names.add("src")  # workflows are imported as `src.<module>`
    return names


def _check_python(root: Path, files: list[Path]) -> list[Finding]:
    out: list[Finding] = []
    own = _top_level_modules(root)
    stdlib = set(sys.stdlib_module_names)
    locked = locked_modules(root)
    for f in files:
        if f.suffix != ".py" or "tests" in f.relative_to(root).parts:
            continue  # tests are never shipped to or run in the runtime
        rel = str(f.relative_to(root))
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except (SyntaxError, ValueError, OSError) as e:  # ValueError covers UnicodeDecodeError
            out.append(Finding("error", "E031", f"cannot parse Python: {e}", rel))
            continue
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            for m in mods:
                top = m.split(".")[0]
                if top in RISKY_IMPORTS:
                    out.append(
                        Finding(
                            "warning",
                            "W041",
                            f"imports '{top}' (blocked or useless in the sandbox; "
                            f"flagged for review)",
                            rel,
                        )
                    )
                if top in stdlib or top in own or top in ALLOWED_MODULES or top == "__future__":
                    continue
                if top in locked:
                    out.append(
                        Finding(
                            "warning", "W042",
                            f"imports '{top}', pinned in requirements.lock but not in the runtime "
                            "image, so the sandbox will reject it",
                            rel,
                        )
                    )  # fmt: skip
                    continue
                out.append(
                    Finding(
                        "error", "E040",
                        f"imports '{top}', which is not available in the runtime. "
                        f"Allowed: the Python "
                        f"standard library and {', '.join(sorted(ALLOWED_MODULES))}",
                        rel,
                    )
                )  # fmt: skip
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in RISKY_CALLS
            ):
                out.append(
                    Finding("warning", "W041", f"calls {node.func.id}() (flagged for review)", rel)
                )
            if (  # importlib.import_module(...) hides what is imported from the static check above
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"import_module", "__import__"}
            ):
                out.append(
                    Finding(
                        "warning", "W043",
                        "imports a module by name at run time (not checked: the runtime image "
                        "decides what is importable)",
                        rel,
                    )
                )  # fmt: skip
    return out


def _check_files(root: Path, files: list[Path]) -> list[Finding]:
    out: list[Finding] = []
    for f in files:
        rel = str(f.relative_to(root)).replace("\\", "/")
        if _FORBIDDEN_FILES.search(rel):
            out.append(
                Finding(
                    "error",
                    "E012",
                    "file looks like a credential; remove it and rotate the secret",
                    rel,
                )
            )
            continue
        if f.stat().st_size > MAX_TEXT_SCAN_BYTES or f.suffix in {
            ".png",
            ".jpg",
            ".gif",
            ".pdf",
            ".zip",
            ".npy",
            ".npz",
        }:
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pat in _SECRET_PATTERNS.items():
            if pat.search(text):
                out.append(
                    Finding(
                        "error",
                        "E013",
                        f"possible {label} committed; remove it and rotate the secret",
                        rel,
                    )
                )
    return out


def _check_hardware_cases(root: Path, m: Any) -> list[Finding]:
    """`hardware_cases:` are run on a real device on request; keep them few, small and declared."""
    from .benchmark import HARDWARE_TARGET, MAX_HARDWARE_CASES, load_hardware_cases

    rel = "benchmarks/cases.yaml"
    try:
        cases = load_hardware_cases(root)
    except (ValidationError, ValueError, TypeError, yaml.YAMLError) as e:
        return [Finding("error", "E052", f"hardware_cases are invalid: {e}"[:300], rel)]
    if not cases:
        return []
    out: list[Finding] = []
    if len(cases) > MAX_HARDWARE_CASES:
        out.append(
            Finding(
                "error", "E053", f"at most {MAX_HARDWARE_CASES} hardware_cases are allowed", rel
            )
        )
    for c in cases:
        if not c.target or not HARDWARE_TARGET.match(c.target):
            out.append(
                Finding(
                    "error",
                    "E052",
                    f"hardware case '{c.name}' needs target: ibm:<device>, for example ibm:ibm_fez",
                    rel,
                )
            )
    if "ibm" not in m.targets.qpu:
        out.append(
            Finding(
                "error", "E054", "hardware_cases need `qpu: [ibm]` under targets", "qsolve.yaml"
            )
        )
    return out


def _check_case_input(root: Path, case: Any, schema: dict[str, Any], rel: str) -> list[Finding]:
    """One case's input: present, inside the repository, readable and valid for the input schema."""
    label = f"case '{case.name}'"
    if case.input is None and case.input_file is None:
        return [Finding("error", "E055", f"{label} needs `input` or `input_file`", rel)]
    data: Any = case.input
    if case.input is None:
        ref = Path(case.input_file)
        if ref.is_absolute() or ref.drive or ref.root or ".." in ref.parts:
            return [
                Finding(
                    "error", "E056",
                    f"{label}: input_file {case.input_file!r} must be a relative path inside the "
                    f"repository (no absolute path, no '..')",
                    rel,
                )
            ]  # fmt: skip
        target = (root / ref).resolve()
        if root not in target.parents:
            return [Finding("error", "E056", f"{label}: input_file leaves the repository", rel)]
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return [
                Finding("error", "E056", f"{label}: input_file {case.input_file!r} not found", rel)
            ]
        except (OSError, ValueError) as e:
            return [Finding("error", "E056", f"{label}: input_file unreadable: {e}"[:300], rel)]
        if not isinstance(data, dict):
            return [Finding("error", "E056", f"{label}: input_file must hold a JSON object", rel)]
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        return [
            Finding(
                "error",
                "E059",
                f"{label}: input does not match the input schema: {e.message}"[:300],
                rel,
            )
        ]
    except jsonschema.SchemaError:
        pass  # the schema itself is invalid; already reported as E020
    return []


def _check_cases(root: Path, schema: dict[str, Any]) -> list[Finding]:
    """What `benchmarks/cases.yaml` says: each case has a usable input, names are unique and
    hardware targets stay in `hardware_cases`. (A file that is not valid YAML is `E052`.)"""
    from .benchmark import Case

    rel = "benchmarks/cases.yaml"
    path = root / "benchmarks" / "cases.yaml"
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, yaml.YAMLError):
        return []  # absent: E050; unreadable or not YAML: E052 from the hardware check
    if not isinstance(raw, dict):
        return [Finding("error", "E055", "cases.yaml must be a mapping with `cases:`", rel)]
    out: list[Finding] = []
    seen: set[str] = set()
    for key in ("cases", "hardware_cases"):
        try:
            cases = [Case.model_validate(c) for c in raw.get(key) or []]
        except (ValidationError, TypeError) as e:
            if key == "cases":  # an invalid hardware_cases list is E052
                out.append(Finding("error", "E055", f"cases are invalid: {e}"[:300], rel))
            continue
        for c in cases:
            if c.name in seen:
                out.append(Finding("error", "E057", f"duplicate case name '{c.name}'", rel))
            seen.add(c.name)
            if key == "cases" and c.target and c.target.startswith("ibm:"):
                out.append(
                    Finding(
                        "error", "E058",
                        f"case '{c.name}' targets {c.target}: a real device belongs under "
                        f"`hardware_cases`, not `cases` (which run on every validation)",
                        rel,
                    )
                )  # fmt: skip
            out += _check_case_input(root, c, schema, rel)
    return out


def _check_params(m: Any) -> list[Finding]:
    out: list[Finding] = []
    for p in m.parameters:
        if p.name == "shots" and p.type == "int":
            top = p.max if p.max is not None else float("inf")
            if top > MAX_SHOTS_PER_CIRCUIT or p.default > MAX_SHOTS_PER_CIRCUIT:
                out.append(
                    Finding(
                        "error", "E025",
                        f"parameter 'shots' allows up to {p.max} (default {p.default}) but the "
                        f"platform runs at most {MAX_SHOTS_PER_CIRCUIT:,} shots per circuit: "
                        f"set max (and default) to {MAX_SHOTS_PER_CIRCUIT} or less",
                        "qsolve.yaml",
                    )
                )  # fmt: skip
    if m.targets.min_qubits_qpu is not None:
        out.append(
            Finding(
                "warning", "W052",
                "targets.min_qubits_qpu is currently ignored by the platform",
                "qsolve.yaml",
            )
        )  # fmt: skip
    return out


def lint_workflow(root: str | Path) -> list[Finding]:
    """Run every static check on a workflow directory. No workflow code is executed.

    Args:
        root: The workflow directory.

    Returns:
        The findings (empty when clean). A missing or invalid `qsolve.yaml` returns `E001` or `E002`
        rather than raising.
    """
    root = Path(root).resolve()
    files, findings = _iter_files(root)
    try:
        wf = load_workflow(root)
    except FileNotFoundError:
        return [*findings, Finding("error", "E001", "qsolve.yaml not found at the repository root")]
    except (ValidationError, ValueError, json.JSONDecodeError) as e:
        return [
            *findings,
            Finding("error", "E002", f"invalid workflow definition: {e}", "qsolve.yaml"),
        ]

    m = wf.manifest
    for rel, schema in (
        (m.inputs.schema_path, wf.input_schema),
        (m.outputs.schema_path, wf.output_schema),
    ):
        if len(json.dumps(schema)) > MAX_SCHEMA_BYTES:
            findings.append(Finding("error", "E022", "schema is too large", rel))
        findings += _check_schema(rel, schema)
    for ex in m.inputs.examples:
        try:
            jsonschema.validate(
                json.loads((root / ex).read_text(encoding="utf-8")), wf.input_schema
            )
        except (OSError, json.JSONDecodeError) as e:
            findings.append(Finding("error", "E023", f"example input unreadable: {e}", ex))
        except jsonschema.ValidationError as e:
            findings.append(
                Finding(
                    "error",
                    "E023",
                    f"example input does not match the input schema: {e.message}",
                    ex,
                )
            )
        except jsonschema.SchemaError:
            pass  # the schema itself is invalid; already reported as E020
    if not m.inputs.examples:
        findings.append(
            Finding("error", "E024", "at least one example input is required", "qsolve.yaml")
        )
    if not (root / "benchmarks" / "cases.yaml").is_file():
        findings.append(
            Finding(
                "error",
                "E050",
                "benchmarks/cases.yaml is required: the platform benchmarks every workflow",
                "benchmarks/cases.yaml",
            )
        )
    findings += _check_params(m)
    findings += _check_cases(root, wf.input_schema)
    findings += _check_hardware_cases(root, m)
    if not m.license.strip():
        findings.append(Finding("error", "E051", "a license identifier is required", "qsolve.yaml"))
    if not any((root / n).is_file() for n in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING")):
        findings.append(Finding("warning", "W051", "no LICENSE file found", ""))
    if not (root / "README.md").is_file():
        findings.append(
            Finding(
                "warning",
                "W060",
                "no README.md; describe what the workflow solves and its limits",
                "",
            )
        )

    findings += _check_components(root, {"entrypoint": m.entrypoint})
    findings += _check_python(root, files)
    findings += _check_files(root, files)
    return findings


def has_errors(findings: list[Finding]) -> bool:
    """True when at least one finding has severity `error`."""
    return any(f.severity == "error" for f in findings)
