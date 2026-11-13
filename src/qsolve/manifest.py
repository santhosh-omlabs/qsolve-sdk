"""qsolve.yaml manifest: schema, loading and parameter handling."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1
# The most shots the platform runs for one circuit. A `shots` parameter may not declare more
# (lint `E025`); the platform executor refuses a larger request.
MAX_SHOTS_PER_CIRCUIT = 100_000
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")
_REF_RE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SchemaRef(_Strict):
    """A JSON Schema file (`schema:`) and example inputs (`examples:`), relative to the root."""

    schema_path: str = Field(alias="schema")
    examples: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Parameter(_Strict):
    """A user-tunable parameter declared under `parameters:`.

    Attributes:
        name: Key in `params`.
        type: `int`, `float`, `bool`, `str` or `enum`.
        default: Used when the user gives no value; must itself pass `check`.
        label: Short name for the form (up to 60 characters).
        description: Help text.
        min / max: Bounds for `int` and `float`; finite, and `min` at most `max`.
        choices: Allowed values for `enum`; required (non-empty) for `enum`.
    """

    name: str
    type: Literal["int", "float", "bool", "str", "enum"]
    default: Any
    label: str = Field(default="", max_length=60)
    description: str = ""
    min: float | None = Field(default=None, allow_inf_nan=False)
    max: float | None = Field(default=None, allow_inf_nan=False)
    choices: list[Any] | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Parameter:
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(
                f"parameter {self.name!r}: min {self.min} is greater than max {self.max}"
            )
        if self.type == "enum" and not self.choices:
            raise ValueError(f"parameter {self.name!r}: an enum needs a non-empty `choices` list")
        return self

    def check(self, value: Any) -> Any:
        """Validate one value against the type and bounds.

        Returns:
            The value, unchanged.

        Raises:
            ValueError: wrong type (a bool is not a number or an enum choice), NaN or infinite,
                out of bounds, or not one of `choices`.
        """
        t = self.type
        if t == "int" and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"parameter {self.name!r} must be an int")
        if t == "float" and (isinstance(value, bool) or not isinstance(value, int | float)):
            raise ValueError(f"parameter {self.name!r} must be a number")
        if t == "bool" and not isinstance(value, bool):
            raise ValueError(f"parameter {self.name!r} must be a bool")
        if t == "str" and not isinstance(value, str):
            raise ValueError(f"parameter {self.name!r} must be a string")
        if t == "enum" and not any(
            isinstance(value, bool) == isinstance(c, bool) and value == c
            for c in self.choices or []
        ):  # True == 1 in Python, but True is not the choice 1
            raise ValueError(f"parameter {self.name!r} must be one of {self.choices}")
        if t in ("int", "float"):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"parameter {self.name!r} must be a finite number")
            if self.min is not None and value < self.min:
                raise ValueError(f"parameter {self.name!r} must be >= {self.min}")
            if self.max is not None and value > self.max:
                raise ValueError(f"parameter {self.name!r} must be <= {self.max}")
        return value


class ObjectiveDecl(_Strict):
    """What the numbers a method reaches mean: shown next to every value."""

    label: str = "Objective"
    unit: str = ""
    decimals: int = Field(default=4, ge=0, le=10)


class MetricDecl(_Strict):
    """How the headline accuracy is computed. The kinds are generic maths implemented once by the
    platform; a workflow only picks one and explains it in words."""

    kind: Literal["relative_gap", "relative_error", "error_rate"] = "relative_gap"
    note: str = Field(default="", max_length=600)
    # accuracy within this of 100% counts as matching the reference (e.g. 0.002)
    tolerance: float = Field(default=0.0, ge=0, le=1)
    # or: a difference to the reference up to this, in the objective's own units, counts as matching
    # (e.g. 0.0016 Ha, chemical accuracy)
    tolerance_abs: float | None = Field(default=None, ge=0)


class MethodDecl(_Strict):
    """A method that is run and compared on every input, and what it is, in plain words. It is
    implemented by the workflow class: `quantum`, `reference`, or `classical_<id>` (dashes in the
    id become underscores)."""

    id: str
    role: Literal["quantum", "reference", "classical"]
    title: str = Field(max_length=80)
    what: str = Field(default="", max_length=500)
    strength: str = Field(default="", max_length=300)
    limit: str = Field(default="", max_length=300)

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not _NAME_RE.match(v):
            raise ValueError("method id must be lowercase letters, digits and dashes")
        return v


class Envelope(_Strict):
    """The size limits a workflow supports: `max_qubits`, `max_variables` and a `notes` text."""

    max_qubits: int = Field(gt=0, le=1000)
    max_variables: int = Field(gt=0, le=100000)
    notes: str = ""


class Targets(_Strict):
    """Where it may run: `simulators` (`cpu`, `gpu`), `qpu` (`ibm`) and `min_qubits_qpu`."""

    simulators: list[Literal["cpu", "gpu"]] = ["cpu"]
    qpu: list[Literal["ibm"]] = Field(default_factory=list)
    min_qubits_qpu: int | None = None


class Resources(_Strict):
    """Sandbox limits: `timeout_s` (1800), `memory_mb` (4096) and `network` (always `none`)."""

    timeout_s: int = Field(default=1800, gt=0, le=86400)
    memory_mb: int = Field(default=4096, ge=256, le=65536)
    network: Literal["none"] = "none"


class Runtime(_Strict):
    """The runtime a workflow needs: `python` version and the `dependencies` lock file."""

    python: str = "3.12"
    dependencies: str = "requirements.lock"


class Manifest(_Strict):
    """The parsed `qsolve.yaml`. Unknown keys are rejected (see the manifest docs)."""

    schema_version: Literal[1]
    name: str
    version: str
    family: str
    summary: str
    license: str
    tags: list[str] = Field(default_factory=list)
    inputs: SchemaRef
    parameters: list[Parameter] = Field(default_factory=list)
    outputs: SchemaRef
    # the Workflow subclass: package.module:ClassName
    entrypoint: str
    # what a person sees: the field it belongs to, what its numbers mean, how accuracy is measured
    # and every method compared. All optional; a workflow that declares none is still compared with
    # its reference, just without descriptions.
    category: str = Field(default="", max_length=60)
    objective: ObjectiveDecl = Field(default_factory=ObjectiveDecl)
    metric: MetricDecl = Field(default_factory=MetricDecl)
    methods: list[MethodDecl] = Field(min_length=2)
    envelope: Envelope
    targets: Targets = Field(default_factory=Targets)
    resources: Resources = Field(default_factory=Resources)
    runtime: Runtime = Field(default_factory=Runtime)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not _NAME_RE.match(v):
            raise ValueError("name must be lowercase letters, digits and dashes (2-63 chars)")
        return v

    @field_validator("version")
    @classmethod
    def _version(cls, v: str) -> str:
        if not _SEMVER_RE.match(v):
            raise ValueError("version must be semver, e.g. 0.1.0")
        return v

    @field_validator("entrypoint")
    @classmethod
    def _entrypoint(cls, v: str) -> str:
        if not _REF_RE.match(v):
            raise ValueError("entrypoint must look like 'package.module:ClassName'")
        return v

    @field_validator("methods")
    @classmethod
    def _methods(cls, v: list[MethodDecl]) -> list[MethodDecl]:
        ids = [m.id for m in v]
        if len(ids) != len(set(ids)):
            raise ValueError("method ids must be unique")
        for role in ("quantum", "reference"):
            if sum(m.role == role for m in v) != 1:
                raise ValueError(f"declare exactly one method with role {role!r}")
        return v

    @field_validator("parameters")
    @classmethod
    def _unique_params(cls, v: list[Parameter]) -> list[Parameter]:
        names = [p.name for p in v]
        if len(names) != len(set(names)):
            raise ValueError("parameter names must be unique")
        for p in v:
            p.check(p.default)
        return v

    def resolve_params(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        """Defaults + overrides, validated. Unknown parameters are rejected."""
        overrides = overrides or {}
        known = {p.name: p for p in self.parameters}
        unknown = set(overrides) - set(known)
        if unknown:
            raise ValueError(f"unknown parameters: {sorted(unknown)}")
        return {n: p.check(overrides.get(n, p.default)) for n, p in known.items()}


class LoadedWorkflow(_Strict):
    """A workflow directory read from disk: `root`, parsed `manifest` and the two JSON Schemas."""

    root: Path
    manifest: Manifest
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]


def load_workflow(root: str | Path) -> LoadedWorkflow:
    """Read and validate a workflow directory.

    Args:
        root: Directory holding `qsolve.yaml`.

    Returns:
        A `LoadedWorkflow` with the manifest and both schemas.

    Raises:
        FileNotFoundError: no `qsolve.yaml`.
        ValueError: invalid YAML, a missing or non-object schema, or a schema path that
            leaves the directory.
        pydantic.ValidationError: the manifest breaks a rule.
    """
    root = Path(root).resolve()
    mpath = root / "qsolve.yaml"
    if not mpath.is_file():
        raise FileNotFoundError(f"No qsolve.yaml in {root}")
    try:
        raw = yaml.safe_load(mpath.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ValueError(f"qsolve.yaml is not valid YAML: {e}") from e
    manifest = Manifest.model_validate(raw)

    def _read_schema(rel: str) -> dict[str, Any]:
        p = (root / rel).resolve()
        if root not in p.parents:
            raise ValueError(f"schema path escapes workflow directory: {rel}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError as e:
            raise ValueError(f"schema file not found: {rel}") from e
        if not isinstance(data, dict):
            raise ValueError(f"{rel} must contain a JSON object")
        return data

    return LoadedWorkflow(
        root=root,
        manifest=manifest,
        input_schema=_read_schema(manifest.inputs.schema_path),
        output_schema=_read_schema(manifest.outputs.schema_path),
    )
