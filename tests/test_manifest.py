"""The manifest contract (schema_version 1)."""

from __future__ import annotations

import pytest

from qsolve.manifest import Manifest, Parameter

METHODS = [
    {"id": "qq", "role": "quantum", "title": "Q"},
    {"id": "rr", "role": "reference", "title": "R"},
]


def manifest(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": 1, "name": "demo-wf", "version": "0.1.0", "family": "x", "summary": "s",
        "license": "MIT", "inputs": {"schema": "a.json"}, "outputs": {"schema": "b.json"},
        "entrypoint": "src.workflow:W", "methods": METHODS,
        "envelope": {"max_qubits": 8, "max_variables": 8},
    }  # fmt: skip
    base.update(over)
    return base


def test_parameter_validation() -> None:
    p = Parameter(name="k", type="int", default=2, min=1, max=4)
    assert p.check(3) == 3
    for bad in (0, 5, True, 2.5):
        with pytest.raises(ValueError):
            p.check(bad)


def test_a_valid_manifest_and_the_ways_one_can_be_invalid() -> None:
    assert Manifest.model_validate(manifest()).entrypoint == "src.workflow:W"
    for bad in (
        manifest(name="Bad Name"),
        manifest(version="1.0"),
        manifest(unknown_field=1),
        manifest(schema_version=0),
        manifest(entrypoint="nocolon"),
        manifest(components={"builder": "a:B"}),  # the 0.x layout is gone
    ):
        with pytest.raises(ValueError):
            Manifest.model_validate(bad)


def test_methods_need_exactly_one_quantum_and_one_reference_with_unique_ids() -> None:
    extra = {"id": "cc", "role": "classical", "title": "C"}
    assert len(Manifest.model_validate(manifest(methods=[*METHODS, extra])).methods) == 3
    for bad in (
        [METHODS[0]],
        [],
        [*METHODS, {**METHODS[1], "id": "r2x"}],
        [*METHODS, {**extra, "id": "qq"}],
        [*METHODS, {**extra, "runner": "a:B"}],  # a runner is a 0.x idea
    ):
        with pytest.raises(ValueError):
            Manifest.model_validate(manifest(methods=bad))


def test_resolve_params_rejects_unknown_and_applies_defaults() -> None:
    m = Manifest.model_validate(
        manifest(parameters=[{"name": "shots", "type": "int", "default": 100, "min": 10}])
    )
    assert m.resolve_params() == {"shots": 100}
    assert m.resolve_params({"shots": 50}) == {"shots": 50}
    with pytest.raises(ValueError):
        m.resolve_params({"nope": 1})


def test_parameter_definitions_that_cannot_work_are_rejected() -> None:
    nan, inf = float("nan"), float("inf")
    for bad in (
        {"type": "int", "default": 2, "min": 5, "max": 1},  # min > max
        {"type": "enum", "default": "a"},  # no choices
        {"type": "enum", "default": "a", "choices": []},
        {"type": "float", "default": 1.0, "min": nan},
        {"type": "float", "default": 1.0, "max": inf},
    ):
        with pytest.raises(ValueError):
            Parameter(name="k", **bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):  # the manifest loads parameters the same way
        Manifest.model_validate(
            manifest(parameters=[{"name": "k", "type": "int", "default": 1, "min": 3, "max": 2}])
        )


def test_values_nan_inf_and_bool_do_not_pass() -> None:
    f = Parameter(name="x", type="float", default=1.0, min=0, max=2)
    for bad in (float("nan"), float("inf"), -float("inf"), True):
        with pytest.raises(ValueError):
            f.check(bad)
    unbounded = Parameter(name="x", type="float", default=1.0)
    with pytest.raises(ValueError):
        unbounded.check(float("nan"))  # NaN fails no comparison, so it needs its own check
    e = Parameter(name="e", type="enum", default=1, choices=[1, 2])
    assert e.check(2) == 2
    with pytest.raises(ValueError):
        e.check(True)  # True == 1 in Python, but it is not the choice 1
    assert Parameter(name="b", type="enum", default=True, choices=[True]).check(True) is True
