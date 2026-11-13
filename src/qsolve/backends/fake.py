"""A real device's published noise, simulated: the free rehearsal for hardware runs.

`fake:<device>` (e.g. `fake:fez`) transpiles exactly as for the device and samples on Aer with
the device's calibrated noise model (gate errors, readout errors, T1/T2). `fake:<device>@1.6`
multiplies every error rate by 1.6 and divides T1/T2 by it: a pessimistic rehearsal, since a real
device has tended to be worse than its own published model (the 1.6 is an empirical rule of thumb,
not a fitted constant).

Results carry the same metadata as the IBM backend (readout errors per measured bit, two-qubit
gate count), so workflows exercise their hardware path end to end without spending anything.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from qiskit import QuantumCircuit

from .base import Backend, SampleResult
from .hardware import device_metadata, readout_errors, transpile_best

_RATES = {"gate_error", "readout_error", "prob_meas0_prep1", "prob_meas1_prep0"}


def _fake_device(name: str) -> Any:
    try:
        from qiskit_ibm_runtime import fake_provider
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Install the ibm extra: pip install 'qsolve-sdk[ibm]'") from e
    cls = getattr(fake_provider, f"Fake{name.capitalize()}", None)
    if cls is None:
        raise ValueError(f"no fake device {name!r} (e.g. fez, marrakesh, torino)")
    return cls()


def _scaled(properties: Any, scale: float) -> Any:
    d = properties.to_dict()
    for qubit in d["qubits"]:
        if not any(p["name"] == "frequency" for p in qubit):
            # Aer reads it, but it only matters at a nonzero temperature (we simulate at zero);
            # Heron calibrations do not report it
            qubit.append(
                {"date": qubit[0]["date"], "name": "frequency", "unit": "GHz", "value": 5.0}
            )
        for p in qubit:
            if p["name"] in _RATES:
                p["value"] = min(float(p["value"]) * scale, 0.5)
            elif p["name"] in ("T1", "T2"):
                p["value"] = float(p["value"]) / scale
    for gate in d["gates"]:
        for p in gate["parameters"]:
            if p["name"] == "gate_error":
                p["value"] = min(float(p["value"]) * scale, 0.75)
    return type(properties).from_dict(d)


def _job_seed(seeds: Sequence[int]) -> int:
    """Aer takes one seed per job and derives each experiment's stream from it. For a batch, mix
    every circuit's seed into it, so each circuit's seed matters and the same seeds always give
    the same counts. One circuit keeps its own seed."""
    if len(seeds) == 1:
        return int(seeds[0])
    digest = hashlib.sha256(",".join(map(str, seeds)).encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


class FakeHardwareBackend(Backend):
    """A real IBM device's published noise, simulated on Aer (target `qpu`, provider `fake`).

    Targets `fake:<device>[@scale]`. Needs the `ibm` extra. Costs nothing.
    """

    target = "qpu"
    provider = "fake"

    def __init__(self, device: str = "fez", scale: float = 1.0) -> None:
        """Load the device's noise model.

        Args:
            device: Device name such as `fez`, `marrakesh` or `torino`.
            scale: Multiplier on every error rate (T1 and T2 are divided by it); 1 to 10.

        Raises:
            ValueError: `scale` is outside 1..10, or the device is unknown.
            RuntimeError: the `ibm` extra is not installed.
        """
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel

        if not 1.0 <= scale <= 10.0:
            raise ValueError("noise scale must be between 1 and 10")
        self._device = _fake_device(device)
        self.device = f"fake_{device}" + (f"@{scale:g}" if scale != 1.0 else "")
        self.scale = scale
        self.properties = _scaled(self._device.properties(), scale)
        noise = NoiseModel.from_backend_properties(self.properties)
        self._sim = AerSimulator(noise_model=noise)

    def sample(self, circuit: QuantumCircuit, shots: int, seed: int) -> SampleResult:
        """Run one circuit on the noise model (a batch of one)."""
        return self.sample_batch([circuit], shots, [seed])[0]

    def sample_batch(
        self, circuits: Sequence[QuantumCircuit], shots: int, seeds: Sequence[int]
    ) -> list[SampleResult]:
        """Transpile the circuits for the device and run them as one Aer job.

        Args:
            circuits: Circuits to run.
            shots: Shots per circuit.
            seeds: One seed per circuit; all of them feed the job's seed, so each one matters.

        Returns:
            One `SampleResult` per circuit, with `readout`, `two_qubit_gates` and
            `physical_qubits` metadata.

        Raises:
            ValueError: `circuits` and `seeds` differ in length.
        """
        if len(circuits) != len(seeds):
            raise ValueError("one seed per circuit")
        compiled = transpile_best(circuits, self._device, seeds[0])
        # one Aer job for the batch: each run pays a fixed cost to set up the device-wide noise
        # model (~8 s for 156 qubits), however few shots it takes; experiments get their own seeds
        result = self._sim.run(compiled, shots=shots, seed_simulator=_job_seed(seeds)).result()
        out = []
        for i, (c, meta) in enumerate(zip(compiled, device_metadata(compiled), strict=True)):
            counts = result.get_counts(i)
            out.append(
                SampleResult(
                    counts={k.replace(" ", ""): int(v) for k, v in counts.items()},
                    shots=shots,
                    metadata={"readout": readout_errors(c, self.properties), **meta},
                )
            )
        return out

    def calibration_snapshot(self) -> dict[str, Any]:
        """The device name and the noise scale used."""
        return {"backend": self.device, "noise_scale": self.scale}

    @property
    def max_qubits(self) -> int:
        """Qubit count of the simulated device."""
        return int(self._device.num_qubits)
