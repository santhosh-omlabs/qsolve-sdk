"""Executor protocol: how a sandboxed workflow reaches a quantum backend without holding any
credentials.

Inside the platform sandbox the workflow process has no network and no provider secrets. It talks
to the platform's executor over a local socket (QSOLVE_EXECUTOR_ADDR, e.g. "unix:/run/qsolve.sock"
or "tcp:127.0.0.1:7777"). The executor owns credentials, enforces credit holds, shot/qubit/device
caps, and writes the provenance. This module defines the wire format and both ends.

Wire format: one JSON object per line.
  request : {"op": "sample", "qpy": <base64 QPY circuit>, "shots": int, "seed": int}
            {"op": "sample_batch", "qpy": <base64 QPY, up to MAX_BATCH circuits>, "shots": int,
             "seeds": [int, ...]}       -> {"ok": true, "results": [<sample response>, ...]}
            {"op": "info"}
  response: {"ok": true, ...} or {"ok": false, "error": str}
"""

from __future__ import annotations

import base64
import hmac
import io
import json
import os
import socket
import socketserver
import threading
from collections.abc import Sequence
from typing import Any

from qiskit import QuantumCircuit, qpy

from .base import MAX_BATCH, Backend, SampleResult

MAX_LINE_BYTES = 8 * 1024 * 1024


def _encode(circuit: QuantumCircuit | Sequence[QuantumCircuit]) -> str:
    buf = io.BytesIO()
    qpy.dump(circuit, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_many(blob: str) -> list[QuantumCircuit]:
    circuits = qpy.load(io.BytesIO(base64.b64decode(blob, validate=True)))
    if not 1 <= len(circuits) <= MAX_BATCH or not all(
        isinstance(c, QuantumCircuit) for c in circuits
    ):
        raise ValueError(f"expected 1..{MAX_BATCH} circuits")
    return list(circuits)


def _decode(blob: str) -> QuantumCircuit:
    circuits = _decode_many(blob)
    if len(circuits) != 1:
        raise ValueError("expected exactly one circuit")
    return circuits[0]


def _result(r: dict[str, Any]) -> SampleResult:
    return SampleResult(
        counts={k: int(v) for k, v in r["counts"].items()},
        shots=int(r["shots"]),
        exec_time_s=float(r.get("exec_time_s", 0.0)),
        metadata=r.get("metadata", {}),
    )


def _wire(res: SampleResult) -> dict[str, Any]:
    return {
        "counts": res.counts,
        "shots": res.shots,
        "exec_time_s": res.exec_time_s,
        "metadata": res.metadata,
    }


def _connect(addr: str) -> socket.socket:
    kind, _, rest = addr.partition(":")
    if kind == "unix":
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)  # type: ignore[attr-defined,unused-ignore]
        s.connect(rest)
    elif kind == "tcp":
        host, _, port = rest.rpartition(":")
        s = socket.create_connection((host, int(port)))
    else:
        raise ValueError(f"bad executor address {addr!r}; use unix:<path> or tcp:<host>:<port>")
    return s


class RemoteBackend(Backend):
    """Backend that forwards sampling to the platform executor."""

    provider = "executor"

    def __init__(self, addr: str | None = None, token: str | None = None) -> None:
        """Connect to the platform executor and read the device it serves.

        Args:
            addr: `unix:<path>` or `tcp:<host>:<port>`; defaults to `QSOLVE_EXECUTOR_ADDR`.
            token: Shared secret; defaults to `QSOLVE_EXECUTOR_TOKEN` (may be empty).

        Raises:
            RuntimeError: no address, or the executor refused the request.
        """
        self._token = token or os.environ.get("QSOLVE_EXECUTOR_TOKEN", "")
        self._addr = addr or os.environ.get("QSOLVE_EXECUTOR_ADDR", "")
        if not self._addr:
            raise RuntimeError("QSOLVE_EXECUTOR_ADDR is not set")
        info = self._call({"op": "info"})
        self.target = str(info["target"])
        self.provider = str(info["provider"])
        self.device = str(info["device"])
        self._max_qubits = int(info["max_qubits"])
        self._calibration: dict[str, Any] = info.get("calibration", {})

    def _call(self, req: dict[str, Any]) -> dict[str, Any]:
        if self._token:
            req = {**req, "token": self._token}
        with _connect(self._addr) as s, s.makefile("rwb") as f:
            f.write(json.dumps(req).encode() + b"\n")
            f.flush()
            line = f.readline(MAX_LINE_BYTES + 1)
        if not line or len(line) > MAX_LINE_BYTES:
            raise RuntimeError("executor closed the connection or sent an oversized reply")
        resp: dict[str, Any] = json.loads(line)
        if not resp.get("ok"):
            raise RuntimeError(f"executor error: {resp.get('error', 'unknown')}")
        return resp

    def sample(self, circuit: QuantumCircuit, shots: int, seed: int) -> SampleResult:
        """Send one circuit (QPY, base64) to the executor and return its result."""
        r = self._call({"op": "sample", "qpy": _encode(circuit), "shots": shots, "seed": seed})
        return _result(r)

    def sample_batch(
        self, circuits: Sequence[QuantumCircuit], shots: int, seeds: Sequence[int]
    ) -> list[SampleResult]:
        """Send circuits to the executor in chunks of `MAX_BATCH`.

        Raises:
            ValueError: `circuits` and `seeds` differ in length.
            RuntimeError: the executor failed or returned the wrong number of results.
        """
        if len(circuits) != len(seeds):
            raise ValueError("one seed per circuit")
        out: list[SampleResult] = []
        for i in range(0, len(circuits), MAX_BATCH):
            chunk = list(circuits[i : i + MAX_BATCH])
            r = self._call(
                {"op": "sample_batch", "qpy": _encode(chunk), "shots": shots,
                 "seeds": [int(s) for s in seeds[i : i + MAX_BATCH]]}
            )  # fmt: skip
            if len(r["results"]) != len(chunk):
                raise RuntimeError("executor returned the wrong number of results")
            out += [_result(x) for x in r["results"]]
        return out

    def calibration_snapshot(self) -> dict[str, Any]:
        """The calibration the executor reported when this backend connected."""
        return self._calibration

    @property
    def max_qubits(self) -> int:
        """The qubit cap the executor reported."""
        return self._max_qubits


class _Handler(socketserver.StreamRequestHandler):
    server: Any

    def handle(self) -> None:
        line = self.rfile.readline(MAX_LINE_BYTES + 1)
        try:
            if len(line) > MAX_LINE_BYTES:
                raise ValueError("request too large")
            resp = self.server.dispatch(json.loads(line))
        except Exception as e:
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self.wfile.write(json.dumps(resp).encode() + b"\n")


class _ServerMixin:
    allow_reuse_address = True
    daemon_threads = True
    backend: Backend
    limits: dict[str, int]

    token: str | None = None

    def dispatch(self, req: dict[str, Any]) -> dict[str, Any]:
        if self.token is not None and not hmac.compare_digest(
            str(req.get("token", "")), self.token
        ):
            raise PermissionError("bad executor token")
        op = req.get("op")
        b = self.backend
        if op == "info":
            return {
                "ok": True,
                "target": b.target,
                "provider": b.provider,
                "device": b.device,
                "max_qubits": min(b.max_qubits, self.limits["max_qubits"]),
                "calibration": b.calibration_snapshot(),
            }
        if op in ("sample", "sample_batch"):
            shots = int(req["shots"])
            if not 1 <= shots <= self.limits["max_shots"]:
                raise ValueError(f"shots must be in 1..{self.limits['max_shots']}")
            circuits = _decode(req["qpy"]) if op == "sample" else _decode_many(req["qpy"])
            batch = circuits if isinstance(circuits, list) else [circuits]
            cap = min(b.max_qubits, self.limits["max_qubits"])
            if any(qc.num_qubits > cap for qc in batch):
                raise ValueError("circuit exceeds the allowed qubit count")
            if op == "sample":
                return {"ok": True, **_wire(b.sample(batch[0], shots, int(req["seed"])))}
            seeds = [int(s) for s in req["seeds"]]
            if len(seeds) != len(batch):
                raise ValueError("one seed per circuit")
            return {"ok": True, "results": [_wire(r) for r in b.sample_batch(batch, shots, seeds)]}
        raise ValueError(f"unknown op {op!r}")


class _TcpServer(_ServerMixin, socketserver.ThreadingTCPServer):
    def __init__(self, addr: tuple[str, int], backend: Backend, limits: dict[str, int]) -> None:
        socketserver.ThreadingTCPServer.__init__(self, addr, _Handler)
        self.backend, self.limits = backend, limits


_Server = _TcpServer  # backwards-compatible name


if hasattr(socketserver, "ThreadingUnixStreamServer"):

    class _UnixServer(_ServerMixin, socketserver.ThreadingUnixStreamServer):  # type: ignore[name-defined,misc,unused-ignore]
        def __init__(self, path: str, backend: Backend, limits: dict[str, int]) -> None:
            socketserver.ThreadingUnixStreamServer.__init__(self, path, _Handler)  # type: ignore[attr-defined,unused-ignore]
            self.backend, self.limits = backend, limits


def serve_executor(
    backend: Backend,
    host: str = "127.0.0.1",
    port: int = 0,
    *,
    address: str | None = None,
    max_shots: int = 100_000,
    max_qubits: int = 32,
    token: str | None = None,
) -> tuple[Any, str]:
    """Reference executor: serves `backend` and returns (server, address).

    `address` is "unix:<path>" (production: mount the socket into a network-less sandbox) or
    "tcp:<host>:<port>". The platform executor wraps this with credit metering and per-run quotas.
    Credentials stay in `backend`, which lives only in this trusted process.
    """
    limits = {"max_shots": max_shots, "max_qubits": max_qubits}
    return _start(backend, host, port, address, limits, token)


def _start(
    backend: Backend,
    host: str,
    port: int,
    address: str | None,
    limits: dict[str, int],
    token: str | None,
) -> tuple[Any, str]:
    if address and address.startswith("unix:"):
        path = address.removeprefix("unix:")
        server: Any = _UnixServer(path, backend, limits)
        server.token = token
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, address
    if address and address.startswith("tcp:"):
        host, _, p = address.removeprefix("tcp:").rpartition(":")
        port = int(p)
    server = _TcpServer((host, port), backend, limits)
    server.token = token
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"tcp:{host}:{server.server_address[1]}"
