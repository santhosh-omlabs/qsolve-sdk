"""Structured run events. Solvers and workflows call `emit(...)`; the runner (or the platform)
installs a sink that streams them to the UI/log store. With no sink installed `emit` is free."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

Sink = Callable[[dict[str, Any]], None]
_sink: ContextVar[Sink | None] = ContextVar("qsolve_event_sink", default=None)


def emit(type: str, **data: Any) -> None:
    """Send a progress event to the installed sink; free and silent when there is none.

    Args:
        type: Event kind, for example `stage` or `train_step`.
        **data: JSON-compatible fields; `t` (a timestamp) is added.

    Never raises: a failing sink is ignored.
    """
    sink = _sink.get()
    if sink is None:
        return
    with contextlib.suppress(Exception):  # observability must never break a run
        sink({"type": type, "t": time.time(), **data})


@contextmanager
def capture(sink: Sink | None) -> Iterator[None]:
    """Install `sink` as the event receiver inside the `with` block (`None` turns events off)."""
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)
