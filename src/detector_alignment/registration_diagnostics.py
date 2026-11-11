"""Low-overhead, opt-in progress instrumentation for ladder registration.

The heartbeat runs in a daemon thread and reports where the registering thread
is executing. It does not modify the optimization objective or solver settings.
"""

from __future__ import annotations

from contextlib import contextmanager
import sys
from threading import Event, Thread, get_ident
from time import perf_counter
import traceback

from .ladder_registration import LadderRegistrar


def _stack_location(thread_id: int) -> str:
    frame = sys._current_frames().get(thread_id)
    if frame is None:
        return "thread_unavailable"
    stack = traceback.extract_stack(frame)
    # The last few frames provide the active optimizer and the current operation.
    tail = stack[-5:]
    return " <- ".join(
        f"{entry.name}@{entry.filename.rsplit('/', 1)[-1]}:{entry.lineno}"
        for entry in reversed(tail)
    )


@contextmanager
def registration_heartbeat(logger, ladder_name: str, phase: str,
                           interval_seconds: float = 15.0, registrar=None):
    """Log progress while a potentially blocking registration fit is running."""
    stop = Event()
    owner = get_ident()
    started = perf_counter()

    def report():
        while not stop.wait(interval_seconds):
            extra = ""
            if registrar is not None:
                extra = (
                    f" | evaluate_calls={registrar.evaluate_calls}"
                    f" | evaluate_seconds={registrar.evaluate_seconds:.3f}"
                )
            logger.warning(
                "Ladder %s registration | phase=%s | status=running | "
                "elapsed_s=%.1f%s | stack=%s",
                ladder_name, phase, perf_counter() - started, extra,
                _stack_location(owner),
            )

    logger.info(
        "Ladder %s registration | phase=%s | status=started | heartbeat_s=%.1f",
        ladder_name, phase, interval_seconds,
    )
    thread = Thread(target=report, name="ladder-registration-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1.0)
        logger.info(
            "Ladder %s registration | phase=%s | status=finished | elapsed_s=%.3f",
            ladder_name, phase, perf_counter() - started,
        )


class InstrumentedNearestBoxRegistrar(LadderRegistrar):
    """Count and time completed dense nearest-box evaluations without altering math."""

    def __init__(self, options=None):
        super().__init__(options)
        self.evaluate_calls = 0
        self.evaluate_seconds = 0.0

    def evaluate(self, *args, **kwargs):
        started = perf_counter()
        try:
            return super().evaluate(*args, **kwargs)
        finally:
            self.evaluate_calls += 1
            self.evaluate_seconds += perf_counter() - started
