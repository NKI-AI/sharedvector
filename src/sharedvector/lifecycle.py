# Copyright 2026 AI for Oncology. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Process-lifecycle helpers: idempotent cleanup on exit/signals and parent-death handling."""

import atexit
import contextlib
import ctypes
import logging
import os
import signal
import threading
from collections.abc import Callable
from types import FrameType
from typing import Any

logger = logging.getLogger(__name__)

_PR_SET_PDEATHSIG = 1

SignalHandler = Callable[[int, FrameType | None], Any] | int | None


class LifecycleGuard:
    """Idempotent cleanup runner wired to ``atexit`` and ``SIGINT``/``SIGTERM``.

    The cleanup is invoked at most once and only in the process that armed the guard, so handlers
    inherited by forked children (e.g. DataLoader workers) are no-ops there. Prior signal handlers
    are chained so frameworks that install their own keep working.
    """

    _SIGNALS = (signal.SIGINT, signal.SIGTERM)

    def __init__(self, cleanup: Callable[[], None]) -> None:
        """Wrap ``cleanup`` without arming any handlers yet.

        Args:
            cleanup: Callable to run once on shutdown.
        """
        self._cleanup = cleanup
        self._lock = threading.Lock()
        self._cleaned = False
        self._cleaning = False
        self._armed = False
        self._owner_pid: int | None = None
        self._previous: dict[int, SignalHandler] = {}

    @property
    def armed(self) -> bool:
        """Whether handlers are currently installed."""
        return self._armed

    def _run_cleanup(self) -> bool:
        """Run the cleanup if this is the first call in the owning process.

        Returns:
            ``True`` if the cleanup ran in this call, ``False`` if it already ran or is running.
        """
        if self._owner_pid is not None and os.getpid() != self._owner_pid:
            return False
        with self._lock:
            if self._cleaned:
                return False
            self._cleaned = True
            self._cleaning = True
        try:
            self._cleanup()
        finally:
            self._cleaning = False
        return True

    def _signal_handler(self, signum: int, frame: FrameType | None) -> None:
        if self._cleaning:
            # A second signal (e.g. parent-death signal plus watchdog) arrived while the cleanup is
            # still running; absorb it so the default action cannot kill the process half-way.
            logger.info("Signal %d received during cleanup in pid %d; ignoring", signum, os.getpid())
            return
        logger.info("Received signal %d in pid %d; running cleanup", signum, os.getpid())
        self._run_cleanup()
        previous = self._previous.get(signum, signal.SIG_DFL)
        if callable(previous):
            previous(signum, frame)
            return
        signal.signal(signum, previous)
        signal.raise_signal(signum)

    def arm(self) -> None:
        """Install the ``atexit`` hook and signal handlers (idempotent).

        Signals that are currently ignored stay ignored, following the Unix convention that a parent
        (a shell running a background job, or a supervisor that opted out of ``SIGINT``) has made
        that decision for this process.
        """
        if self._armed:
            return
        self._armed = True
        self._owner_pid = os.getpid()
        atexit.register(self._run_cleanup)
        for sig in self._SIGNALS:
            previous = signal.getsignal(sig)
            if previous == signal.SIG_IGN:
                continue
            self._previous[sig] = previous
            signal.signal(sig, self._signal_handler)

    def disarm(self) -> None:
        """Restore the previous handlers and unregister the ``atexit`` hook (idempotent)."""
        if not self._armed:
            return
        self._armed = False
        atexit.unregister(self._run_cleanup)
        for sig, previous in self._previous.items():
            with contextlib.suppress(TypeError, ValueError):
                signal.signal(sig, previous)
        self._previous.clear()

    def trigger(self) -> None:
        """Run the cleanup now (idempotent) and detach the handlers."""
        self._run_cleanup()
        self.disarm()

    @property
    def cleaned(self) -> bool:
        """Whether the cleanup has run (or is running)."""
        return self._cleaned


def install_parent_death_signal(sig: int = signal.SIGKILL) -> bool:
    """Ask the kernel to deliver ``sig`` to this process when its parent thread exits (Linux only).

    Prevents orphaned helper processes from spinning forever when the owner dies without running
    its cleanup. The check against ``getppid() == 1`` closes the race where the parent died between
    ``fork`` and ``prctl``.

    Args:
        sig: Signal to receive on parent death.

    Returns:
        ``True`` if the request was installed, ``False`` when ``prctl`` is unavailable.
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, int(sig))
    except (OSError, AttributeError):
        return False
    if os.getppid() == 1:
        os.kill(os.getpid(), sig)
    return True


class ParentWatchdog(threading.Thread):
    """Daemon thread invoking ``on_orphan`` once the parent process has gone away.

    Backstop for :func:`install_parent_death_signal`, which tracks the parent *thread* rather than
    the parent process and is Linux specific.
    """

    def __init__(self, on_orphan: Callable[[], None], interval: float = 0.5) -> None:
        """Create the watchdog without starting it.

        Args:
            on_orphan: Called once from the watchdog thread when the parent is gone.
            interval: Seconds between parent checks.
        """
        super().__init__(name="sharedvector-parent-watchdog", daemon=True)
        self._on_orphan = on_orphan
        self._interval = interval
        self._parent_pid = os.getppid()
        self._stop = threading.Event()

    def stop(self) -> None:
        """Stop polling."""
        self._stop.set()

    def run(self) -> None:
        """Poll the parent pid until it changes, then fire ``on_orphan``."""
        while not self._stop.wait(self._interval):
            if os.getppid() != self._parent_pid:
                logger.warning("Parent process %d is gone; running orphan handler", self._parent_pid)
                self._on_orphan()
                return


__all__ = ["LifecycleGuard", "ParentWatchdog", "install_parent_death_signal"]
