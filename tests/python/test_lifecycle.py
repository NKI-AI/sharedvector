import os
import signal
import threading

from sharedvector.lifecycle import LifecycleGuard, ParentWatchdog, install_parent_death_signal


def test_trigger_runs_cleanup_once() -> None:
    calls: list[int] = []
    guard = LifecycleGuard(lambda: calls.append(1))
    guard.arm()

    guard.trigger()
    guard.trigger()

    assert calls == [1]
    assert not guard.armed


def test_signal_runs_cleanup_and_chains_previous_handler() -> None:
    calls: list[str] = []
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda signum, frame: calls.append("previous"))
    guard = LifecycleGuard(lambda: calls.append("cleanup"))
    try:
        guard.arm()

        signal.raise_signal(signal.SIGTERM)

        assert calls == ["cleanup", "previous"]
    finally:
        guard.disarm()
        signal.signal(signal.SIGTERM, previous)


def test_disarm_restores_previous_handlers() -> None:
    previous = signal.getsignal(signal.SIGINT)
    guard = LifecycleGuard(lambda: None)

    guard.arm()
    assert signal.getsignal(signal.SIGINT) != previous
    guard.disarm()

    assert signal.getsignal(signal.SIGINT) == previous


def test_cleanup_is_noop_in_forked_child() -> None:
    calls: list[int] = []
    guard = LifecycleGuard(lambda: calls.append(1))
    guard.arm()
    try:
        pid = os.fork()
        if pid == 0:  # pragma: no cover - child
            guard.trigger()
            os._exit(0 if not calls else 1)
        _, status = os.waitpid(pid, 0)

        assert os.waitstatus_to_exitcode(status) == 0
        assert calls == []
    finally:
        guard.disarm()


def test_install_parent_death_signal_succeeds_on_linux() -> None:
    assert install_parent_death_signal(signal.SIGTERM) is True


def test_parent_watchdog_fires_when_parent_changes(monkeypatch) -> None:
    fired = threading.Event()
    watchdog = ParentWatchdog(on_orphan=fired.set, interval=0.02)
    monkeypatch.setattr(os, "getppid", lambda: -1)

    watchdog.start()

    assert fired.wait(2.0)
    watchdog.stop()


def test_signal_during_cleanup_does_not_interrupt_it() -> None:
    events: list[str] = []
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda signum, frame: events.append("previous"))

    def cleanup() -> None:
        events.append("start")
        signal.raise_signal(signal.SIGTERM)  # second signal while cleaning up
        events.append("end")

    guard = LifecycleGuard(cleanup)
    try:
        guard.arm()

        signal.raise_signal(signal.SIGTERM)

        assert events == ["start", "end", "previous"]
    finally:
        guard.disarm()
        signal.signal(signal.SIGTERM, previous)
