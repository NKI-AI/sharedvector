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
"""Obtain a ready pool by name: attach to an external producer, or spawn one as a detached process.

Every process on a node may call :func:`ensure_pool` with the same arguments (DDP ranks, DataLoader
workers). Exactly one becomes the owner through a node-local file lock and spawns the producer in a
supervisor child; all others wait for the owner's ready marker and attach. Ownership never crosses
pickling or forking, so workers can only ever attach.
"""

import logging
import multiprocessing as mp
import os
import signal
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Literal

from sharedvector.coordination import (
    OwnerLock,
    PoolBusyError,
    PoolPaths,
    PoolSpawnError,
    attach,
    clear_marker,
    owner_alive,
    read_generation,
    read_marker,
    wait_until_ready,
)
from sharedvector.lifecycle import LifecycleGuard, ParentWatchdog, install_parent_death_signal
from sharedvector.loaders import Loader
from sharedvector.stats import PoolStats, collect_stats
from sharedvector.vector import SharedVector, remove_shared_memory

logger = logging.getLogger(__name__)

Mode = Literal["attach", "spawn", "auto"]

_REGISTRY: dict[str, "PoolHandle"] = {}
_REGISTRY_LOCK = threading.Lock()


def _demote_registry_after_fork() -> None:
    """Forked children inherit handles but never ownership."""
    for handle in _REGISTRY.values():
        handle._demote()


os.register_at_fork(after_in_child=_demote_registry_after_fork)


@dataclass(frozen=True)
class ProducerSpec:
    """Everything needed to spawn a producer for a pool.

    Attributes:
        loader: Picklable loader (shipped to the supervisor via the spawn context).
        chunk_size: Per-chunk byte budget.
        source_size: Number of source indices; ``None`` uses ``len(loader)``.
        dtype: Segment element dtype.
        num_producers: Worker processes.
        sleep_seconds: Worker cooldown between empty cycles.
        producer_join_timeout: Seconds before workers are terminated on shutdown.
    """

    loader: Loader
    chunk_size: int
    source_size: int | None = None
    dtype: Literal["float32", "int16"] = "float32"
    num_producers: int = 4
    sleep_seconds: float = 0.01
    producer_join_timeout: float = 5.0

    def resolve_source_size(self) -> int:
        """Return the configured source size or ``len(loader)``."""
        if self.source_size is not None:
            return self.source_size
        return len(self.loader)


class PoolHandle:
    """Reference to a ready pool. Only the process that spawned the producer owns it."""

    def __init__(
        self,
        paths: PoolPaths,
        slot_count: int,
        generation: str | None,
        lock: OwnerLock | None = None,
        supervisor: BaseProcess | None = None,
        join_timeout: float = 15.0,
    ) -> None:
        """Create a handle; use :func:`ensure_pool` instead of calling this directly.

        Args:
            paths: Pool paths.
            slot_count: Number of chunks in the pool.
            generation: Generation the pool was observed under.
            lock: Owner lock, only for the owning process.
            supervisor: Producer supervisor process, only for the owning process.
            join_timeout: Seconds to wait for the supervisor to exit before killing it.
        """
        self._paths = paths
        self._slot_count = slot_count
        self._generation = generation
        self._lock = lock
        self._supervisor = supervisor
        self._join_timeout = join_timeout
        self._owner_pid = os.getpid() if lock is not None else None
        self._closed = False
        self._vector: SharedVector | None = None
        self._guard: LifecycleGuard | None = None
        if self.is_owner:
            self._guard = LifecycleGuard(self._teardown)
            self._guard.arm()

    @property
    def name(self) -> str:
        """Pool name."""
        return self._paths.name

    @property
    def slot_count(self) -> int:
        """Number of chunks in the pool."""
        return self._slot_count

    @property
    def lock_dir(self) -> Path:
        """Directory holding the lock and ready files."""
        return self._paths.lock_dir

    @property
    def generation(self) -> str | None:
        """Generation token the pool was observed under."""
        return self._generation

    @property
    def is_owner(self) -> bool:
        """Whether this process owns the producer (and therefore the teardown)."""
        return self._lock is not None and self._owner_pid == os.getpid()

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has run."""
        return self._closed

    def attach(self) -> SharedVector:
        """Open the segment without creating it."""
        return attach(self._paths)

    def stats(self) -> PoolStats:
        """Return the pool's read statistics (``handle.stats().as_dict()`` suits ``wandb.log``)."""
        if self._vector is None:
            self._vector = self.attach()
        return collect_stats(self._vector)

    def close(self) -> None:
        """Owner: stop the producer and remove the segment. Follower: forget the handle. Idempotent."""
        if self._closed:
            return
        if self.is_owner and self._guard is not None:
            self._guard.trigger()
        else:
            self._forget()

    def _forget(self) -> None:
        self._vector = None
        with _REGISTRY_LOCK:
            if _REGISTRY.get(self.name) is self:
                del _REGISTRY[self.name]
        self._closed = True

    def _demote(self) -> None:
        """Drop ownership references in a forked child without releasing the inherited lock."""
        self._lock = None
        self._supervisor = None
        self._guard = None
        self._owner_pid = None
        self._vector = None

    def _teardown(self) -> None:
        if os.getpid() != self._owner_pid:
            return
        supervisor = self._supervisor
        if supervisor is not None and supervisor.is_alive():
            supervisor.terminate()
            supervisor.join(timeout=self._join_timeout)
            if supervisor.is_alive():
                logger.warning("Producer supervisor for %r did not exit; killing it", self.name)
                supervisor.kill()
                supervisor.join()
        clear_marker(self._paths)
        remove_shared_memory(self.name)
        if self._lock is not None:
            self._lock.release()
        self._forget()

    def __enter__(self) -> "PoolHandle":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        return {"paths": self._paths, "slot_count": self._slot_count, "generation": self._generation}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(state["paths"], state["slot_count"], state["generation"])  # type: ignore[misc]
        _register(self, replace_existing=False)

    def __repr__(self) -> str:
        role = "owner" if self.is_owner else "follower"
        return f"PoolHandle(name={self.name!r}, slot_count={self._slot_count}, {role})"


def _register(handle: PoolHandle, replace_existing: bool = True) -> PoolHandle:
    with _REGISTRY_LOCK:
        existing = _REGISTRY.get(handle.name)
        if existing is not None and not existing.closed and not replace_existing:
            return existing
        _REGISTRY[handle.name] = handle
    return handle


def _supervisor_main(
    name: str,
    slot_count: int,
    spec: ProducerSpec,
    generation: str,
    lock_dir: str,
    log_level: int,
) -> None:
    """Entry point of the detached supervisor process running the producer service."""
    from sharedvector.service import ProducerService

    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # SIGTERM (not SIGKILL) on parent death so the service's own cleanup still runs; the owner drives
    # shutdown through SIGTERM too, so SIGINT from a shared terminal must not race it.
    install_parent_death_signal(signal.SIGTERM)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    ParentWatchdog(on_orphan=lambda: os.kill(os.getpid(), signal.SIGTERM)).start()

    ProducerService(
        name=name,
        loader=spec.loader,
        source_size=spec.resolve_source_size(),
        slot_count=slot_count,
        chunk_size=spec.chunk_size,
        num_producers=spec.num_producers,
        sleep_seconds=spec.sleep_seconds,
        producer_join_timeout=spec.producer_join_timeout,
        dtype=spec.dtype,
        generation=generation,
        lock_dir=lock_dir,
        log_level=log_level,
    ).run()


def _follow(paths: PoolPaths, slot_count: int, deadline: float, poll_interval: float) -> PoolHandle:
    state = wait_until_ready(paths, slot_count, deadline, poll_interval, strict=True)
    logger.info("Attached to pool %r (generation %s)", paths.name, state.generation)
    return PoolHandle(paths, slot_count, state.generation)


def _spawn_as_owner(
    paths: PoolPaths,
    slot_count: int,
    spec: ProducerSpec,
    lock: OwnerLock,
    deadline: float,
    poll_interval: float,
) -> PoolHandle:
    clear_marker(paths)
    remove_shared_memory(paths.name)
    ctx = mp.get_context("spawn")
    supervisor = ctx.Process(
        target=_supervisor_main,
        name=f"sharedvector-supervisor[{paths.name}]",
        args=(
            paths.name,
            slot_count,
            spec,
            lock.generation,
            str(paths.lock_dir),
            logging.getLogger().getEffectiveLevel(),
        ),
        daemon=False,  # a daemonic process cannot spawn the producer workers
    )
    supervisor.start()
    handle = PoolHandle(paths, slot_count, lock.generation, lock=lock, supervisor=supervisor)
    logger.info("Spawned producer supervisor pid %s for pool %r", supervisor.pid, paths.name)

    def abort() -> str | None:
        return None if supervisor.is_alive() else f"supervisor exited with code {supervisor.exitcode}"

    try:
        wait_until_ready(paths, slot_count, deadline, poll_interval, strict=True, abort=abort)
    except (TimeoutError, PoolSpawnError):
        handle.close()
        raise
    return handle


def _live_pool_exists(paths: PoolPaths) -> bool:
    if not owner_alive(paths):
        return False
    generation = read_generation(paths)
    return generation is not None and read_marker(paths) == generation


def ensure_pool(
    name: str,
    slot_count: int,
    *,
    producer: ProducerSpec | None = None,
    mode: Mode = "auto",
    ready_timeout: float = 600.0,
    poll_interval: float = 0.1,
    lock_dir: Path | str | None = None,
) -> PoolHandle:
    """Return a handle on a ready pool, spawning its producer if needed.

    Modes:

    * ``"attach"``: the pool must be produced elsewhere (e.g. the ``sharedvector-producer`` CLI in
      its own job step); never spawns.
    * ``"spawn"``: run a node-local election; the winner spawns the producer in a detached
      supervisor process and owns its lifetime, losers attach to the winner's pool.
    * ``"auto"``: like ``"spawn"``, but attaches without an election when a live, ready pool exists.

    Calling this repeatedly in one process returns the same handle. Forked or spawned children
    (DataLoader workers) never become owners.

    Args:
        name: Pool name.
        slot_count: Number of chunks the producer keeps resident.
        producer: Producer configuration; required unless ``mode == "attach"``.
        mode: See above.
        ready_timeout: Seconds to wait for the pool to become ready (bootstrap included).
        poll_interval: Sleep between readiness polls.
        lock_dir: Directory for the lock and ready files; defaults as in
            :class:`~sharedvector.coordination.PoolPaths`.

    Returns:
        A :class:`PoolHandle`; ``handle.is_owner`` tells whether this process spawned the producer.

    Raises:
        ValueError: On a missing producer spec or a ``slot_count`` conflicting with an existing handle.
        TimeoutError: If the pool does not become ready within ``ready_timeout``.
        PoolSpawnError: If the spawned producer dies before the pool is ready.
    """
    with _REGISTRY_LOCK:
        existing = _REGISTRY.get(name)
    if existing is not None and not existing.closed:
        if existing.slot_count != slot_count:
            raise ValueError(
                f"Pool {name!r} already open with slot_count={existing.slot_count}, requested {slot_count}"
            )
        return existing

    paths = PoolPaths.for_name(name, lock_dir)
    deadline = time.monotonic() + ready_timeout

    if mode == "attach":
        return _register(_follow(paths, slot_count, deadline, poll_interval))
    if producer is None:
        raise ValueError(f"mode={mode!r} requires a ProducerSpec")

    if mode == "auto" and _live_pool_exists(paths):
        logger.info("Pool %r has a live owner; attaching", name)
        return _register(_follow(paths, slot_count, deadline, poll_interval))

    lock = OwnerLock.try_acquire(paths)
    if lock is None:
        logger.info("Lost the election for pool %r; attaching to the winner's pool", name)
        return _register(_follow(paths, slot_count, deadline, poll_interval))

    try:
        return _register(_spawn_as_owner(paths, slot_count, producer, lock, deadline, poll_interval))
    except BaseException:
        lock.release()
        raise


@contextmanager
def pool(name: str, slot_count: int, **kwargs: Any) -> Generator[PoolHandle]:
    """Context-manager form of :func:`ensure_pool` that closes the handle on exit."""
    handle = ensure_pool(name, slot_count, **kwargs)
    try:
        yield handle
    finally:
        handle.close()


__all__ = ["Mode", "PoolBusyError", "PoolHandle", "PoolSpawnError", "ProducerSpec", "ensure_pool", "pool"]
