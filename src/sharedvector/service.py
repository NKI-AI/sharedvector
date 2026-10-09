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
"""Standalone producer service owning a shared-memory chunk pool."""

import contextlib
import gc
import logging
import multiprocessing as mp
import threading
import time
import uuid
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Event as EventType
from pathlib import Path

from sharedvector.coordination import PoolPaths, clear_marker, write_marker
from sharedvector.lifecycle import LifecycleGuard
from sharedvector.loaders import Loader
from sharedvector.producer import producer_worker
from sharedvector.stats import PoolStats, collect_stats, stats_capacity_for
from sharedvector.vector import SharedVector, remove_shared_memory

logger = logging.getLogger(__name__)

_CHUNK_OVERHEAD_BYTES = 512
_STATS_OVERHEAD_BYTES = 4096
_BOOTSTRAP_POLL_SECONDS = 0.2
_BOOTSTRAP_STALL_SECONDS = 30.0


def partition(total: int, parts: int) -> list[tuple[int, int]]:
    """Split ``range(total)`` into ``parts`` contiguous, near-equal ``(lo, hi)`` ranges."""
    base, extra = divmod(total, parts)
    ranges: list[tuple[int, int]] = []
    start = 0
    for i in range(parts):
        size = base + (1 if i < extra else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


class ProducerService:
    """Producer that owns a shared-memory chunk pool.

    Removes any stale segment with the same name, creates the :class:`SharedVector`, bootstraps
    ``slot_count`` chunks via the loader with ``num_producers`` worker processes, publishes the
    ready marker, and keeps the workers refreshing idle slots. Cleanup (stop workers, clear marker,
    unlink the segment) is wired to ``atexit``, ``SIGINT`` and ``SIGTERM``.

    The service does not take the owner lock itself; :func:`sharedvector.pool.ensure_pool` and the
    CLI do, and pass the resulting ``generation`` in.

    Typical standalone usage::

        ProducerService(name="run42", loader=..., source_size=..., slot_count=256, chunk_size=16 << 20).run()
    """

    def __init__(
        self,
        name: str,
        loader: Loader,
        source_size: int,
        slot_count: int,
        chunk_size: int,
        num_producers: int = 4,
        sleep_seconds: float = 0.01,
        producer_join_timeout: float = 5.0,
        dtype: str = "float32",
        generation: str | None = None,
        lock_dir: Path | str | None = None,
        log_level: int = logging.INFO,
    ) -> None:
        """Configure the service.

        Args:
            name: Segment name; consumers attach by it.
            loader: Picklable strategy; ``__call__(idx) -> list[Chunk]``.
            source_size: Number of source indices (typically file count).
            slot_count: Resident chunks in shared memory.
            chunk_size: Per-chunk byte budget; every loader array must fit.
            num_producers: Worker processes refreshing idle slots.
            sleep_seconds: Worker cooldown between empty cycles.
            producer_join_timeout: Seconds to wait for clean worker exit before ``terminate``.
            dtype: Segment element dtype, ``"float32"`` or ``"int16"``.
            generation: Owner generation recorded in the ready marker; a fresh token when ``None``.
            lock_dir: Directory of the ready marker; see :class:`~sharedvector.coordination.PoolPaths`.
            log_level: Logging level for the spawned workers.

        Raises:
            ValueError: If ``num_producers < 1``, ``slot_count < num_producers`` or ``source_size < 1``.
        """
        if num_producers < 1:
            raise ValueError("num_producers must be >= 1")
        if slot_count < num_producers:
            raise ValueError("slot_count must be >= num_producers")
        if source_size < 1:
            raise ValueError("source_size must be >= 1")

        self._paths = PoolPaths.for_name(name, lock_dir)
        self._loader = loader
        self._source_size = source_size
        self._slot_count = slot_count
        self._chunk_size = chunk_size
        self._num_producers = num_producers
        self._sleep_seconds = sleep_seconds
        self._join_timeout = producer_join_timeout
        self._dtype = dtype
        self._generation = generation or uuid.uuid4().hex
        self._log_level = log_level
        self._stats_capacity = stats_capacity_for(slot_count)
        self._max_memory_size = (
            slot_count * (chunk_size + _CHUNK_OVERHEAD_BYTES) + 4 * self._stats_capacity + _STATS_OVERHEAD_BYTES
        )

        self._ctx = mp.get_context("spawn")
        self._stop_event: EventType | None = None
        self._processes: list[BaseProcess] = []
        self._vector: SharedVector | None = None
        self._guard = LifecycleGuard(self._cleanup)
        self._started = False

    @property
    def name(self) -> str:
        """Segment name."""
        return self._paths.name

    @property
    def generation(self) -> str:
        """Generation token published in the ready marker."""
        return self._generation

    @property
    def started(self) -> bool:
        """Whether the pool is bootstrapped and workers are running."""
        return self._started

    def start(self) -> None:
        """Allocate the segment, spawn workers, await the bootstrap, publish readiness, arm cleanup."""
        if self._started:
            return
        clear_marker(self._paths)
        remove_shared_memory(self.name)
        self._vector = SharedVector(
            self.name,
            self._chunk_size,
            self._max_memory_size,
            dtype=self._dtype,
            stats_capacity=self._stats_capacity,
        )

        self._stop_event = self._ctx.Event()
        slot_ranges = partition(self._slot_count, self._num_producers)
        logger.info("Spawning %d producers to bootstrap %d slots", self._num_producers, self._slot_count)
        for worker_id, slot_range in enumerate(slot_ranges):
            process = self._ctx.Process(
                target=producer_worker,
                args=(
                    self.name,
                    self._chunk_size,
                    self._max_memory_size,
                    slot_range,
                    self._source_size,
                    self._slot_count,
                    self._loader,
                    self._stop_event,
                    self._sleep_seconds,
                    worker_id,
                    self._dtype,
                    self._log_level,
                ),
                daemon=True,
            )
            process.start()
            self._processes.append(process)

        self._guard.arm()
        self._await_bootstrap()
        write_marker(self._paths, self._generation)
        self._started = True

    def stats(self) -> PoolStats:
        """Return the pool's read statistics.

        Raises:
            RuntimeError: If the service has not been started.
        """
        if self._vector is None:
            raise RuntimeError("ProducerService.stats() requires a started service")
        return collect_stats(self._vector)

    def run(self) -> None:
        """Start the service and block until terminated by signal or exit."""
        self.start()
        threading.Event().wait()

    def stop(self) -> None:
        """Tear everything down (idempotent)."""
        self._guard.trigger()

    def _await_bootstrap(self) -> None:
        """Block until workers have filled ``slot_count`` slots, logging stalls and dead workers."""
        assert self._vector is not None
        last_size = -1
        last_change = time.monotonic()
        while self._vector.size() < self._slot_count:
            if not any(process.is_alive() for process in self._processes):
                raise RuntimeError(f"All producer workers exited before the pool {self.name!r} was filled")
            size = self._vector.size()
            if size != last_size:
                last_size = size
                last_change = time.monotonic()
            elif time.monotonic() - last_change > _BOOTSTRAP_STALL_SECONDS:
                logger.warning(
                    "Bootstrap stalled at %d/%d slots for %.0fs (is the loader yielding chunks?)",
                    size,
                    self._slot_count,
                    _BOOTSTRAP_STALL_SECONDS,
                )
                last_change = time.monotonic()
            time.sleep(_BOOTSTRAP_POLL_SECONDS)
        logger.info("Bootstrap complete: %d slots filled; producers now in replacement mode", self._slot_count)

    def _cleanup(self) -> None:
        logger.info("Stopping producer service for pool %r", self.name)
        clear_marker(self._paths)
        if self._stop_event is not None:
            self._stop_event.set()
        for process in self._processes:
            process.join(timeout=self._join_timeout)
        for process in self._processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        self._processes.clear()
        self._vector = None
        # Drop the last reference to the stop event now so its POSIX semaphores are unlinked by their
        # finalizer. Exiting through a re-raised signal skips finalizers, which otherwise leaves them
        # to the multiprocessing resource tracker and its "leaked semaphore objects" warning.
        self._stop_event = None
        gc.collect()
        try:
            remove_shared_memory(self.name)
        except Exception:
            logger.exception("Failed to remove segment %r", self.name)
        self._started = False
        logger.info("Producer service for pool %r stopped", self.name)

    def __enter__(self) -> "ProducerService":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()

    def __del__(self) -> None:
        # Best effort during interpreter teardown.
        with contextlib.suppress(Exception):
            self._guard.trigger()


__all__ = ["ProducerService", "partition"]
