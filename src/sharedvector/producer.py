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
"""Producer worker: fills the pool during bootstrap, then keeps refreshing idle slots."""

import logging
import os
import random
import time
import warnings
from multiprocessing.synchronize import Event as EventType

from sharedvector.lifecycle import install_parent_death_signal
from sharedvector.loaders import Loader
from sharedvector.vector import OutOfMemoryError, SharedVector

logger = logging.getLogger(__name__)


def _stopped(stop_event: EventType) -> bool:
    """Return ``True`` when the worker should exit: told to stop, or orphaned."""
    return stop_event.is_set() or os.getppid() == 1


def _next_idle_slot(vec: SharedVector, slot_lo: int, slot_hi: int, start: int) -> int | None:
    """Return the next slot in ``[slot_lo, slot_hi)`` with ``ref_count == 0``, wrapping from ``start``."""
    span = slot_hi - slot_lo
    for offset in range(span):
        slot = slot_lo + (start - slot_lo + offset) % span
        if vec.get_chunk_ref_count(slot) == 0:
            return slot
    return None


def producer_worker(
    name: str,
    chunk_size: int,
    max_memory_size: int,
    slot_range: tuple[int, int],
    source_size: int,
    slot_count: int,
    loader: Loader,
    stop_event: EventType,
    sleep_seconds: float,
    worker_id: int = 0,
    dtype: str = "float32",
    log_level: int = logging.INFO,
) -> None:
    """Run one producer process until ``stop_event`` is set.

    The worker first cooperates on the initial fill (``append``) and then refreshes idle slots
    within ``slot_range`` (``replace``). Both phases draw a uniformly random source index per cycle
    so the data order is unpredictable.

    Args:
        name: Segment name.
        chunk_size: Per-chunk byte budget.
        max_memory_size: Segment size; ignored when the segment already exists.
        slot_range: ``(lo, hi)`` slots this worker refreshes.
        source_size: Number of source indices the loader accepts.
        slot_count: Total slots in the pool.
        loader: Strategy producing chunks for a source index.
        stop_event: Set by the service to request shutdown.
        sleep_seconds: Cooldown between empty cycles.
        worker_id: Identifier used in log messages.
        dtype: Segment element dtype.
        log_level: Logging level configured in the spawned process.
    """
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    install_parent_death_signal()
    vec = SharedVector(name, chunk_size, max_memory_size, dtype=dtype)
    rng = random.Random()
    try:
        _bootstrap_append(vec, source_size, slot_count, loader, stop_event, sleep_seconds, rng, worker_id)
        _refresh_loop(vec, slot_range, source_size, loader, stop_event, sleep_seconds, rng, worker_id)
    finally:
        del vec


def _bootstrap_append(
    vec: SharedVector,
    source_size: int,
    slot_count: int,
    loader: Loader,
    stop_event: EventType,
    sleep_seconds: float,
    rng: random.Random,
    worker_id: int,
) -> None:
    """Append random sources until the pool reaches ``slot_count`` chunks."""
    logger.info("worker %d: appending toward %d slots", worker_id, slot_count)
    while not _stopped(stop_event) and vec.size() < slot_count:
        src = rng.randrange(source_size)
        chunks = loader(src)
        if not chunks:
            warnings.warn(f"Loader returned no chunks for source_index={src}", RuntimeWarning, stacklevel=2)
            continue
        for chunk in chunks:
            if _stopped(stop_event) or vec.size() >= slot_count:
                break
            try:
                vec.append(chunk.data, chunk.meta)
            except (OutOfMemoryError, RuntimeError):
                time.sleep(sleep_seconds)
                break
    # Barrier: only refresh once every slot exists, so idle-slot scans are valid.
    while not _stopped(stop_event) and vec.size() < slot_count:
        time.sleep(sleep_seconds)
    logger.info("worker %d: bootstrap done (size=%d), switching to replacement mode", worker_id, vec.size())


def _refresh_loop(
    vec: SharedVector,
    slot_range: tuple[int, int],
    source_size: int,
    loader: Loader,
    stop_event: EventType,
    sleep_seconds: float,
    rng: random.Random,
    worker_id: int,
) -> None:
    """Replace idle slots in ``slot_range`` with random sources, indefinitely."""
    slot_lo, slot_hi = slot_range
    slot_cursor = slot_lo
    while not _stopped(stop_event):
        src = rng.randrange(source_size)
        chunks = loader(src)
        if not chunks:
            warnings.warn(f"Loader returned no chunks for source_index={src}", RuntimeWarning, stacklevel=2)
            continue

        placed = 0
        for chunk in chunks:
            if _stopped(stop_event):
                break
            slot = _next_idle_slot(vec, slot_lo, slot_hi, slot_cursor)
            if slot is None:
                break
            try:
                vec.replace(slot, chunk.data, chunk.meta)
                placed += 1
            except RuntimeError:
                pass
            slot_cursor = slot + 1 if slot + 1 < slot_hi else slot_lo

        if placed == 0:
            time.sleep(sleep_seconds)


__all__ = ["producer_worker"]
