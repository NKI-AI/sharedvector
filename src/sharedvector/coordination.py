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
"""Node-local coordination between a pool owner and its followers.

Three files per pool name live on a node-local filesystem (``/dev/shm`` by default):

* ``<name>``: the Boost.Interprocess segment itself.
* ``<name>.lock``: ``flock``-ed exclusively by the owner for as long as it lives; the kernel releases
  it on owner death, which makes "is the owner alive" a cheap, race-free question. Its content is a
  JSON document with the owner's ``generation`` token. The file is never unlinked, because
  unlink-and-recreate races would allow two simultaneous owners.
* ``<name>.ready``: written atomically by the producer service once the pool is fully bootstrapped;
  its content is the generation it was produced under, so followers never attach to a stale segment
  that a new owner is about to replace.
"""

import contextlib
import fcntl
import json
import logging
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from sharedvector.vector import SharedVector

logger = logging.getLogger(__name__)

DEFAULT_LOCK_DIR = Path("/dev/shm")
LOCK_DIR_ENV = "SHAREDVECTOR_LOCK_DIR"
_ATTACH_CHUNK_SIZE = 4096
_ATTACH_MAX_MEMORY = 65536


class PoolBusyError(RuntimeError):
    """Another process already owns the pool."""


class PoolSpawnError(RuntimeError):
    """The producer spawned for a pool failed before the pool became ready."""


def validate_name(name: str) -> str:
    """Return ``name`` if it is a valid segment name, else raise.

    Args:
        name: Candidate pool name.

    Returns:
        The validated name.

    Raises:
        ValueError: If the name is empty, contains a path separator, or is too long.
    """
    if not name or "/" in name or name in (".", "..") or len(name) > 200:
        raise ValueError(f"Invalid pool name {name!r}: must be a non-empty filename without '/'")
    return name


@dataclass(frozen=True)
class PoolPaths:
    """Filesystem locations backing one pool name."""

    name: str
    lock_dir: Path

    @classmethod
    def for_name(cls, name: str, lock_dir: Path | str | None = None) -> "PoolPaths":
        """Resolve the paths for ``name``.

        Args:
            name: Pool name.
            lock_dir: Directory for the lock and ready files; defaults to ``$SHAREDVECTOR_LOCK_DIR``
                or ``/dev/shm``. Must be node-local (never NFS).

        Returns:
            The resolved paths.
        """
        validate_name(name)
        resolved = Path(lock_dir) if lock_dir is not None else Path(os.environ.get(LOCK_DIR_ENV, DEFAULT_LOCK_DIR))
        return cls(name=name, lock_dir=resolved)

    @property
    def segment(self) -> Path:
        """Path of the shared-memory segment."""
        return Path("/dev/shm") / self.name

    @property
    def lock(self) -> Path:
        """Path of the owner lock file."""
        return self.lock_dir / f"{self.name}.lock"

    @property
    def marker(self) -> Path:
        """Path of the ready marker file."""
        return self.lock_dir / f"{self.name}.ready"


class OwnerLock:
    """Exclusive ``flock`` on the pool's lock file, held for the owner's lifetime."""

    def __init__(self, paths: PoolPaths, fd: int, generation: str) -> None:
        """Wrap an already-acquired lock; use :meth:`try_acquire` instead.

        Args:
            paths: Pool paths.
            fd: Open descriptor holding the lock.
            generation: Token written into the lock file.
        """
        self._paths = paths
        self._fd: int | None = fd
        self.generation = generation

    @classmethod
    def try_acquire(cls, paths: PoolPaths, grace: float = 1.0, poll: float = 0.05) -> "OwnerLock | None":
        """Try to become the owner of ``paths``.

        Retries for ``grace`` seconds so transient shared probes by followers do not steal an
        election from a would-be owner.

        Args:
            paths: Pool paths.
            grace: Seconds to keep retrying a contended lock.
            poll: Sleep between retries.

        Returns:
            The held lock, or ``None`` if another process holds it.
        """
        paths.lock_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(paths.lock, os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + grace
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    return None
                time.sleep(poll)
        generation = uuid.uuid4().hex
        payload = json.dumps({"generation": generation, "owner_pid": os.getpid(), "created": time.time()})
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, payload.encode())
        os.fsync(fd)
        return cls(paths, fd, generation)

    @property
    def held(self) -> bool:
        """Whether this object still holds the lock."""
        return self._fd is not None

    def release(self) -> None:
        """Release the lock (idempotent). The lock file itself is kept on disk."""
        if self._fd is None:
            return
        with contextlib.suppress(OSError):
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
        self._fd = None


def owner_alive(paths: PoolPaths) -> bool:
    """Return whether some process currently holds the owner lock.

    Args:
        paths: Pool paths.

    Returns:
        ``True`` if an exclusive lock is held by any process, ``False`` otherwise or if no lock
        file exists.
    """
    try:
        fd = os.open(paths.lock, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def read_generation(paths: PoolPaths) -> str | None:
    """Read the generation token from the lock file, or ``None`` if absent or unparseable."""
    try:
        payload = json.loads(paths.lock.read_text())
    except (FileNotFoundError, ValueError):
        return None
    generation = payload.get("generation") if isinstance(payload, dict) else None
    return generation if isinstance(generation, str) and generation else None


def write_marker(paths: PoolPaths, generation: str) -> None:
    """Atomically publish the ready marker for ``generation``."""
    paths.lock_dir.mkdir(parents=True, exist_ok=True)
    tmp = paths.marker.with_suffix(f".ready.{os.getpid()}.tmp")
    tmp.write_text(generation)
    os.replace(tmp, paths.marker)


def read_marker(paths: PoolPaths) -> str | None:
    """Return the generation recorded in the ready marker, or ``None`` if absent."""
    try:
        return paths.marker.read_text() or None
    except FileNotFoundError:
        return None


def clear_marker(paths: PoolPaths) -> None:
    """Remove the ready marker if present."""
    with contextlib.suppress(FileNotFoundError):
        paths.marker.unlink()


def attach(paths: PoolPaths) -> SharedVector:
    """Attach to an existing segment without ever creating one.

    Args:
        paths: Pool paths.

    Returns:
        A handle on the segment. Size arguments are ignored for an existing segment.

    Raises:
        FileNotFoundError: If the segment does not exist.
    """
    return SharedVector(paths.name, _ATTACH_CHUNK_SIZE, _ATTACH_MAX_MEMORY, create=False)


class ReadyState(NamedTuple):
    """Outcome of a readiness probe."""

    ready: bool
    reason: str
    generation: str | None


def probe_ready(paths: PoolPaths, slot_count: int, strict: bool = True) -> ReadyState:
    """Check whether the pool can be consumed right now.

    In strict mode the ready marker must exist and, while an owner holds the lock, must match the
    owner's generation; this prevents attaching to a segment a new owner is about to replace. A
    lock file that nobody holds is treated as a stale artifact and ignored. Non-strict mode only
    requires the segment to exist and hold ``slot_count`` chunks.

    Args:
        paths: Pool paths.
        slot_count: Number of chunks the pool must hold.
        strict: Whether to require the marker/generation protocol.

    Returns:
        The readiness state and a human-readable reason when not ready.
    """
    marker = read_marker(paths)
    if strict:
        if marker is None:
            return ReadyState(False, "ready marker missing", None)
        if owner_alive(paths):
            generation = read_generation(paths)
            if generation != marker:
                return ReadyState(False, "ready marker belongs to a previous owner", marker)
    try:
        vector = attach(paths)
    except FileNotFoundError:
        return ReadyState(False, "segment missing", marker)
    size = vector.size()
    del vector
    if size < slot_count:
        return ReadyState(False, f"{size}/{slot_count} slots filled", marker)
    return ReadyState(True, "ready", marker)


def wait_until_ready(
    paths: PoolPaths,
    slot_count: int,
    deadline: float,
    poll_interval: float = 0.1,
    strict: bool = True,
    abort: Callable[[], str | None] | None = None,
) -> ReadyState:
    """Block until :func:`probe_ready` succeeds, ``deadline`` passes, or ``abort`` asks to stop.

    Args:
        paths: Pool paths.
        slot_count: Number of chunks the pool must hold.
        deadline: Absolute ``time.monotonic()`` deadline.
        poll_interval: Sleep between probes.
        strict: Forwarded to :func:`probe_ready`.
        abort: Optional callable returning a reason string when waiting should stop early.

    Returns:
        The successful readiness state.

    Raises:
        PoolSpawnError: If ``abort`` returned a reason.
        TimeoutError: If the deadline passed; the message carries the last probe reason.
    """
    while True:
        state = probe_ready(paths, slot_count, strict)
        if state.ready:
            return state
        if abort is not None and (reason := abort()) is not None:
            raise PoolSpawnError(f"Pool {paths.name!r} did not become ready: {reason} (last probe: {state.reason})")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Pool {paths.name!r} not ready before timeout: {state.reason}")
        time.sleep(poll_interval)


__all__ = [
    "DEFAULT_LOCK_DIR",
    "LOCK_DIR_ENV",
    "OwnerLock",
    "PoolBusyError",
    "PoolPaths",
    "PoolSpawnError",
    "ReadyState",
    "attach",
    "clear_marker",
    "owner_alive",
    "probe_ready",
    "read_generation",
    "read_marker",
    "validate_name",
    "wait_until_ready",
    "write_marker",
]
