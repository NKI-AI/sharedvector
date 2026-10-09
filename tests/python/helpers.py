"""Picklable helpers shared by the multi-process tests (importable from spawned children)."""

import multiprocessing as mp
import os
import signal
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from multiprocessing.process import BaseProcess
from pathlib import Path

import numpy as np

from sharedvector.coordination import OwnerLock, PoolPaths
from sharedvector.loaders import Chunk
from sharedvector.service import ProducerService
from sharedvector.vector import remove_shared_memory

SPACING = np.array([1.0, 2.0, 3.0])


class TaggedLoader:
    """Loader returning one chunk whose elements all equal ``source_index``."""

    def __init__(
        self, source_size: int, element_count: int = 4, dtype: str = "float32", with_meta: bool = True
    ) -> None:
        self.source_size = source_size
        self.element_count = element_count
        self.dtype = np.dtype(dtype)
        self.with_meta = with_meta

    def __len__(self) -> int:
        return self.source_size

    def __call__(self, source_index: int) -> list[Chunk]:
        arr = np.full(self.element_count, source_index, dtype=self.dtype)
        meta = SPACING if self.with_meta else np.empty(0, dtype=np.float64)
        return [Chunk(arr, meta)]


class FailingLoader:
    """Loader that always raises, so every producer worker dies."""

    def __len__(self) -> int:
        return 8

    def __call__(self, source_index: int) -> list[Chunk]:
        raise RuntimeError("loader failure")


def unique_name(prefix: str) -> str:
    return f"sv_{prefix}_{uuid.uuid4().hex[:8]}"


def segment_exists(name: str) -> bool:
    return (Path("/dev/shm") / name).exists()


def wait_until(predicate, timeout: float = 10.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def run_producer(
    name: str,
    lock_dir: str,
    source_size: int,
    element_count: int,
    slot_count: int,
    chunk_size: int,
    num_producers: int,
    sleep_seconds: float,
    dtype: str,
    hold_lock: bool,
) -> None:
    """Child-process entry point running a ProducerService, optionally as a locked owner."""
    paths = PoolPaths.for_name(name, lock_dir)
    lock = OwnerLock.try_acquire(paths) if hold_lock else None
    if hold_lock and lock is None:
        raise SystemExit(2)
    loader = TaggedLoader(source_size=source_size, element_count=element_count, dtype=dtype)
    ProducerService(
        name=name,
        loader=loader,
        source_size=source_size,
        slot_count=slot_count,
        chunk_size=chunk_size,
        num_producers=num_producers,
        sleep_seconds=sleep_seconds,
        dtype=dtype,
        generation=None if lock is None else lock.generation,
        lock_dir=lock_dir,
    ).run()


def spawn_producer(
    name: str,
    lock_dir: Path,
    source_size: int = 32,
    element_count: int = 4,
    slot_count: int = 8,
    chunk_size: int = 4096,
    num_producers: int = 2,
    sleep_seconds: float = 0.001,
    dtype: str = "float32",
    hold_lock: bool = False,
) -> BaseProcess:
    proc = mp.get_context("spawn").Process(
        target=run_producer,
        args=(
            name,
            str(lock_dir),
            source_size,
            element_count,
            slot_count,
            chunk_size,
            num_producers,
            sleep_seconds,
            dtype,
            hold_lock,
        ),
        daemon=False,
    )
    proc.start()
    return proc


def terminate_producer(proc: BaseProcess, name: str, sig: int = signal.SIGTERM) -> None:
    if proc.is_alive() and proc.pid is not None:
        os.kill(proc.pid, sig)
        proc.join(timeout=15.0)
        if proc.is_alive():
            proc.kill()
            proc.join()
    remove_shared_memory(name)


@contextmanager
def producer_subprocess(name: str, lock_dir: Path, **kwargs) -> Generator[BaseProcess]:
    proc = spawn_producer(name, lock_dir, **kwargs)
    try:
        yield proc
    finally:
        terminate_producer(proc, name)
