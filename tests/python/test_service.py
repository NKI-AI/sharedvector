import os
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from helpers import FailingLoader, TaggedLoader, producer_subprocess, segment_exists, spawn_producer, wait_until
from sharedvector.coordination import PoolPaths, attach, probe_ready, read_marker
from sharedvector.service import ProducerService, partition


def test_partition_covers_range_evenly() -> None:
    assert partition(10, 3) == [(0, 4), (4, 7), (7, 10)]
    assert partition(2, 2) == [(0, 1), (1, 2)]


@pytest.mark.parametrize(
    "kwargs",
    [{"num_producers": 0}, {"slot_count": 1, "num_producers": 2}, {"source_size": 0}],
)
def test_init_validates_arguments(shm_name: str, lock_dir: Path, kwargs: dict) -> None:
    base: dict[str, Any] = {"source_size": 4, "slot_count": 4, "chunk_size": 1024, "num_producers": 1}
    base.update(kwargs)

    with pytest.raises(ValueError):
        ProducerService(shm_name, TaggedLoader(4), lock_dir=lock_dir, **base)


@pytest.mark.slow
def test_bootstrap_fills_pool_and_publishes_marker(shm_name: str, lock_dir: Path) -> None:
    paths = PoolPaths.for_name(shm_name, lock_dir)
    with producer_subprocess(shm_name, lock_dir, slot_count=8, source_size=32):
        assert wait_until(lambda: probe_ready(paths, 8).ready, timeout=20.0)

        vector = attach(paths)
        values = {float(vector.get(i)[0]) for i in range(8)}

        assert vector.size() == 8
        assert all(0 <= value < 32 and value == int(value) for value in values)
        assert read_marker(paths) is not None


@pytest.mark.slow
def test_held_reference_is_not_replaced(shm_name: str, lock_dir: Path) -> None:
    paths = PoolPaths.for_name(shm_name, lock_dir)
    with producer_subprocess(shm_name, lock_dir, slot_count=4, source_size=32):
        assert wait_until(lambda: probe_ready(paths, 4).ready, timeout=20.0)
        vector = attach(paths)
        held = vector.get(1)
        snapshot = held.copy()

        time.sleep(1.0)

        np.testing.assert_array_equal(held, snapshot)
        del held


@pytest.mark.slow
def test_refresh_eventually_replaces_idle_slot(shm_name: str, lock_dir: Path) -> None:
    paths = PoolPaths.for_name(shm_name, lock_dir)
    with producer_subprocess(shm_name, lock_dir, slot_count=4, source_size=32):
        assert wait_until(lambda: probe_ready(paths, 4).ready, timeout=20.0)
        vector = attach(paths)
        original = float(vector.get(0)[0])

        changed = wait_until(lambda: float(vector.get(0)[0]) != original, timeout=10.0)

        assert changed


@pytest.mark.slow
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_signal_unlinks_segment_and_marker(shm_name: str, lock_dir: Path, sig: int) -> None:
    paths = PoolPaths.for_name(shm_name, lock_dir)
    proc = spawn_producer(shm_name, lock_dir, slot_count=4, source_size=16)
    try:
        assert wait_until(lambda: probe_ready(paths, 4).ready, timeout=20.0)
        assert segment_exists(shm_name)
        assert proc.pid is not None

        os.kill(proc.pid, sig)
        proc.join(timeout=15.0)

        assert wait_until(lambda: not segment_exists(shm_name))
        assert read_marker(paths) is None
    finally:
        if proc.is_alive():
            proc.kill()
            proc.join()


@pytest.mark.slow
def test_context_manager_starts_and_stops_in_process(shm_name: str, lock_dir: Path) -> None:
    paths = PoolPaths.for_name(shm_name, lock_dir)
    service = ProducerService(
        shm_name, TaggedLoader(16), 16, 4, 4096, num_producers=2, lock_dir=lock_dir, generation="gen-1"
    )

    with service:
        assert service.started
        assert read_marker(paths) == "gen-1"
        assert probe_ready(paths, 4).ready

    assert not service.started
    assert read_marker(paths) is None
    assert not segment_exists(shm_name)


@pytest.mark.slow
def test_bootstrap_fails_fast_when_all_workers_die(shm_name: str, lock_dir: Path) -> None:
    service = ProducerService(shm_name, FailingLoader(), 8, 4, 4096, num_producers=1, lock_dir=lock_dir)

    with pytest.raises(RuntimeError, match="exited"):
        service.start()

    service.stop()
    assert not segment_exists(shm_name)
