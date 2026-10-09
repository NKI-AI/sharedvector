import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path

import pytest

from helpers import FailingLoader, TaggedLoader, producer_subprocess, segment_exists, wait_until
from sharedvector.coordination import (
    OwnerLock,
    PoolPaths,
    PoolSpawnError,
    owner_alive,
    probe_ready,
    read_marker,
    write_marker,
)
from sharedvector.pool import PoolHandle, ProducerSpec, ensure_pool
from sharedvector.vector import SharedVector

pytestmark = pytest.mark.slow

SLOTS = 4


def _spec(source_size: int = 16) -> ProducerSpec:
    return ProducerSpec(loader=TaggedLoader(source_size), chunk_size=4096, num_producers=2, sleep_seconds=0.001)


def test_attach_times_out_with_reason(shm_name: str, lock_dir: Path) -> None:
    with pytest.raises(TimeoutError, match="ready marker missing"):
        ensure_pool(shm_name, SLOTS, mode="attach", ready_timeout=0.3, poll_interval=0.05, lock_dir=lock_dir)


def test_spawn_mode_requires_producer(shm_name: str, lock_dir: Path) -> None:
    with pytest.raises(ValueError, match="ProducerSpec"):
        ensure_pool(shm_name, SLOTS, mode="spawn", lock_dir=lock_dir)


def test_spawn_creates_pool_and_close_tears_it_down(shm_name: str, lock_dir: Path) -> None:
    paths = PoolPaths.for_name(shm_name, lock_dir)

    handle = ensure_pool(shm_name, SLOTS, producer=_spec(), mode="spawn", ready_timeout=60.0, lock_dir=lock_dir)

    assert handle.is_owner
    assert handle.generation == read_marker(paths)
    assert owner_alive(paths)
    assert handle.attach().size() == SLOTS

    handle.close()

    assert handle.closed
    assert not segment_exists(shm_name)
    assert read_marker(paths) is None
    assert not owner_alive(paths)
    assert handle._supervisor is not None and not handle._supervisor.is_alive()


def test_ensure_pool_is_idempotent_per_process(shm_name: str, lock_dir: Path) -> None:
    with ensure_pool(shm_name, SLOTS, producer=_spec(), ready_timeout=60.0, lock_dir=lock_dir) as handle:
        assert ensure_pool(shm_name, SLOTS, producer=_spec(), lock_dir=lock_dir) is handle
        with pytest.raises(ValueError, match="slot_count"):
            ensure_pool(shm_name, SLOTS + 1, producer=_spec(), lock_dir=lock_dir)


def test_auto_attaches_to_external_locked_producer(shm_name: str, lock_dir: Path) -> None:
    paths = PoolPaths.for_name(shm_name, lock_dir)
    with producer_subprocess(shm_name, lock_dir, slot_count=SLOTS, hold_lock=True) as proc:
        assert wait_until(lambda: probe_ready(paths, SLOTS).ready, timeout=30.0)
        handle = ensure_pool(shm_name, SLOTS, producer=_spec(), mode="auto", ready_timeout=60.0, lock_dir=lock_dir)

        assert not handle.is_owner
        assert handle.attach().size() == SLOTS
        handle.close()

        assert segment_exists(shm_name)
        assert proc.is_alive()


def test_pickled_handle_is_a_follower(shm_name: str, lock_dir: Path) -> None:
    with ensure_pool(shm_name, SLOTS, producer=_spec(), ready_timeout=60.0, lock_dir=lock_dir) as handle:
        revived = pickle.loads(pickle.dumps(handle))

        assert isinstance(revived, PoolHandle)
        assert not revived.is_owner
        assert revived.attach().size() == SLOTS
        revived.close()

        assert segment_exists(shm_name)
        assert not handle.closed


def test_forked_child_never_owns(shm_name: str, lock_dir: Path) -> None:
    with ensure_pool(shm_name, SLOTS, producer=_spec(), ready_timeout=60.0, lock_dir=lock_dir) as handle:
        pid = os.fork()
        if pid == 0:  # pragma: no cover - child
            ok = not handle.is_owner
            handle.close()
            os._exit(0 if ok else 1)
        _, status = os.waitpid(pid, 0)

        assert os.waitstatus_to_exitcode(status) == 0
        assert segment_exists(shm_name)
        assert handle.is_owner and not handle.closed


def test_stale_pool_is_taken_over(shm_name: str, lock_dir: Path) -> None:
    paths = PoolPaths.for_name(shm_name, lock_dir)
    stale = SharedVector(shm_name, 1024, 65536)
    write_marker(paths, "stale-generation")
    del stale

    with ensure_pool(shm_name, SLOTS, producer=_spec(), mode="auto", ready_timeout=60.0, lock_dir=lock_dir) as handle:
        assert handle.is_owner
        assert handle.generation != "stale-generation"
        assert read_marker(paths) == handle.generation
        assert handle.attach().size() == SLOTS


def test_failing_producer_raises_and_cleans_up(shm_name: str, lock_dir: Path) -> None:
    spec = ProducerSpec(loader=FailingLoader(), chunk_size=4096, num_producers=1)

    with pytest.raises(PoolSpawnError, match="supervisor exited"):
        ensure_pool(shm_name, SLOTS, producer=spec, mode="spawn", ready_timeout=60.0, lock_dir=lock_dir)

    assert not segment_exists(shm_name)
    assert not owner_alive(PoolPaths.for_name(shm_name, lock_dir))


def _elect(name: str, lock_dir: str, barrier, results, stop) -> None:
    barrier.wait()
    handle = ensure_pool(name, SLOTS, producer=_spec(), mode="spawn", ready_timeout=90.0, lock_dir=lock_dir)
    results.put((os.getpid(), handle.is_owner, handle.attach().size()))
    if handle.is_owner:
        stop.wait()
    handle.close()


def test_election_yields_exactly_one_owner(shm_name: str, lock_dir: Path) -> None:
    ctx = mp.get_context("spawn")
    barrier, results, stop = ctx.Barrier(4), ctx.Queue(), ctx.Event()
    procs = [ctx.Process(target=_elect, args=(shm_name, str(lock_dir), barrier, results, stop)) for _ in range(4)]
    for proc in procs:
        proc.start()

    collected = [results.get(timeout=120.0) for _ in range(4)]
    stop.set()
    for proc in procs:
        proc.join(timeout=60.0)

    assert sum(owner for _, owner, _ in collected) == 1
    assert all(size == SLOTS for _, _, size in collected)
    assert all(proc.exitcode == 0 for proc in procs)
    assert wait_until(lambda: not segment_exists(shm_name))


def _owner_hard_exit(name: str, lock_dir: str) -> None:
    ensure_pool(name, SLOTS, producer=_spec(), mode="spawn", ready_timeout=90.0, lock_dir=lock_dir)
    os._exit(0)


def test_owner_hard_exit_cleans_up_via_parent_death_signal(shm_name: str, lock_dir: Path) -> None:
    proc = mp.get_context("spawn").Process(target=_owner_hard_exit, args=(shm_name, str(lock_dir)))
    proc.start()
    proc.join(timeout=120.0)
    assert proc.exitcode == 0

    assert wait_until(lambda: not segment_exists(shm_name), timeout=20.0)
    assert wait_until(lambda: not owner_alive(PoolPaths.for_name(shm_name, lock_dir)), timeout=5.0)


def test_lock_file_survives_release(shm_name: str, lock_dir: Path) -> None:
    paths = PoolPaths.for_name(shm_name, lock_dir)
    lock = OwnerLock.try_acquire(paths)
    assert lock is not None

    lock.release()

    assert paths.lock.exists()
    assert not owner_alive(paths)
    time.sleep(0)
