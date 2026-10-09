from pathlib import Path

import numpy as np
import pytest

from helpers import TaggedLoader, producer_subprocess, wait_until
from sharedvector.coordination import PoolPaths, attach, probe_ready
from sharedvector.pool import ProducerSpec, ensure_pool
from sharedvector.stats import PoolStats, collect_stats, stats_capacity_for
from sharedvector.vector import SharedVector

CHUNK = 1024
MEMORY = 4 * 1024 * 1024


def _vector(name: str, **kwargs) -> SharedVector:
    return SharedVector(name, CHUNK, MEMORY, **kwargs)


def _arr(value: float = 1.0) -> np.ndarray:
    return np.full(8, value, dtype=np.float32)


def test_get_counts_reads_and_get_meta_does_not(shm_name: str) -> None:
    vec = _vector(shm_name)
    vec.append(_arr(), np.array([1.0, 2.0, 3.0]))

    a = vec.get(0)
    b = vec.get(0)
    vec.get_meta(0)
    vec.get_chunk_shape(0)
    del a, b

    assert vec.get_chunk_read_count(0) == 2
    assert vec.get_chunk_write_count(0) == 1
    assert vec.get_chunk_last_reads(0) == 0


def test_replace_records_sample_and_resets_slot_counter(shm_name: str) -> None:
    vec = _vector(shm_name)
    vec.append(_arr())
    for _ in range(3):
        vec.get(0)

    vec.replace(0, _arr(2.0))

    assert vec.get_chunk_read_count(0) == 0
    assert vec.get_chunk_last_reads(0) == 3
    assert vec.get_chunk_write_count(0) == 2
    assert vec.stats == {
        "replace_events": 1,
        "reads_at_replace_total": 3,
        "reads_at_replace_max": 3,
        "blocked_replaces": 0,
        "sample_count": 1,
        "sample_capacity": 2 * (MEMORY // CHUNK),  # C++ default: two refresh cycles
    }
    samples = vec.read_samples()
    assert samples.dtype == np.uint32
    np.testing.assert_array_equal(samples, [3])


def test_blocked_replace_is_counted(shm_name: str) -> None:
    vec = _vector(shm_name)
    vec.append(_arr())
    pinned = vec.get(0)

    with pytest.raises(RuntimeError):
        vec.replace(0, _arr(2.0))
    del pinned

    assert vec.stats["blocked_replaces"] == 1
    assert vec.stats["replace_events"] == 0


def test_per_slot_arrays_and_sample_order(shm_name: str) -> None:
    vec = _vector(shm_name, stats_capacity=4)
    vec.append(_arr())
    vec.append(_arr())
    for _ in range(5):
        vec.get(0)
    vec.get(1)
    vec.replace(0, _arr())
    vec.replace(1, _arr())
    vec.replace(1, _arr())

    np.testing.assert_array_equal(vec.read_samples(), [5, 1, 0])
    np.testing.assert_array_equal(vec.last_reads(), [5, 0])
    np.testing.assert_array_equal(vec.write_counts(), [2, 3])
    np.testing.assert_array_equal(vec.read_counts(), [0, 0])
    assert vec.read_counts().dtype == np.uint64


def test_ring_buffer_wraps_keeping_newest(shm_name: str) -> None:
    vec = _vector(shm_name, stats_capacity=4)
    vec.append(_arr())
    for round_ in range(1, 7):
        for _ in range(round_):
            vec.get(0)
        vec.replace(0, _arr())

    assert vec.stats["sample_count"] == 4
    assert vec.stats["replace_events"] == 6
    np.testing.assert_array_equal(vec.read_samples(), [3, 4, 5, 6])


def test_reset_stats_clears_pool_counters_only(shm_name: str) -> None:
    vec = _vector(shm_name)
    vec.append(_arr())
    vec.get(0)
    vec.replace(0, _arr())

    vec.reset_stats()

    assert vec.stats["replace_events"] == 0
    assert vec.read_samples().size == 0
    assert vec.get_chunk_last_reads(0) == 1


def test_attached_consumer_shares_stats(shm_name: str) -> None:
    owner = _vector(shm_name)
    owner.append(_arr())
    consumer = SharedVector(shm_name, 4096, 65536, create=False)

    consumer.get(0)
    consumer.get(0)
    owner.replace(0, _arr())

    assert consumer.has_stats
    assert consumer.stats["replace_events"] == 1
    np.testing.assert_array_equal(consumer.read_samples(), [2])


def test_collect_stats_on_empty_pool_is_all_zeros(shm_name: str) -> None:
    stats = collect_stats(_vector(shm_name))

    assert isinstance(stats, PoolStats)
    assert stats.slots == 0
    assert stats.recent_count == 0
    assert stats.recent_median == 0.0
    assert stats.slot_last_reads_mean == 0.0


def test_collect_stats_summarises_counters_and_as_dict_prefix(shm_name: str) -> None:
    vec = _vector(shm_name)
    for _ in range(3):
        vec.append(_arr())
    for slot, reads in enumerate((2, 4, 9)):
        for _ in range(reads):
            vec.get(slot)
        vec.replace(slot, _arr())
    vec.get(0)

    stats = collect_stats(vec)
    flat = stats.as_dict(prefix="pool/")

    assert stats.replace_events == 3
    assert stats.reads_at_replace_mean == pytest.approx(5.0)
    assert stats.recent_median == 4.0
    assert stats.recent_max == 9
    assert stats.slot_last_reads_median == 4.0
    assert stats.slots_never_replaced == 0
    assert stats.slot_pending_reads_mean == pytest.approx(1 / 3)
    assert set(flat) == {f"pool/{name}" for name in PoolStats.__dataclass_fields__}
    assert all(isinstance(value, int | float) for value in flat.values())


def test_stats_capacity_follows_slot_count() -> None:
    assert stats_capacity_for(16) == 4096
    assert stats_capacity_for(10_000) == 20_000


@pytest.mark.slow
def test_consumer_reads_show_up_in_pool_stats(shm_name: str, lock_dir: Path) -> None:
    paths = PoolPaths.for_name(shm_name, lock_dir)
    with producer_subprocess(shm_name, lock_dir, slot_count=4, source_size=16):
        assert wait_until(lambda: probe_ready(paths, 4).ready, timeout=20.0)
        vec = attach(paths)
        for _ in range(20):
            for slot in range(4):
                vec.get(slot)
        assert wait_until(lambda: collect_stats(vec).replace_events >= 8, timeout=10.0)

        stats = collect_stats(vec)

        assert stats.slots == 4
        assert 0 <= stats.recent_median <= stats.recent_max <= stats.reads_at_replace_max
        assert stats.recent_count == min(stats.replace_events, stats_capacity_for(4))


@pytest.mark.slow
def test_pool_handle_stats(shm_name: str, lock_dir: Path) -> None:
    spec = ProducerSpec(loader=TaggedLoader(16), chunk_size=4096, num_producers=1, sleep_seconds=0.001)
    with ensure_pool(shm_name, 4, producer=spec, mode="spawn", ready_timeout=60.0, lock_dir=lock_dir) as handle:
        vec = handle.attach()
        for _ in range(10):
            vec.get(0)

        stats = handle.stats()

        assert stats.slots == 4
        assert stats.as_dict()["sharedvector/slots"] == 4
        assert handle.stats() is not stats


@pytest.mark.slow
@pytest.mark.torch
def test_dataset_stats(shm_name: str, lock_dir: Path) -> None:
    from sharedvector.dataset import SharedMemoryDataset

    with producer_subprocess(shm_name, lock_dir, slot_count=4):
        ds = SharedMemoryDataset(shm_name, 4, ready_timeout=20.0, lock_dir=lock_dir)
        for i in range(4):
            ds[i]

        stats = ds.stats()

        assert stats.slots == 4
        assert stats.slot_pending_reads_mean + stats.reads_at_replace_total > 0
