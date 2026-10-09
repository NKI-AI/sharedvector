import multiprocessing as mp

import numpy as np
import pytest

from sharedvector.vector import SharedVector, remove_shared_memory

MAX_MEMORY_SIZE = 64 * 1024 * 1024
CHUNK_SIZE = 10 * 10 * 10


def _vector(name: str, **kwargs) -> SharedVector:
    kwargs.setdefault("chunk_size", CHUNK_SIZE)
    kwargs.setdefault("max_memory_size", MAX_MEMORY_SIZE)
    return SharedVector(name, **kwargs)


def test_reference_count_and_pointer(shm_name: str) -> None:
    manager = _vector(shm_name)
    arr = np.random.random((2, 2)).astype(np.float32)
    manager.append(arr)
    manager.append(arr * 2)

    x = manager.get(0)

    assert not x.flags.owndata
    assert manager.get_chunk_pointer(0) == x.__array_interface__["data"][0]
    assert (x == arr).all()
    assert manager.get_chunk_ref_count(0) == 1
    y = manager.get(0)
    assert manager.get_chunk_ref_count(0) == 2
    del x
    assert manager.get_chunk_ref_count(0) == 1
    del y
    assert manager.get_chunk_ref_count(0) == 0
    manager.get(1)
    assert manager.get_chunk_ref_count(1) == 0


def _producer(memory_name: str, process_index: int, num_items: int) -> None:
    manager = SharedVector(memory_name, max_memory_size=MAX_MEMORY_SIZE, chunk_size=CHUNK_SIZE)
    for i in range(num_items):
        manager.append(np.array([float(process_index * num_items + i)], dtype=np.float32))


def _consumer(memory_name: str, results) -> None:
    manager = SharedVector(memory_name, max_memory_size=MAX_MEMORY_SIZE, chunk_size=CHUNK_SIZE)
    idx = 0
    while True:
        try:
            results.append(manager.get(idx)[0])
            idx += 1
        except IndexError:
            if len(results) == len(manager):
                break


@pytest.mark.slow
def test_multi_producer_consumer(shm_name: str) -> None:
    ctx = mp.get_context("spawn")
    manager = _vector(shm_name)
    assert manager.ref_count == 1
    num_producers, items_per_producer = 3, 5
    results = ctx.Manager().list()
    producers = [ctx.Process(target=_producer, args=(shm_name, i, items_per_producer)) for i in range(num_producers)]
    consumer = ctx.Process(target=_consumer, args=(shm_name, results))

    for p in producers:
        p.start()
    for p in producers:
        p.join()
    consumer.start()
    consumer.join()

    assert sorted(results) == list(range(num_producers * items_per_producer))


def test_replace(shm_name: str) -> None:
    manager = _vector(shm_name)
    arr = np.random.random((2, 2)).astype(np.float32)
    arr2 = np.random.random((3, 3)).astype(np.float32)
    manager.append(arr)
    manager.append(arr * 2)

    x = manager.get(1)
    assert (x == arr * 2).all()
    del x
    manager.replace(1, arr2)

    assert (manager.get(1) == arr2).all()


def test_replace_in_use_raises(shm_name: str) -> None:
    manager = _vector(shm_name)
    manager.append(np.ones(4, dtype=np.float32))
    held = manager.get(0)

    with pytest.raises(RuntimeError):
        manager.replace(0, np.zeros(4, dtype=np.float32))
    del held


def test_non_contiguous_array(shm_name: str) -> None:
    manager = _vector(shm_name)
    arr = np.random.random((2, 2)).astype(np.float32).T
    assert not arr.flags.contiguous

    with pytest.raises(RuntimeError):
        manager.append(arr)
    with pytest.raises(RuntimeError):
        manager.replace(0, arr)


def test_arbitrary_shapes(shm_name: str) -> None:
    manager = _vector(shm_name)
    arrays = [np.random.random((n, n)).astype(np.float32) for n in (2, 3, 4)]
    for arr in arrays:
        manager.append(arr)

    views = [manager.get(i) for i in range(3)]

    for view, arr in zip(views, arrays, strict=True):
        assert (view == arr).all()
    assert [manager.get_chunk_ref_count(i) for i in range(3)] == [1, 1, 1]
    del views, view
    assert [manager.get_chunk_ref_count(i) for i in range(3)] == [0, 0, 0]


def test_manager_ref_count(shm_name: str) -> None:
    manager = _vector(shm_name)
    assert manager.ref_count == 1

    others = [_vector(shm_name) for _ in range(4)]
    assert manager.ref_count == 5
    while others:
        others.pop()
    assert manager.ref_count == 1


def test_chunk_shape(shm_name: str) -> None:
    manager = _vector(shm_name)
    arr = np.random.random((2, 2)).astype(np.float32)

    manager.append(arr)
    manager.append(arr.reshape(4))

    assert manager.get_chunk_shape(0) == (2, 2)
    assert manager.get_chunk_shape(1) == (4,)


def test_size_visible_to_later_and_earlier_attachers(shm_name: str) -> None:
    producer = _vector(shm_name)
    early = _vector(shm_name)
    assert early.size() == 0

    for _ in range(5):
        producer.append(np.random.random((5, 5)).astype(np.float32))
    late = _vector(shm_name)

    assert early.size() == 5
    assert late.size() == 5
    assert len(late) == 5


def test_oversized_array_raises_memory_error(shm_name: str) -> None:
    from sharedvector.vector import MemoryError as SharedMemoryError

    manager = _vector(shm_name)

    with pytest.raises(SharedMemoryError):
        manager.append(np.zeros(CHUNK_SIZE, dtype=np.float32))


def test_out_of_range_raises_index_error(shm_name: str) -> None:
    manager = _vector(shm_name)

    with pytest.raises(IndexError):
        manager.get(0)


def test_int16_segment_returns_int16_zero_copy_view(shm_name: str) -> None:
    manager = _vector(shm_name, dtype="int16")
    arr = np.arange(12, dtype=np.int16).reshape(3, 4) - 6

    manager.append(arr)
    view = manager.get(0)

    assert manager.dtype == "int16"
    assert view.dtype == np.int16 and not view.flags.owndata
    np.testing.assert_array_equal(view, arr)


def test_append_rejects_dtype_mismatch(shm_name: str) -> None:
    manager = _vector(shm_name, dtype="int16")

    with pytest.raises(ValueError):
        manager.append(np.ones(4, dtype=np.float32))
    assert manager.size() == 0


def test_attach_adopts_dtype_and_rejects_conflict(shm_name: str) -> None:
    producer = _vector(shm_name, dtype="int16")

    consumer = SharedVector(shm_name, 4096, 65536)

    assert consumer.dtype == "int16"
    with pytest.raises(ValueError):
        SharedVector(shm_name, 4096, 65536, dtype="float32")
    del producer


def test_meta_roundtrip_and_clear_on_replace(shm_name: str) -> None:
    manager = _vector(shm_name)
    arr = np.ones(4, dtype=np.float32)

    manager.append(arr, np.array([0.5, 0.7, 2.0]))
    stored = manager.get_meta(0)
    manager.replace(0, arr)
    cleared = manager.get_meta(0)

    assert stored.dtype == np.float64 and stored.flags.owndata
    np.testing.assert_array_equal(stored, [0.5, 0.7, 2.0])
    assert cleared.shape == (0,)


def test_meta_over_capacity_raises(shm_name: str) -> None:
    manager = _vector(shm_name)

    with pytest.raises(ValueError):
        manager.append(np.ones(4, dtype=np.float32), np.ones(9))
    assert manager.size() == 0


def test_open_only_missing_segment_raises(shm_name: str) -> None:
    remove_shared_memory(shm_name)

    with pytest.raises(FileNotFoundError):
        SharedVector(shm_name, 4096, 65536, create=False)
    assert not (__import__("pathlib").Path("/dev/shm") / shm_name).exists()


def test_open_only_attaches_to_existing_segment(shm_name: str) -> None:
    producer = _vector(shm_name, dtype="int16")
    producer.append(np.arange(6, dtype=np.int16).reshape(2, 3))

    consumer = SharedVector(shm_name, 4096, 65536, create=False)

    assert consumer.size() == 1
    assert consumer.dtype == "int16"
    np.testing.assert_array_equal(consumer.get(0), np.arange(6).reshape(2, 3))
