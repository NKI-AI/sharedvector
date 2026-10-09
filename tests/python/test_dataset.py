import pickle
from pathlib import Path

import pytest

from helpers import producer_subprocess, wait_until
from sharedvector.coordination import PoolPaths, probe_ready

pytestmark = [pytest.mark.torch, pytest.mark.slow]


def _ready(name: str, lock_dir: Path, slot_count: int) -> None:
    paths = PoolPaths.for_name(name, lock_dir)
    assert wait_until(lambda: probe_ready(paths, slot_count).ready, timeout=20.0)


def test_consumer_sees_initial_fill(shm_name: str, lock_dir: Path) -> None:
    from sharedvector.dataset import SharedMemoryDataset

    with producer_subprocess(shm_name, lock_dir, slot_count=8, source_size=32):
        ds = SharedMemoryDataset(shm_name, 8, ready_timeout=20.0, lock_dir=lock_dir)

        assert len(ds) == 8
        for i in range(8):
            tensor = ds[i]["data"]
            assert tensor.shape == (1, 4)
            assert 0 <= float(tensor[0, 0]) < 32


def test_timeout_when_no_producer(shm_name: str, lock_dir: Path) -> None:
    from sharedvector.dataset import SharedMemoryDataset

    with pytest.raises(TimeoutError, match="ready marker missing"):
        SharedMemoryDataset(shm_name, 4, ready_timeout=0.3, poll_interval=0.05, lock_dir=lock_dir)


def test_pickle_roundtrip_reattaches(shm_name: str, lock_dir: Path) -> None:
    from sharedvector.dataset import SharedMemoryDataset

    with producer_subprocess(shm_name, lock_dir, slot_count=8):
        ds = SharedMemoryDataset(shm_name, 8, ready_timeout=20.0, lock_dir=lock_dir)

        revived = pickle.loads(pickle.dumps(ds))

        assert len(revived) == 8
        assert revived[0]["data"].shape == (1, 4)


@pytest.mark.parametrize("num_workers,context", [(0, None), (2, "fork"), (2, "spawn")])
def test_dataloader_integration(shm_name: str, lock_dir: Path, num_workers: int, context: str | None) -> None:
    from torch.utils.data import DataLoader

    from sharedvector.dataset import SharedMemoryDataset

    with producer_subprocess(shm_name, lock_dir, slot_count=8):
        ds = SharedMemoryDataset(shm_name, 8, ready_timeout=20.0, lock_dir=lock_dir)

        batches = list(DataLoader(ds, batch_size=4, num_workers=num_workers, multiprocessing_context=context))

        assert len(batches) == 2
        assert all(batch["data"].shape == (4, 1, 4) for batch in batches)


def test_getitem_returns_spacing_as_float32(shm_name: str, lock_dir: Path) -> None:
    import torch

    from sharedvector.dataset import SharedMemoryDataset

    with producer_subprocess(shm_name, lock_dir):
        item = SharedMemoryDataset(shm_name, 8, ready_timeout=20.0, lock_dir=lock_dir)[0]

        assert item["spacing"].dtype == torch.float32
        assert torch.equal(item["spacing"], torch.tensor([1.0, 2.0, 3.0]))


def test_int16_segment_yields_int16_tensor_and_transforms_apply(shm_name: str, lock_dir: Path) -> None:
    import torch

    from sharedvector.dataset import SharedMemoryDataset

    with producer_subprocess(shm_name, lock_dir, dtype="int16"):
        ds = SharedMemoryDataset(
            shm_name, 8, ready_timeout=20.0, lock_dir=lock_dir, transforms=[lambda item: {**item, "tagged": True}]
        )
        item = ds[0]

        assert item["data"].dtype == torch.int16
        assert item["tagged"] is True


def test_non_strict_mode_ignores_marker(shm_name: str, lock_dir: Path) -> None:
    from sharedvector.dataset import SharedMemoryDataset

    with producer_subprocess(shm_name, lock_dir, slot_count=4):
        _ready(shm_name, lock_dir, 4)
        PoolPaths.for_name(shm_name, lock_dir).marker.unlink()

        ds = SharedMemoryDataset(shm_name, 4, ready_timeout=5.0, lock_dir=lock_dir, strict=False)

        assert len(ds) == 4
