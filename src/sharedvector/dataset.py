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
"""Consumer-side :class:`torch.utils.data.Dataset` over a producer-managed pool."""

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from sharedvector.coordination import PoolPaths, attach, wait_until_ready
from sharedvector.pool import PoolHandle
from sharedvector.vector import SharedVector

Transform = Callable[[dict[str, Any]], dict[str, Any]]


class SharedMemoryDataset(Dataset[dict[str, Any]]):
    """Dataset that attaches to a shared-memory pool by name.

    Safe to instantiate concurrently from many processes (e.g. DDP ranks). Construction blocks until
    the pool is ready, i.e. the producer has published its ready marker and filled ``slot_count``
    chunks. The dataset never owns the pool: no signal handlers, no segment removal. Detaching is
    automatic on garbage collection, and unpickling (DataLoader workers) re-attaches.

    Each item is ``{"data": tensor[1, ...]}`` plus ``meta_key`` holding the slot's metadata as a
    float32 tensor when the producer attached any.
    """

    def __init__(
        self,
        name: str,
        slot_count: int,
        ready_timeout: float = 60.0,
        poll_interval: float = 0.1,
        transforms: Sequence[Transform] = (),
        meta_key: str = "spacing",
        strict: bool = True,
        lock_dir: Path | str | None = None,
    ) -> None:
        """Attach to the named pool and wait for it to be ready.

        Args:
            name: Pool name.
            slot_count: Number of chunks the producer keeps resident; used as ``__len__`` and as the
                readiness threshold.
            ready_timeout: Total seconds to wait for readiness before raising.
            poll_interval: Sleep between readiness polls.
            transforms: CPU transforms applied in order to each item dict.
            meta_key: Item key receiving a slot's metadata; omitted when the slot carries none.
            strict: Require the owner's ready marker (see
                :func:`~sharedvector.coordination.probe_ready`); ``False`` only checks the segment.
            lock_dir: Directory of the lock and ready files; defaults as in
                :class:`~sharedvector.coordination.PoolPaths`.

        Raises:
            TimeoutError: If the pool does not become ready in time.
        """
        self._paths = PoolPaths.for_name(name, lock_dir)
        self._slot_count = slot_count
        self._ready_timeout = ready_timeout
        self._poll_interval = poll_interval
        self._transforms = tuple(transforms)
        self._meta_key = meta_key
        self._strict = strict
        self._vector: SharedVector = self._attach()

    @classmethod
    def from_pool(cls, handle: PoolHandle, **kwargs: Any) -> "SharedMemoryDataset":
        """Build a dataset for a pool obtained from :func:`~sharedvector.pool.ensure_pool`.

        Args:
            handle: Pool handle supplying ``name``, ``slot_count`` and ``lock_dir``.
            **kwargs: Remaining :class:`SharedMemoryDataset` arguments.

        Returns:
            The attached dataset.
        """
        return cls(handle.name, handle.slot_count, lock_dir=handle.lock_dir, **kwargs)

    @property
    def name(self) -> str:
        """Pool name."""
        return self._paths.name

    def _attach(self) -> SharedVector:
        """Wait for readiness and open the segment without creating it."""
        deadline = time.monotonic() + self._ready_timeout
        wait_until_ready(self._paths, self._slot_count, deadline, self._poll_interval, strict=self._strict)
        return attach(self._paths)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["_vector"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._vector = self._attach()

    def __len__(self) -> int:
        return self._slot_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        view = self._vector.get(index)
        meta = self._vector.get_meta(index)
        item: dict[str, Any] = {"data": torch.tensor(view).unsqueeze(0)}
        del view
        if meta.size:
            item[self._meta_key] = torch.from_numpy(meta).to(torch.float32)
        for transform in self._transforms:
            item = transform(item)
        return item


__all__ = ["SharedMemoryDataset", "Transform"]
