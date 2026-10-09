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
"""Shared-memory chunk pool for multi-process data loading.

Importing this package needs only numpy. :class:`SharedMemoryDataset` is resolved lazily so torch
is imported only when it is actually used.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from sharedvector.coordination import OwnerLock, PoolBusyError, PoolPaths, PoolSpawnError, probe_ready
from sharedvector.lifecycle import LifecycleGuard
from sharedvector.loaders import Chunk, Loader, NumpyLoader, ZarrLoader, cast_volume, no_meta
from sharedvector.pool import PoolHandle, ProducerSpec, ensure_pool, pool
from sharedvector.service import ProducerService
from sharedvector.spacing import collect_spacings, write_spacing_manifest
from sharedvector.vector import (
    ArraySizeError,
    DataModifiedError,
    InUseError,
    MemoryError,
    OutOfMemoryError,
    SharedVector,
    remove_shared_memory,
)

try:
    __version__ = version("sharedvector")
except PackageNotFoundError:  # pragma: no cover - source tree without an install
    __version__ = "0.0.0"

_LAZY = {"SharedMemoryDataset": "sharedvector.dataset"}


def __getattr__(name: str) -> Any:
    """Resolve torch-dependent names on first use."""
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ArraySizeError",
    "Chunk",
    "DataModifiedError",
    "InUseError",
    "LifecycleGuard",
    "Loader",
    "MemoryError",
    "NumpyLoader",
    "OutOfMemoryError",
    "OwnerLock",
    "PoolBusyError",
    "PoolHandle",
    "PoolPaths",
    "PoolSpawnError",
    "ProducerService",
    "ProducerSpec",
    "SharedMemoryDataset",
    "SharedVector",
    "ZarrLoader",
    "__version__",
    "cast_volume",
    "collect_spacings",
    "ensure_pool",
    "no_meta",
    "pool",
    "probe_ready",
    "remove_shared_memory",
    "write_spacing_manifest",
]
