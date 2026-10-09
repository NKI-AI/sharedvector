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
"""Console entry points: the standalone producer and the spacing-manifest writer."""

import argparse
import importlib
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sharedvector.coordination import OwnerLock, PoolBusyError, PoolPaths
from sharedvector.loaders import Loader
from sharedvector.service import ProducerService
from sharedvector.spacing import write_spacing_manifest

logger = logging.getLogger(__name__)

EXIT_BUSY = 2


def instantiate_loader(spec: str, args_json: str | None) -> Loader:
    """Build a loader from ``"package.module:ClassName"`` and a JSON object of kwargs.

    Args:
        spec: Dotted module path and class name separated by ``:``.
        args_json: JSON object with constructor kwargs, or ``None``.

    Returns:
        The loader instance.

    Raises:
        ValueError: If ``spec`` lacks the ``:ClassName`` part.
    """
    module_name, _, class_name = spec.partition(":")
    if not class_name:
        raise ValueError("--loader-class must be 'package.module:ClassName'")
    cls = getattr(importlib.import_module(module_name), class_name)
    kwargs: dict[str, Any] = json.loads(args_json) if args_json else {}
    return cls(**kwargs)


def resolve_source_size(loader: Loader, override: int | None) -> int:
    """Return ``override`` or ``len(loader)``.

    Raises:
        ValueError: If neither is available.
    """
    if override is not None:
        return override
    if not hasattr(loader, "__len__"):
        raise ValueError("Loader does not define __len__; pass --source-size explicitly.")
    return len(loader)


def _producer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sharedvector-producer",
        description="Run a standalone shared-memory producer. Consumers attach by --name.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", required=True, help="Pool name (shared-memory segment name).")
    parser.add_argument("--slot-count", type=int, required=True, help="Resident chunks kept in shared memory.")
    parser.add_argument("--chunk-size-mb", type=float, required=True, help="Per-chunk budget in MiB.")
    parser.add_argument("--dtype", choices=("float32", "int16"), default="float32", help="Segment element dtype.")
    parser.add_argument("--num-producers", type=int, default=4, help="Worker processes refreshing idle slots.")
    parser.add_argument("--sleep-seconds", type=float, default=0.01, help="Worker cooldown between empty cycles.")
    parser.add_argument("--producer-join-timeout", type=float, default=5.0, help="Seconds before terminating workers.")
    parser.add_argument("--loader-class", required=True, help="Loader as 'package.module:ClassName'.")
    parser.add_argument("--loader-args", help="JSON object of kwargs passed to --loader-class.")
    parser.add_argument("--source-size", type=int, default=None, help="Override the loader's __len__.")
    parser.add_argument(
        "--lock-dir", type=Path, default=None, help="Directory for lock/ready files (default /dev/shm)."
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level name.")
    return parser


def producer_main(argv: Sequence[str] | None = None) -> int:
    """Run the producer until interrupted.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        Process exit code: ``0`` on clean shutdown, ``2`` if another owner holds the pool.
    """
    args = _producer_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    loader = instantiate_loader(args.loader_class, args.loader_args)
    source_size = resolve_source_size(loader, args.source_size)
    paths = PoolPaths.for_name(args.name, args.lock_dir)

    lock = OwnerLock.try_acquire(paths)
    if lock is None:
        logger.error("Pool %r is already owned by another process (lock %s)", args.name, paths.lock)
        return EXIT_BUSY
    try:
        service = ProducerService(
            name=args.name,
            loader=loader,
            source_size=source_size,
            slot_count=args.slot_count,
            chunk_size=int(args.chunk_size_mb * 1024 * 1024),
            num_producers=args.num_producers,
            sleep_seconds=args.sleep_seconds,
            producer_join_timeout=args.producer_join_timeout,
            dtype=args.dtype,
            generation=lock.generation,
            lock_dir=args.lock_dir,
            log_level=logging.getLogger().getEffectiveLevel(),
        )
        logger.info(
            "name=%s slot_count=%d source_size=%d chunk_size_mb=%s num_producers=%d dtype=%s",
            args.name,
            args.slot_count,
            source_size,
            args.chunk_size_mb,
            args.num_producers,
            args.dtype,
        )
        service.run()
    finally:
        lock.release()
    return 0


def _spacing_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sharedvector-spacing-manifest",
        description="Write {name: [x, y, z]} spacings of a zarr database to JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("root", type=Path, help="Zarr database root.")
    parser.add_argument("--output", type=Path, default=None, help="Manifest path; defaults to <root>/spacing.json.")
    parser.add_argument("--attr-key", default="spacing", help="Array attribute holding the (x, y, z) spacing.")
    parser.add_argument("--skip-missing", action="store_true", help="Warn and skip arrays without a valid spacing.")
    return parser


def spacing_manifest_main(argv: Sequence[str] | None = None) -> int:
    """Write the spacing manifest for a zarr database.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        Process exit code.
    """
    args = _spacing_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    output = write_spacing_manifest(args.root, args.output, args.attr_key, strict=not args.skip_missing)
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(producer_main())


__all__ = [
    "EXIT_BUSY",
    "PoolBusyError",
    "instantiate_loader",
    "producer_main",
    "resolve_source_size",
    "spacing_manifest_main",
]
