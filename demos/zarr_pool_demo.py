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
"""Producer + consumer demo on a toy zarr database.

Requires the ``torch`` and ``zarr`` extras (``uv sync --all-extras``). Three subcommands:

* ``make-data``: write a small zarr v3 database (one array per volume, each filled with its volume index,
  tagged with an ``(x, y, z)`` spacing attribute) and its ``spacing.json`` manifest.
* ``run``: obtain the pool with :func:`sharedvector.ensure_pool` and stream batches out of a
  :class:`~sharedvector.dataset.SharedMemoryDataset` through a DataLoader until ``Ctrl-C`` (or for
  ``--batches N``). With ``--mode auto`` the script spawns and owns the producer itself; with
  ``--mode attach`` it expects a producer started elsewhere, for example with
  ``sharedvector-producer`` (see ``demos/README.md``).
* ``producer-args``: print the ``sharedvector-producer`` command line matching the chosen settings.

Example::

    uv run python demos/zarr_pool_demo.py make-data
    uv run python demos/zarr_pool_demo.py run
"""

import argparse
import faulthandler
import logging
import os
import shlex
import signal
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

logger = logging.getLogger("zarr_pool_demo")

DEFAULT_ROOT = Path("/tmp/sharedvector_demo.zarr")
DEFAULT_NAME = "sv_zarr_demo"


def make_data(root: Path, volumes: int, shape: tuple[int, int, int]) -> Path:
    """Create ``volumes`` int16 arrays under ``root`` and write the spacing manifest.

    Args:
        root: Zarr database root to create.
        volumes: Number of volumes.
        shape: ``(z, y, x)`` shape of every volume.

    Returns:
        Path of the written spacing manifest.
    """
    import zarr

    from sharedvector import write_spacing_manifest

    if root.exists():
        raise FileExistsError(f"{root} already exists; remove it first or pick another --root")
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for index in range(volumes):
        array = zarr.create_array(store=str(root / f"volume_{index:03d}"), shape=shape, dtype="int16", zarr_format=3)
        array[:] = np.full(shape, index, dtype=np.int16)
        array.attrs["spacing"] = [float(value) for value in rng.uniform(0.5, 3.0, size=3).round(2)]
    manifest = write_spacing_manifest(root)
    logger.info("Wrote %d volumes of shape %s under %s and manifest %s", volumes, shape, root, manifest)
    return manifest


def producer_spec(args: argparse.Namespace):
    """Build the :class:`~sharedvector.ProducerSpec` for the demo database."""
    from sharedvector import ProducerSpec, ZarrLoader

    loader = ZarrLoader(args.root, slices_per_chunk=args.slices_per_chunk, spacing_file=args.root / "spacing.json")
    return ProducerSpec(loader=loader, chunk_size=args.chunk_size_mb * 1024 * 1024, dtype="int16", num_producers=2)


def _batches(loader: "DataLoader", stop: threading.Event) -> Iterator[dict]:
    """Yield batches until ``stop`` is set, re-iterating the loader after every pass over the slots."""
    while not stop.is_set():
        for batch in loader:
            yield batch
            if stop.is_set():
                return


class _StopRequest:
    """Turn the first ``SIGINT``/``SIGTERM`` into a graceful stop and a second one into a forced exit.

    Installed after :func:`sharedvector.ensure_pool`, so it runs before the pool owner's own handler.
    The first signal only sets ``event``; the streaming loop then exits normally and closes the handle.
    A second signal chains to the previous handler, which tears the pool down immediately.
    """

    _DEBOUNCE_SECONDS = 1.0  # `uv run` forwards the terminal's SIGINT to its child: one Ctrl-C arrives twice

    def __init__(self) -> None:
        self.event = threading.Event()
        self._previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        self._pid = os.getpid()
        self._first_at = 0.0

    def install(self) -> None:
        """Install the handlers."""
        for sig in self._previous:
            signal.signal(sig, self)

    def __call__(self, signum: int, frame: object) -> None:
        """Handle one signal delivery."""
        if os.getpid() != self._pid:  # inherited by forked DataLoader workers; the main process decides
            return
        if not self.event.is_set():
            self.event.set()
            self._first_at = time.monotonic()
            logger.info("Signal %d received; finishing the current batch (send again to force)", signum)
            return
        if time.monotonic() - self._first_at < self._DEBOUNCE_SECONDS:
            return
        previous = self._previous[signum]
        if callable(previous):
            previous(signum, frame)
        else:
            signal.signal(signum, previous if previous is not None else signal.SIG_DFL)
            signal.raise_signal(signum)


def run(args: argparse.Namespace) -> None:
    """Attach to (or spawn) the pool and stream batches until interrupted or ``--batches`` is reached."""
    from torch.utils.data import DataLoader

    from sharedvector import SharedMemoryDataset, ensure_pool

    spec = None if args.mode == "attach" else producer_spec(args)
    started = time.monotonic()
    handle = ensure_pool(args.name, args.slot_count, producer=spec, mode=args.mode, ready_timeout=args.ready_timeout)
    role = "owner (spawned the producer)" if handle.is_owner else "follower (attached to an existing producer)"
    logger.info("Pool %r ready after %.1fs; this process is the %s", args.name, time.monotonic() - started, role)

    dataset = SharedMemoryDataset.from_pool(handle, transforms=[lambda item: {**item, "data": item["data"].float()}])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    limit = "until Ctrl-C" if args.batches is None else f"for {args.batches} batches"
    logger.info("Streaming %s; logging every %d batches", limit, args.log_every)
    stop = _StopRequest()
    stop.install()

    window_started = time.monotonic()
    window_volumes: set[int] = set()
    step = 0
    try:
        for step, batch in enumerate(_batches(loader, stop.event), start=1):
            window_volumes.update(int(value) for value in batch["data"][:, 0, 0, 0, 0].tolist())
            if step % args.log_every == 0:
                elapsed = time.monotonic() - window_started
                stats = handle.stats()  # same numbers end users would hand to e.g. wandb.log(stats.as_dict())
                logger.info(
                    "batch %d: data %s %s | %.1f batches/s | spacing[0]=%s | volumes in last %d batches %s | "
                    "reads/replace mean %.1f median %.0f max %d (%d replaces, %d blocked)",
                    step,
                    tuple(batch["data"].shape),
                    batch["data"].dtype,
                    args.log_every / elapsed,
                    [round(value, 2) for value in batch["spacing"][0].tolist()],
                    args.log_every,
                    sorted(window_volumes),
                    stats.recent_mean,
                    stats.recent_median,
                    stats.recent_max,
                    stats.replace_events,
                    stats.blocked_replaces,
                )
                window_started, window_volumes = time.monotonic(), set()
            if args.batches is not None and step >= args.batches:
                break
    except KeyboardInterrupt:
        logger.info("Interrupted after %d batches", step)
    finally:
        if stop.event.is_set():
            logger.info("Stopped by user after %d batches", step)
        was_owner = handle.is_owner
        handle.close()
        logger.info("Handle closed (%s)", "producer stopped, segment removed" if was_owner else "producer left running")


def producer_args(args: argparse.Namespace) -> None:
    """Print the ``sharedvector-producer`` command for a standalone producer."""
    loader_args = {
        "directory": str(args.root),
        "slices_per_chunk": args.slices_per_chunk,
        "spacing_file": str(args.root / "spacing.json"),
    }
    command = [
        "sharedvector-producer",
        "--name", args.name,
        "--slot-count", str(args.slot_count),
        "--chunk-size-mb", str(args.chunk_size_mb),
        "--dtype", "int16",
        "--num-producers", "2",
        "--loader-class", "sharedvector.loaders:ZarrLoader",
        "--loader-args", __import__("json").dumps(loader_args),
    ]  # fmt: skip
    print(shlex.join(command))


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the demo command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Zarr database root.")
    parser.add_argument("--name", default=DEFAULT_NAME, help="Pool name.")
    parser.add_argument("--slot-count", type=int, default=16, help="Chunks kept resident in shared memory.")
    parser.add_argument("--slices-per-chunk", type=int, default=8, help="Slices per chunk emitted by ZarrLoader.")
    parser.add_argument("--chunk-size-mb", type=int, default=1, help="Per-chunk budget in MiB.")
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("make-data", help="Create the toy zarr database.")
    make.add_argument("--volumes", type=int, default=6)
    make.add_argument("--shape", type=int, nargs=3, default=(32, 64, 64), metavar=("Z", "Y", "X"))

    runner = sub.add_parser("run", help="Stream batches from the pool.")
    runner.add_argument("--mode", choices=("auto", "attach"), default="auto")
    runner.add_argument("--batches", type=int, default=None, help="Stop after N batches (default: run until Ctrl-C).")
    runner.add_argument("--log-every", type=int, default=20, help="Log a progress line every N batches.")
    runner.add_argument("--batch-size", type=int, default=4)
    runner.add_argument("--num-workers", type=int, default=2)
    runner.add_argument("--ready-timeout", type=float, default=120.0)

    sub.add_parser("producer-args", help="Print the matching sharedvector-producer command.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    faulthandler.register(signal.SIGUSR1, all_threads=True)  # `kill -USR1 <pid>` dumps all Python stacks
    args = parse_args(argv)
    if args.command == "make-data":
        make_data(args.root, args.volumes, tuple(args.shape))
    elif args.command == "run":
        run(args)
    else:
        producer_args(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
