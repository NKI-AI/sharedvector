# sharedvector

Shared-memory chunk pool for zero-copy, multi-process data loading. A producer keeps a pool of
array chunks resident in a named `/dev/shm` segment; any number of consumer processes (DDP ranks,
DataLoader workers) attach by name and read chunks without copying. Loading is fully decoupled from
training: the pool has a fixed number of slots whose contents are continuously refreshed from random
sources.

Built on Boost.Interprocess and pybind11; ships as binary wheels, so consumers only need `pip`/`uv`.
Linux only.

## Install

```bash
uv pip install sharedvector            # core: numpy only
uv pip install "sharedvector[torch]"   # + SharedMemoryDataset
uv pip install "sharedvector[zarr]"    # + ZarrLoader and the spacing manifest tool
```

## Usage

### Standalone producer (e.g. its own Slurm step), trainers attach

```bash
sharedvector-producer --name run42 --slot-count 256 --chunk-size-mb 16 --dtype int16 \
    --loader-class sharedvector.loaders:ZarrLoader \
    --loader-args '{"directory": "/data/db.zarr", "slices_per_chunk": 16, "spacing_file": "/data/db.zarr/spacing.json"}'
```

```python
from sharedvector import SharedMemoryDataset, ensure_pool

handle = ensure_pool("run42", slot_count=256, mode="attach")  # waits until the producer is ready
dataset = SharedMemoryDataset.from_pool(handle)
```

### Trainer spawns the producer itself

Every process on the node may run the same code. A node-local file lock elects one owner, which
spawns the producer in a detached supervisor process and tears it down on exit, `SIGINT` or `SIGTERM`.
All other processes (other DDP ranks, DataLoader workers) attach.

```python
from sharedvector import ProducerSpec, SharedMemoryDataset, ZarrLoader, ensure_pool

spec = ProducerSpec(loader=ZarrLoader("/data/db.zarr", slices_per_chunk=16), chunk_size=16 << 20, dtype="int16")
handle = ensure_pool("run42", slot_count=256, producer=spec, mode="auto")
dataset = SharedMemoryDataset.from_pool(handle, transforms=[...])
...
handle.close()  # owner only: stops the producer and removes the segment; followers just forget
```

`mode="auto"` attaches to a live pool when one exists and spawns otherwise; `mode="spawn"` always
runs the election; `mode="attach"` never spawns.

### Low-level API

```python
from sharedvector import SharedVector, remove_shared_memory

owner = SharedVector("demo", chunk_size=1 << 20, max_memory_size=64 << 20, dtype="float32")
owner.append(array, meta=np.array([2.5, 0.8, 0.8]))  # C-contiguous, <= chunk_size bytes, <= 8 meta values
view = SharedVector("demo", create=False).get(0)  # zero-copy view; pins the slot while alive
owner.replace(0, other_array)  # RuntimeError while a view is pinned
remove_shared_memory("demo")
```

## Statistics

Every attached process can read how the pool is being used. The segment counts, per slot, the reads
since the slot's last write, its write count and the read count it had when it was last replaced, and
pool-wide it keeps a ring of recent reads-before-replacement samples plus lifetime counters, including
replace attempts that were refused because a consumer had the slot pinned.

```python
stats = handle.stats()  # PoolHandle, SharedMemoryDataset and ProducerService all have .stats()
stats.recent_median  # reads a slot got before being replaced, over the recent window
stats.reads_at_replace_mean  # lifetime mean
stats.blocked_replaces  # producer/consumer contention
wandb.log(stats.as_dict())  # flat {"sharedvector/<field>": number} mapping

vec = handle.attach()  # low level: per-slot arrays and the raw ring
vec.read_counts(), vec.write_counts(), vec.last_reads(), vec.read_samples(), vec.stats, vec.reset_stats()
```

`get()` counts as one read; `get_meta()` does not. Counters are relaxed atomics, so a snapshot can lag a
few events. The ring holds two refresh cycles of the pool by default (`stats_capacity` on
`SharedVector`; the service sizes it with a 4096-sample floor).

## Coordination files

For a pool named `run42` (directory `/dev/shm`, override with `lock_dir=` or `SHAREDVECTOR_LOCK_DIR`;
must be node-local):

| File | Purpose |
|---|---|
| `/dev/shm/run42` | the segment |
| `run42.lock` | `flock`-ed by the owner for its lifetime; released by the kernel on death |
| `run42.ready` | written after bootstrap; carries the owner's generation so stale segments are never attached |

## Building from source

Requirements: a C++17 compiler and `uv`. Boost is fetched through Conan automatically during the
build (CMake and Conan are installed into the isolated build environment).

```bash
uv build                                     # wheel + sdist
uv sync --all-extras && uv run pytest        # dev environment + tests
cmake -S . -B build/ctest -DBUILD_TESTING=ON -DSHAREDVECTOR_BUILD_PYTHON=OFF && cmake --build build/ctest && ctest --test-dir build/ctest
```

Containers default to a 64 MiB `/dev/shm`; size it (`docker run --shm-size=...`) for real pools.

## License

Apache-2.0.
