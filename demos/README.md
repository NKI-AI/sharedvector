# Demos

## `zarr_pool_demo.py`: producer + consumer on a toy zarr database

Needs the `torch` and `zarr` extras: `uv sync --all-extras`.

```bash
uv run python demos/zarr_pool_demo.py make-data        # 6 int16 volumes of 32x64x64 under /tmp/sharedvector_demo.zarr
uv run python demos/zarr_pool_demo.py run              # spawns the producer, streams batches until Ctrl-C, tears everything down
```

Every volume is filled with its own index, so the periodic log line shows which volumes the recent
batches were drawn from (the set keeps changing as the producer refreshes slots), the throughput, and
the `spacing` metadata travelling along with the chunks. Use `--batches N` for a bounded run and
`--log-every N` to change the reporting interval. The first `Ctrl-C` finishes the current batch and
shuts down cleanly; a second one (more than a second later) forces an immediate teardown.

### Two-terminal variant: standalone producer, attaching consumer

Terminal 1 (for example a CPU job step):

```bash
eval "$(uv run python demos/zarr_pool_demo.py producer-args)"
```

Terminal 2 (as many of these as you like, e.g. one per DDP rank):

```bash
uv run python demos/zarr_pool_demo.py run --mode attach
```

The consumer attaches as a follower; stopping it with `Ctrl-C` leaves the producer running. Stop the producer
with `Ctrl-C`; it removes the segment on the way out.

The pool's `.lock` file (`/dev/shm/sv_zarr_demo.lock`) is intentionally never deleted: it is the
owner-election anchor and is reused by the next producer with the same name.

Clean up the toy data with `rm -rf /tmp/sharedvector_demo.zarr`.
