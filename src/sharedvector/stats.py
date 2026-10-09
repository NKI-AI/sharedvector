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
"""Summaries of the pool's read statistics, ready for logging.

The segment keeps, per slot, the number of reads since the slot's last write, its write count and the
read count it had when it was last replaced. Pool-wide it keeps lifetime counters and a ring buffer of
recent reads-before-replacement samples. :func:`collect_stats` turns all of that into one flat,
JSON-friendly :class:`PoolStats`.
"""

from dataclasses import asdict, dataclass

import numpy as np

from sharedvector.vector import SharedVector


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else 0.0


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def _max(values: np.ndarray) -> int:
    return int(values.max()) if values.size else 0


@dataclass(frozen=True)
class PoolStats:
    """Snapshot of a pool's read statistics. All fields are plain ``int``/``float``.

    Attributes:
        replace_events: Replacements since the pool was created (or stats were reset).
        blocked_replaces: Replace attempts refused because a consumer had the slot pinned.
        reads_at_replace_total: Sum of reads-before-replacement over all replacements.
        reads_at_replace_max: Largest reads-before-replacement ever observed.
        reads_at_replace_mean: ``reads_at_replace_total / replace_events``.
        recent_count: Samples currently held in the ring buffer.
        recent_mean: Mean reads-before-replacement over the ring buffer.
        recent_median: Median over the ring buffer.
        recent_p10: 10th percentile over the ring buffer.
        recent_p90: 90th percentile over the ring buffer.
        recent_max: Maximum over the ring buffer.
        slots: Number of slots in the pool.
        slot_last_reads_mean: Mean over slots of the reads each slot had at its last replacement.
        slot_last_reads_median: Median of the same.
        slot_last_reads_max: Maximum of the same.
        slots_never_replaced: Slots still holding their initial (appended) chunk.
        slot_pending_reads_mean: Mean over slots of reads accumulated since the slot's last write.
    """

    replace_events: int
    blocked_replaces: int
    reads_at_replace_total: int
    reads_at_replace_max: int
    reads_at_replace_mean: float
    recent_count: int
    recent_mean: float
    recent_median: float
    recent_p10: float
    recent_p90: float
    recent_max: int
    slots: int
    slot_last_reads_mean: float
    slot_last_reads_median: float
    slot_last_reads_max: int
    slots_never_replaced: int
    slot_pending_reads_mean: float

    def as_dict(self, prefix: str = "sharedvector/") -> dict[str, int | float]:
        """Return the fields as a flat mapping, e.g. for ``wandb.log``.

        Args:
            prefix: Prepended to every key.

        Returns:
            Mapping of ``prefix + field`` to its value.
        """
        return {f"{prefix}{key}": value for key, value in asdict(self).items()}


def collect_stats(vector: SharedVector) -> PoolStats:
    """Read the statistics of ``vector``'s pool into a :class:`PoolStats`.

    Args:
        vector: Any handle attached to the pool (owner or consumer).

    Returns:
        The snapshot. Empty windows yield zeros rather than NaN.

    Raises:
        RuntimeError: If the segment was created without statistics support.
    """
    counters = vector.stats
    samples = vector.read_samples()
    write_counts = vector.write_counts()
    last_reads = vector.last_reads()
    pending = vector.read_counts()
    events = int(counters["replace_events"])
    total = int(counters["reads_at_replace_total"])
    return PoolStats(
        replace_events=events,
        blocked_replaces=int(counters["blocked_replaces"]),
        reads_at_replace_total=total,
        reads_at_replace_max=int(counters["reads_at_replace_max"]),
        reads_at_replace_mean=total / events if events else 0.0,
        recent_count=int(samples.size),
        recent_mean=_mean(samples),
        recent_median=_quantile(samples, 0.5),
        recent_p10=_quantile(samples, 0.1),
        recent_p90=_quantile(samples, 0.9),
        recent_max=_max(samples),
        slots=int(write_counts.size),
        slot_last_reads_mean=_mean(last_reads),
        slot_last_reads_median=_quantile(last_reads, 0.5),
        slot_last_reads_max=_max(last_reads),
        slots_never_replaced=int(np.count_nonzero(write_counts == 1)),
        slot_pending_reads_mean=_mean(pending),
    )


def stats_capacity_for(slot_count: int) -> int:
    """Ring-buffer capacity covering at least two refresh cycles of ``slot_count`` slots."""
    return max(4096, 2 * slot_count)


__all__ = ["PoolStats", "collect_stats", "stats_capacity_for"]
