# some code from lthotse https://github.com/lhotse-speech/lhotse
import os
import warnings
import random
import librosa
import copy

from bisect import bisect_right
from typing import Iterable, Optional, Tuple, Union, List, Generator
from dataclasses import dataclass
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


Seconds = float


@dataclass
class Cut:
    name: str
    index: int
    duration: float


def is_none_or_gt(value, threshold) -> bool:
    return value is None or value > threshold


def check_constraint(constraint: Optional, max_duration: Optional, max_cuts: Optional):
    if constraint is not None:
        assert (
            max_duration is None and max_cuts is None
        ), "Cannot specify both constraint= and max_duration=/max_cuts="
    else:
        assert (
            max_duration is not None or max_cuts is not None
        ), "At least one of max_duration= or max_cuts= has to be defined (or provide constraint=)."


def pick_at_random(
    bucket: List[Cut], rng: random.Random, out_indexes_used: list):
    """
    Generator which will yield items in a sequence in a random order.
    It will append the indexes of items yielded during iteration via ``out_used_indexes``.
    """
    indexes = list(range(len(bucket)))
    rng.shuffle(indexes)
    for idx in indexes:
        out_indexes_used.append(idx)
        yield bucket[idx]


@dataclass
class TimeConstraint(object):
    """
    Represents a time-based constraint for sampler classes.
    It is defined as maximum total batch duration (in seconds) and/or the total number of cuts.

    :class:`TimeConstraint` can be used for tracking whether the criterion has been exceeded
    via the `add(cut)`, `exceeded()` and `reset()` methods.
    It will automatically track the right criterion (i.e. select duration from the cut).
    It can also be a null constraint (never exceeded).

    When ``quadratic_duration`` is set, we will try to compensate for models that have a
    quadratic complexity w.r.t. the input sequence length. We use the following formula
    to determine the effective duration for each cut::

        effective_duration = duration + (duration ** 2) / quadratic_duration

    We recomend setting quadratic_duration to something between 15 and 40 for transformer architectures.
    """

    max_duration: Optional[Seconds] = None
    max_cuts: Optional[int] = None
    current: Union[int, Seconds] = 0
    num_cuts: int = 0
    longest_seen: Union[int, float] = 0
    quadratic_duration: Optional[Seconds] = None

    def __post_init__(self) -> None:
        assert is_none_or_gt(self.max_duration, 0)
        assert is_none_or_gt(self.max_cuts, 0)
        assert is_none_or_gt(self.quadratic_duration, 0)

    def is_active(self) -> bool:
        """Is it an actual constraint, or a dummy one (i.e. never exceeded)."""
        return self.max_duration is not None or self.max_cuts is not None

    def add(self, example: Cut) -> None:
        """
        Increment the internal counter for the time constraint,
        selecting the right property from the input ``cut`` object.
        """
        if self.max_duration is not None:
            duration = self._maybe_apply_quadratic_correction(example.duration)
            self.current += duration
            self.longest_seen = max(self.longest_seen, duration)
        self.num_cuts += 1

    def _maybe_apply_quadratic_correction(self, duration: Seconds) -> Seconds:
        if self.quadratic_duration is None:
            return duration
        # For the quadratic complexity case, we add a term that accounts for
        # extra memory occupied by the model. The 1/quadratic_duration term causes
        # the effective duration to be doubled when it's equal to quadratic_duration.
        return duration + (duration**2) / self.quadratic_duration

    def exceeded(self) -> bool:
        """Is the constraint exceeded or not."""
        if self.max_cuts is not None and self.num_cuts > self.max_cuts:
            return True
        if self.max_duration is None:
            return False
        effective_duration = self.num_cuts * self.longest_seen
        return effective_duration > self.max_duration

    def close_to_exceeding(self) -> bool:
        """
        Check if the batch is close to satisfying the constraints.
        We define "closeness" as: if we added one more cut that has
        duration/num_frames/num_samples equal to the longest seen cut
        in the current batch, then the batch would have exceeded the constraints.
        """
        if self.max_cuts is not None and self.num_cuts >= self.max_cuts:
            return True

        if self.max_duration is not None:
            effective_duration = (self.num_cuts + 1) * self.longest_seen
            return effective_duration > self.max_duration
        return False

    def reset(self) -> None:
        """
        Reset the internal counter (to be used after a batch was created,
        to start collecting a new one).
        """
        self.current = 0
        self.num_cuts = 0
        self.longest_seen = 0

    def copy(self):
        """Return a shallow copy of this constraint."""
        return copy.copy(self)


# Note: this class is a subset of SimpleCutSampler and is "datapipes" ready.
class DurationBatcher:
    def __init__(
        self,
        datapipe: Iterable[Cut],
        max_duration: Seconds = None,
        max_cuts: Optional[int] = None,
        constraint: Optional[TimeConstraint] = None,
        drop_last: bool = False,
        quadratic_duration: Optional[Seconds] = None,
    ) -> None:
        self.datapipe = datapipe
        self.reuse_cuts_buffer = deque()
        self.drop_last = drop_last
        check_constraint(constraint, max_duration, max_cuts)
        if constraint is not None:
            self.constraint = constraint
        else:
            self.constraint = TimeConstraint(
                max_duration=max_duration,
                max_cuts=max_cuts,
                quadratic_duration=quadratic_duration,
            )

    def __iter__(self) -> Generator[Tuple[Cut], None, None]:
        self.cuts_iter = iter(self.datapipe)
        try:
            while True:
                yield self._collect_batch()
        except StopIteration:
            pass
        self.cuts_iter = None

    def _collect_batch(self) -> Union[Tuple[Cut]]:

        self.constraint.reset()
        cuts = []
        while True:
            # Check that we have not reached the end of the dataset.
            try:
                # If this doesn't raise (typical case), it's not the end: keep processing.
                cut = next(self.cuts_iter)
            except StopIteration:
                # No more cuts to sample from: if we have a partial batch,
                # we may output it, unless the user requested to drop it.
                # We also check if the batch is "almost there" to override drop_last.
                if cuts and (
                    not self.drop_last or self.constraint.close_to_exceeding()
                ):
                    # We have a partial batch and we can return it.
                    return cuts
                else:
                    # There is nothing more to return or it's discarded:
                    # signal the iteration code to stop.
                    raise StopIteration()

            # Track the duration/frames/etc. constraints.
            cuts.append(cut)
            self.constraint.add(cut)

            # Did we exceed the max_frames and max_cuts constraints?
            if self.constraint.close_to_exceeding():
                # Yes. Finish sampling this batch.
                if self.constraint.exceeded() and len(cuts) == 1:
                    warnings.warn(
                        "We have exceeded the max_duration constraint during sampling but have only 1 cut. "
                        "This is likely because max_duration was set to a very low value ~10s, "
                        "or you're using a CutSet with very long cuts (e.g. 100s of seconds long)."
                    )
                break

        return cuts


def estimate_duration_buckets(
    cuts: Iterable[Cut],
    num_buckets: int,
) -> List[float]:
    """
    Given an iterable of cuts and a desired number of buckets, select duration values
    that should start each bucket.

    The returned list, ``bins``, has ``num_buckets - 1`` elements.
    The first bucket should contain cuts with duration ``0 <= d < bins[0]``;
    the last bucket should contain cuts with duration ``bins[-1] <= d < float("inf")``,
    ``i``-th bucket should contain cuts with duration ``bins[i - 1] <= d < bins[i]``.

    :param cuts: an iterable of :class:`lhotse.cut.Cut`.
    :param num_buckets: desired number of buckets.
    :param constraint: object with ``.measure_length()`` method that's used to determine
        the size of each sample. If ``None``, we'll use ``TimeConstraint``.
    :return: a list of boundary duration values (floats).
    """
    assert num_buckets > 1
    sizes = np.array([c.duration for c in cuts])
    sizes.sort()
    assert num_buckets <= sizes.shape[0], (
        f"The number of buckets ({num_buckets}) must be smaller than "
        f"or equal to the number of cuts ({sizes.shape[0]})."
    )
    size_per_bucket = sizes.sum() / num_buckets

    bins = []
    tot = 0.0
    for size in sizes:
        if tot > size_per_bucket:
            bins.append(size)
            tot = 0.0
        tot += size

    return bins


class DynamicBucketingDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, max_duration, max_cuts, num_buckets,
                 shuffle=False, drop_last=False, quadratic_duration=None,
                 filter_max_duration=None, buffer_size=1000):
        self.max_duration = max_duration
        self.max_cuts = max_cuts
        self.shuffle = shuffle
        self.quadratic_duration = quadratic_duration
        self.drop_last = drop_last
        self.filter_max_duration = filter_max_duration
        self.constraint = TimeConstraint(
            max_duration=max_duration, max_cuts=max_cuts, quadratic_duration=quadratic_duration)
        self.rng = None
        self.epoch = 0
        self.seed = 1234
        self.buffer_size = buffer_size

        with open(filelist_path) as f:
            self.filelist = f.read().splitlines()

        if self.filter_max_duration is not None:
            self.filelist = self.filter_by_duration()
        # filter by length here.
        self.cuts = self.get_cuts(self.filelist)
        self.duration_bins = estimate_duration_buckets(self.cuts, num_buckets=num_buckets)
        self.buckets = [deque() for _ in range(len(self.duration_bins) + 1)]
        self.cuts_iter = iter(self.cuts)

    def get_name_duration(self, filelist_line):
        fields = filelist_line.split('|')
        if len(fields) > 1:
            name, duration = fields[0], float(fields[-1])
        else:
            name = Path(filelist_line).stem
            duration = librosa.get_duration(path=filelist_line)
        return name, duration

    def filter_by_duration(self):
        filelist = []
        for l in self.filelist:
            name, duration = self.get_name_duration(l)
            if duration < self.filter_max_duration:
                filelist.append(l)
        return filelist

    def get_cuts(self, filelist):
        cuts = []
        for i, l in enumerate(filelist):
            name, duration = self.get_name_duration(l)
            cuts.append(Cut(index=i, name=name, duration=duration))
        return cuts

    def _collect_cuts_in_buckets(self, n_cuts: int):
        try:
            for _ in range(n_cuts):
                cut = next(self.cuts_iter)
                bucket_idx = bisect_right(self.duration_bins, cut.duration)
                self.buckets[bucket_idx].append(cut)
        except StopIteration:
            pass

    def get_dynamic_batches(self):
        # if self.shuffle:
        #     print(f"pid [{os.getpid()}] shuffle data at epoch [{self.epoch}]")
        # else:
        #     print(f"pid [{os.getpid()}] organize data at epoch [{self.epoch}]")

        def is_ready(bucket):
            tot = self.constraint.copy()
            for c in bucket:
                tot.add(c[0] if isinstance(c, tuple) else c)
                if tot.close_to_exceeding():
                    return True
            return False

        self._collect_cuts_in_buckets(self.buffer_size)
        self.rng = random.Random(self.epoch + self.seed)
        self.epoch += 1
        batch_index = len(self.buckets) - 1  # test long sentence.
        batches = []
        try:
            while True:
                ready_buckets = [b for b in self.buckets if is_ready(b)]
                if not ready_buckets:
                    # No bucket has enough data to yield for the last full batch.
                    non_empty_buckets = [b for b in self.buckets if b]
                    if self.drop_last or len(non_empty_buckets) == 0:
                        # Either the user requested only full batches, or we have nothing left.
                        raise StopIteration()
                    else:
                        # Sample from partial batches that are left.
                        ready_buckets = non_empty_buckets

                indexes_used = []
                if self.shuffle:
                    sampling_bucket = self.rng.choice(ready_buckets)
                    maybe_shuffled = pick_at_random(
                        sampling_bucket, rng=self.rng, out_indexes_used=indexes_used)
                else:
                    sampling_bucket = self.buckets[batch_index % len(self.buckets)]
                    maybe_shuffled = sampling_bucket
                batch_index += 1
                batcher = DurationBatcher(maybe_shuffled, constraint=self.constraint.copy())
                batch = next(iter(batcher))
                batch_size = len(batch)
                batches.append(batch)
                if indexes_used:
                    # Shuffling, sort indexes of yielded elements largest -> smallest and remove them
                    indexes_used.sort(reverse=True)
                    for idx in indexes_used:
                        del sampling_bucket[idx]
                else:
                    # No shuffling, remove first N
                    for _ in range(batch_size):
                        sampling_bucket.popleft()
                self._collect_cuts_in_buckets(batch_size)
        except StopIteration:
            pass

        # clean up
        self.cuts_iter = iter(self.cuts)
        self.buckets = [deque() for _ in range(len(self.duration_bins) + 1)]

        return batches


class DynamicBucketingSampler(object):
    def __init__(self, dataset: DynamicBucketingDataset, random_batch_every_epoch=False):
        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0
        self.random_batch_every_epoch = random_batch_every_epoch
        self.dataset_shuffle = dataset.shuffle

        self.dataset = dataset
        self.batches = None
        self._len = None
        self.get_batches()

    def get_batches(self):
        batches = self.dataset.get_dynamic_batches()
        batches = batches[: len(batches) // self.num_replicas * self.num_replicas]
        batches = batches[self.rank::self.num_replicas]
        if self._len is None:
            self._len = len(batches)
        elif self._len > len(batches):
            extra_len = self._len - len(batches)
            assert extra_len < len(batches), "dataset too small."
            batches = batches + batches[-extra_len:]
        elif self._len < len(batches):
            batches = batches[:self._len]
        assert len(batches) == self._len
        self.batches = batches

    def __iter__(self):
        batch_indices = [[c.index for c in b] for b in self.batches]
        # print(f"pid [{os.getpid()}] rank [{self.rank}] num_samples [{len(batch_indices)}]")
        if self.random_batch_every_epoch and self.dataset_shuffle:
            self.get_batches()  # shuffle for the next epoch, time-consuming for large dataset.
        elif self.dataset_shuffle:
            rng = random.Random(self.dataset.epoch + self.dataset.seed)  # use a known rng for reproduce.
            rng.shuffle(self.batches)
        return iter(batch_indices)

    def __len__(self):
        return self._len


if __name__ == '__main__':
    filelist = "/data1/corpus/LJSpeech-1.1/synta_filelist.train"
    dataset = DynamicBucketingDataset(
        filelist_path=filelist, max_duration=100, max_cuts=32, num_buckets=20,
        shuffle=False, drop_last=False, quadratic_duration=15, filter_max_duration=10.)
    batches = dataset.get_dynamic_batches()
    for i, batch in enumerate(batches):
        print('batch idx', i, 'duration', sum([c.duration for c in batch]), 'num_samples', len(batch))
