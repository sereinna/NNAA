"""中文说明：CEDG 候选组采样模块，用于让同一 parent scaffold 的候选样本更常在同一 batch 中出现。"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator

from torch.utils.data import Sampler


class CandidateGroupBatchSampler(Sampler[list[int]]):
    """Batch sampler that packs examples from the same candidate group together."""

    def __init__(
        self,
        group_ids: list[str],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 7,
    ) -> None:
        self.group_ids = group_ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        groups: dict[str, list[int]] = defaultdict(list)
        for index, group_id in enumerate(group_ids):
            groups[str(group_id)].append(index)
        self.groups = list(groups.values())

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        groups = [list(group) for group in self.groups]
        if self.shuffle:
            rng.shuffle(groups)
            for group in groups:
                rng.shuffle(group)

        batch: list[int] = []
        for group in groups:
            for index in group:
                batch.append(index)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if batch and not self.drop_last:
            yield batch
        self.epoch += 1

    def __len__(self) -> int:
        n_items = len(self.group_ids)
        if self.drop_last:
            return n_items // self.batch_size
        return (n_items + self.batch_size - 1) // self.batch_size
