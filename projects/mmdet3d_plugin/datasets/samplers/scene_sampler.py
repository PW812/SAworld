"""Finite-epoch sequence sampling for a memory attached to each batch slot."""
import math

import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler

from .sampler import SAMPLER


@SAMPLER.register_module()
class SAWorldSceneSampler(Sampler):
    """Shuffle scenes, keeping frames chronological within every batch stream.

    Like StreamPETR's per-batch-slot sampling, one slot advances one sequence.
    This finite version visits every sample each epoch, with standard tail
    padding to equalize DDP iterations. Stream boundaries may split a scene;
    those streams start with empty memory and never share cached tensors.
    """

    def __init__(self, dataset, samples_per_gpu=1, num_replicas=None,
                 rank=None, seed=0):
        actual_rank, actual_replicas = get_dist_info()
        self.rank = actual_rank if rank is None else rank
        self.num_replicas = actual_replicas if num_replicas is None else num_replicas
        self.batch_size = samples_per_gpu
        self.seed = 0 if seed is None else seed
        self.epoch = 0
        groups = {}
        for index, info in enumerate(dataset.data_infos):
            groups.setdefault(info['scene_token'], []).append(index)
        self.scenes = [sorted(indices, key=lambda i: dataset.data_infos[i]['timestamp'])
                       for indices in groups.values()]
        self.size = len(dataset)
        if not self.size or self.batch_size < 1 or not 0 <= self.rank < self.num_replicas:
            raise ValueError('scene sampler requires a nonempty dataset and valid batch/rank')
        self.streams = self.batch_size * self.num_replicas
        self.steps = math.ceil(self.size / self.streams)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.scenes), generator=generator).tolist()
        indices = [index for scene in order for index in self.scenes[scene]]
        total = self.steps * self.streams
        indices = (indices * math.ceil(total / self.size))[:total]
        streams = torch.tensor(indices).reshape(self.streams, self.steps)
        local = streams[self.rank * self.batch_size:(self.rank + 1) * self.batch_size]
        return iter(local.t().reshape(-1).tolist())

    def __len__(self):
        return self.steps * self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch
