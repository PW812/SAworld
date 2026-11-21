"""Random current-frame sampling with an independent, causal history clip."""
import copy
from collections import OrderedDict

import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from nuscenes.eval.common.utils import Quaternion, quaternion_yaw

from .nuscenes_resworld_dataset import ResWorldCustomNuScenesDataset
from .pipelines.loading import PrepareImageInputs
from .pipelines.memory import SAWorldMemoryMetadata


class _CachedPrepareImageInputs(PrepareImageInputs):
    """Reuse evaluation image transforms within a worker's bounded cache."""

    def __init__(self, cache, cache_size, **kwargs):
        super().__init__(**kwargs)
        self.cache = cache
        self.cache_size = cache_size

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        filename = getattr(img, 'filename', None)
        if self.is_train or not filename:
            return super().img_transform_core(img, resize_dims, crop, flip, rotate)
        key = (filename, tuple(resize_dims), tuple(crop), bool(flip), float(rotate))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key].copy()
        transformed = super().img_transform_core(img, resize_dims, crop, flip, rotate)
        self.cache[key] = transformed.copy()
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return transformed


@DATASETS.register_module()
class SAWorldMemoryNuScenesDataset(ResWorldCustomNuScenesDataset):
    """Filter/resample the anchor exactly as SAWorld, then attach past frames.

    Each past frame has its own observed adjacent inputs for residual extraction.
    History uses images, calibration, pose and CAN state only: no GT boxes,
    maps, future trajectories, or historical future-derived navigation commands.
    """

    def __init__(self, memory_frames=4, memory_max_gap=1.0,
                 test_image_cache_size=128, **kwargs):
        if not isinstance(memory_frames, int) or memory_frames < 0:
            raise ValueError('memory_frames must be a nonnegative integer')
        if not isinstance(test_image_cache_size, int) or test_image_cache_size < 0:
            raise ValueError('test_image_cache_size must be a nonnegative integer')
        self.memory_frames = memory_frames
        self.memory_max_gap = memory_max_gap
        image_config = next(copy.deepcopy(stage) for stage in kwargs['pipeline']
                            if stage['type'] == 'PrepareImageInputs')
        image_config.pop('type')
        history_config = dict(image_config, load_point_label=False)
        self.history_images = PrepareImageInputs(**history_config)
        self.history_metadata = SAWorldMemoryMetadata()
        super().__init__(**kwargs)
        self._token_to_index = {info['token']: index for index, info in enumerate(self.data_infos)}
        if self.test_mode and test_image_cache_size:
            cache = OrderedDict()
            image_stage = next(index for index, stage in enumerate(self.pipeline.transforms)
                               if type(stage) is PrepareImageInputs)
            self.pipeline.transforms[image_stage] = _CachedPrepareImageInputs(
                cache, test_image_cache_size, **image_config)
            self.history_images = _CachedPrepareImageInputs(
                cache, test_image_cache_size, **history_config)

    def _history_indices(self, index):
        current = self.data_infos[index]
        indices = []
        while len(indices) < self.memory_frames:
            previous = self._token_to_index.get(current['prev'])
            if previous is None:
                break
            info = self.data_infos[previous]
            gap = (current['timestamp'] - info['timestamp']) / 1e6
            if info['scene_token'] != current['scene_token'] or not 0 < gap <= self.memory_max_gap:
                break
            indices.append(previous)
            current = info
        # Oldest -> newest. Missing history is masked padding, never a future frame.
        return [None] * (self.memory_frames - len(indices)) + indices[::-1]

    def _history_info(self, index):
        info = copy.deepcopy(self.data_infos[index])
        rotation = Quaternion(info['ego2global_rotation'])
        can_bus = info['can_bus'].copy()
        can_bus[:3] = info['ego2global_translation']
        can_bus[3:7] = rotation.elements
        angle = quaternion_yaw(rotation) % (2 * np.pi)
        can_bus[-2], can_bus[-1] = angle, np.degrees(angle)
        info['can_bus'] = can_bus
        return dict(curr=info, adjacent=copy.deepcopy(self.get_adj_info(info, index)),
                    timestamp=info['timestamp'] / 1e6, scene_token=info['scene_token'],
                    sample_idx=info['token'], prev_idx=info['prev'])

    def _attach_history(self, example, index):
        if example is None or not self.memory_frames:
            return example
        history = []
        # CPU-only payload avoids transferring all K image clips to GPU at once.
        for previous in self._history_indices(index):
            if previous is None:
                history.append(None)
                continue
            data = self.history_metadata(self.history_images(self._history_info(previous)))
            history.append(dict(
                img_inputs=data['img_inputs'], can_bus=data['can_bus'],
                img_metas={key: data[key] for key in (
                    'scene_token', 'sample_idx', 'prev_idx',
                    'memory_timestamp', 'memory_ego2global')}))
        example['history_memory'] = DC(history, cpu_only=True)
        return example

    def prepare_train_data(self, index):
        # Rejected anchors return None before loading history. The inherited
        # __getitem__ retries with SAWorld's original _rand_another behavior.
        return self._attach_history(super().prepare_train_data(index), index)

    def prepare_test_data(self, index):
        return self._attach_history(super().prepare_test_data(index), index)
