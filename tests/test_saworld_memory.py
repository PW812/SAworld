"""CPU checks for independently rebuilt, detached SAWorld history memory."""
import copy
import tempfile
import types
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from mmcv import Config
from mmcv.parallel import collate
from mmcv.runner import BaseModule
from mmdet.models import build_head

import projects.mmdet3d_plugin  # register project components
from projects.mmdet3d_plugin.saworld.saworld_v2 import SAWorldV2
from projects.mmdet3d_plugin.datasets.nuscenes_saworld_memory_dataset import (
    DeferredTestHistory, SAWorldMemoryNuScenesDataset, _CachedPrepareImageInputs)
from projects.mmdet3d_plugin.datasets.nuscenes_resworld_dataset import ResWorldCustomNuScenesDataset
from projects.mmdet3d_plugin.datasets.pipelines.loading import PrepareImageInputs
from projects.mmdet3d_plugin.datasets.pipelines.memory import SAWorldMemoryMetadata

CONFIG = Path(__file__).resolve().parents[1] / 'projects/configs/saworld/saworld_config_v2_memory.py'


def head_config():
    config = copy.deepcopy(Config.fromfile(str(CONFIG)).model.pts_bbox_head)
    config.bev_h, config.bev_w = 4, 4
    config.positional_encoding.row_num_embed = 4
    config.positional_encoding.col_num_embed = 4
    return config


def metadata(frame, scene='a', pose=None):
    return dict(scene_token=scene, sample_idx=f'{scene}-{frame}',
                prev_idx=f'{scene}-{frame-1}' if frame else '',
                memory_timestamp=1_500_000_000.0 + frame * .5,
                memory_ego2global=np.eye(4) if pose is None else pose)


class ImageTransformCacheTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.paths = [Path(directory.name) / f'{index}.jpg' for index in range(3)]
        pixels = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
        for path in self.paths:
            Image.fromarray(pixels).save(path)
        self.cache = OrderedDict()
        self.loader = _CachedPrepareImageInputs(self.cache, 2, data_config={})

    def transform(self, loader, path, resize_dims=(12, 9), crop=(1, 1, 10, 8),
                  flip=False, rotate=0):
        with Image.open(path) as image:
            return loader.img_transform_core(image, resize_dims, crop, flip, rotate)

    def test_cached_pixels_match_each_augmentation(self):
        baseline = PrepareImageInputs(data_config={})
        for options in ({}, dict(flip=True), dict(rotate=7),
                        dict(resize_dims=(15, 11), crop=(2, 1, 10, 8))):
            expected = self.transform(baseline, self.paths[0], **options)
            for _ in range(2):
                actual = self.transform(self.loader, self.paths[0], **options)
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_current_and_history_share_transforms_without_mutation(self):
        history = _CachedPrepareImageInputs(self.cache, 2, data_config={})
        original = PrepareImageInputs.img_transform_core
        with patch.object(PrepareImageInputs, 'img_transform_core', autospec=True,
                          side_effect=original) as transform:
            image = self.transform(self.loader, self.paths[0])
            expected = np.asarray(image).copy()
            image.putpixel((0, 0), (255, 255, 255))
            actual = self.transform(history, self.paths[0])
            np.testing.assert_array_equal(np.asarray(actual), expected)
            self.assertEqual(transform.call_count, 1)

    def test_cache_evicts_least_recently_used_image(self):
        original = PrepareImageInputs.img_transform_core
        with patch.object(PrepareImageInputs, 'img_transform_core', autospec=True,
                          side_effect=original) as transform:
            for index in (0, 1, 0, 2, 0, 1):
                self.transform(self.loader, self.paths[index])
                self.assertLessEqual(len(self.cache), 2)
            self.assertEqual(transform.call_count, 4)

    def test_training_transform_bypasses_cache(self):
        self.loader.is_train = True
        original = PrepareImageInputs.img_transform_core
        with patch.object(PrepareImageInputs, 'img_transform_core', autospec=True,
                          side_effect=original) as transform:
            for _ in range(2):
                self.transform(self.loader, self.paths[0])
            self.assertEqual(transform.call_count, 2)
            self.assertFalse(self.cache)

    def test_dataset_enables_shared_cache_only_for_evaluation(self):
        from mmdet.datasets.pipelines import Compose

        def initialize(dataset, pipeline, test_mode):
            dataset.test_mode = test_mode
            dataset.pipeline = Compose(pipeline)
            dataset.data_infos = []

        with patch.object(ResWorldCustomNuScenesDataset, '__init__', initialize):
            for test_mode, size in ((False, 128), (True, 128), (True, 0)):
                dataset = SAWorldMemoryNuScenesDataset(
                    pipeline=[dict(type='PrepareImageInputs', data_config={},
                                   is_train=not test_mode)],
                    test_mode=test_mode, test_image_cache_size=size)
                expected = _CachedPrepareImageInputs if test_mode and size else PrepareImageInputs
                self.assertIs(type(dataset.history_images), expected)
                self.assertIs(type(dataset.pipeline.transforms[0]), expected)
                if test_mode and size:
                    self.assertIs(dataset.history_images.cache, dataset.pipeline.transforms[0].cache)

    def test_deferred_clip_from_worker_can_reconstruct_original_camera_inputs(self):
        from functools import partial

        ds = SAWorldMemoryNuScenesDataset.__new__(SAWorldMemoryNuScenesDataset)
        ds._test_history_id = 'validation'
        ds._test_history_config = dict(data_config=dict(cams=['front'], Ncams=1, input_size=(9, 12),
                                                         crop_h=(0., 0.)), sequential=True)
        ds.history_images = PrepareImageInputs(**ds._test_history_config)
        ds.history_metadata = SAWorldMemoryMetadata()
        ds.memory_frames, ds.memory_max_gap = 2, 1.
        ds.multi_adj_frame_id_cfg = (1, 3, 1)
        ds.data_infos = []
        for i, path in enumerate([self.paths[0], *self.paths]):
            rotation = [0.8, 0., 0., 0.6]
            cam = dict(data_path=str(path), cam_intrinsic=np.eye(3), sensor2ego_rotation=[1, 0, 0, 0],
                       ego2global_rotation=rotation, sensor2ego_translation=[0, 0, 0],
                       ego2global_translation=[i, 0, 0])
            ds.data_infos.append(dict(token=str(i), prev=str(i-1) if i > 1 else '',
                                      scene_token='s' if i else 'prev',
                                      timestamp=i*500000, cams={'front': cam}, can_bus=np.zeros(18),
                                      ego2global_translation=[i, 0, 0], ego2global_rotation=rotation))
        ds._token_to_index = {info['token']: i for i, info in enumerate(ds.data_infos)}
        with patch.object(ResWorldCustomNuScenesDataset, 'prepare_test_data', return_value={}):
            with patch.object(Image, 'open', side_effect=AssertionError('eagerly opened history')):
                sample = ds.prepare_test_data(3)
        loader = torch.utils.data.DataLoader([sample], num_workers=1,
                                             collate_fn=partial(collate, samples_per_gpu=1))
        clip = next(iter(loader))['history_memory'].data[0][0]
        self.assertIsInstance(clip, DeferredTestHistory)
        expected = ds._attach_history({}, 3)['history_memory'].data
        for entry, eager in zip(clip, expected):
            actual = clip.load_observation(entry)
            for value, reference in zip(actual['img_inputs'], eager['img_inputs']):
                torch.testing.assert_close(value, reference, atol=0, rtol=0)
            torch.testing.assert_close(entry['can_bus'], eager['can_bus'], atol=0, rtol=0)
            np.testing.assert_array_equal(actual['memory_ego2global'], eager['img_metas']['memory_ego2global'])


class MemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)
        self.head = build_head(head_config()).eval()

    def memory(self, batch=2):
        shape = (batch, 4, 16, 256)
        return dict(scene=torch.randn(shape), residual=torch.randn(shape),
                    points=torch.randn(batch, 4, 16, 3, dtype=torch.float64),
                    timestamp=torch.tensor([[metadata(f)['memory_timestamp'] for f in range(4)]] * batch,
                                           dtype=torch.float64),
                    valid=torch.ones(batch, 4, dtype=torch.bool))

    def inputs(self, frame=4):
        return dict(bev_inputs=[torch.randn(6, 256, 4, 4), torch.randn(2, 3, 18), self.memory()],
                    img_metas=[metadata(frame), metadata(frame, 'b')],
                    cmd=torch.tensor([[[[1., 0., 0.]]], [[[0., 0., 1.]]]]),
                    ego_his_trajs=torch.zeros(2, 1, 2, 2), ego_lcf_feat=torch.zeros(2, 1, 1, 9))

    def test_no_cross_batch_cache_and_empty_equals_baseline(self):
        self.assertEqual(type(self.head).__bases__, (BaseModule,))
        config = head_config()
        config.type = 'SAWorldHead'
        baseline = build_head(config).eval()
        self.head.load_state_dict(baseline.state_dict(), strict=False)
        inputs = self.inputs()
        inputs['bev_inputs'][2]['valid'][:] = False
        baseline_inputs = dict(inputs, bev_inputs=inputs['bev_inputs'][:2])
        with torch.no_grad():
            expected = baseline(**baseline_inputs)
            first = self.head(**inputs)
            self.head(**self.inputs())
            second = self.head(**inputs)
        for key in ('init_ego_fut_preds', 'ego_fut_preds', 'pred_bev'):
            torch.testing.assert_close(first[key], expected[key], atol=0, rtol=0)
            torch.testing.assert_close(second[key], first[key], atol=0, rtol=0)
        self.assertFalse(hasattr(self.head, 'memory_embedding'))
        self.assertFalse(any('timestamp' in key for key in self.head.state_dict()))
        self.assertTrue(torch.equal(self.head.memory_fusion_gate,
                                    torch.zeros_like(self.head.memory_fusion_gate)))
        config.type, config.memory_frames = 'SAWorldHeadV2Sample', 0
        no_memory = build_head(config).eval()
        no_memory.load_state_dict(baseline.state_dict(), strict=True)
        with torch.no_grad():
            actual = no_memory(**baseline_inputs)
        torch.testing.assert_close(actual['ego_fut_preds'], expected['ego_fut_preds'], atol=0, rtol=0)

    def test_alignment_and_future_rejection(self):
        memory = self.memory(batch=1)
        memory['points'][:] = torch.tensor([13., 2., 0.], dtype=torch.float64)
        pose = np.array([[0., -1., 0., 11.], [1., 0., 0., 0.],
                         [0., 0., 1., 0.], [0., 0., 0., 1.]])
        residual = torch.zeros(1, 16, 256)
        relative, ages = self.head.align_memory(memory, [metadata(4, pose=pose)], residual)
        torch.testing.assert_close(relative[0, 0, 0], torch.tensor([2., -2., 0.]))
        torch.testing.assert_close(ages[0], torch.tensor([2., 1.5, 1., .5]))
        with self.assertRaisesRegex(ValueError, 'strictly before'):
            self.head.align_memory(memory, [metadata(3)], residual)
        with self.assertRaisesRegex(ValueError, 'sample history'):
            self.head.align_memory(None, [metadata(4)], residual)

    def test_memory_detached_but_current_action_and_reader_have_gradients(self):
        memory = self.memory(batch=1)
        memory['scene'].requires_grad_(True)
        memory['residual'].requires_grad_(True)
        residual, scene, points = torch.randn(1, 16, 256), torch.randn(1, 16, 256), torch.randn(1, 16, 3)
        action = torch.randn(1, 3, 256, requires_grad=True)
        relative, ages = self.head.align_memory(memory, [metadata(4)], residual)
        retrieved = self.head.read_memory(scene, residual, action, points, relative, ages, memory)
        retrieved.square().mean().backward()
        self.assertIsNone(memory['scene'].grad)
        self.assertIsNone(memory['residual'].grad)
        self.assertTrue((action.grad.norm(dim=-1) > 0).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for name, p in self.head.named_parameters()
                            if name.startswith('memory_') and name != 'memory_fusion_gate'))

    def test_zero_gate_preserves_current_world_path_with_nonempty_memory(self):
        inputs = self.inputs()
        with torch.no_grad():
            with_memory = self.head(**inputs)
            inputs['bev_inputs'][2]['valid'][:] = False
            without_memory = self.head(**inputs)
        torch.testing.assert_close(with_memory['pred_bev'], without_memory['pred_bev'],
                                   atol=0, rtol=0)
        torch.testing.assert_close(with_memory['ego_fut_preds'], without_memory['ego_fut_preds'],
                                   atol=0, rtol=0)

    def test_training_loss_with_mixed_history_lengths(self):
        self.head.train()
        for history_length in (0, 4):
            inputs = self.inputs()
            inputs['bev_inputs'][2]['valid'][0] = False
            inputs['bev_inputs'][2]['valid'][1] = history_length > 0
            output = self.head(**inputs)
            self.assertEqual(output['memory_valid_frames'].tolist(), [0, history_length])
            losses = self.head.loss(None, None, None, None, output,
                                    torch.randn(2, 1, 6, 2), torch.ones(2, 1, 1, 6), inputs['cmd'], None)
            self.assertEqual(set(losses), {'loss_plan_reg', 'loss_plan_reg_init', 'loss_action_flow'})
            total = sum(losses.values())
            self.assertTrue(torch.isfinite(total))
            total.backward()
            grads = [p.grad for name, p in self.head.named_parameters() if name.startswith('memory_')]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            if history_length:
                self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
            self.head.zero_grad(set_to_none=True)

    def fake_detector(self):
        model = SAWorldV2.__new__(SAWorldV2)
        torch.nn.Module.__init__(model)
        model.pts_bbox_head = self.head
        model.test_bn = torch.nn.BatchNorm2d(256)
        def extract(instance, inputs, metas):
            images = inputs[0].transpose(0, 1).flatten(0, 1)
            return [instance.test_bn(images)], None
        model.extract_img_feat = types.MethodType(extract, model)
        return model

    def history_entry(self, frame, scene='a'):
        return dict(img_inputs=[torch.randn(3, 256, 4, 4)] + [torch.zeros(1)] * 6,
                    can_bus=torch.randn(3, 18), img_metas=metadata(frame, scene))

    def test_history_encoder_does_not_change_bn_and_restores_modes(self):
        model = self.fake_detector().train()
        head_flags = [m.training for m in model.modules()]
        mean, count = model.test_bn.running_mean.clone(), model.test_bn.num_batches_tracked.clone()
        history = [[self.history_entry(i) for i in range(4)],
                   [None, None, None, self.history_entry(0, 'b')]]
        cmd = self.inputs()['cmd']
        memory = model.build_history_memory(history, [metadata(4), metadata(1, 'b')], cmd)
        self.assertEqual(memory['valid'].sum(1).tolist(), [4, 1])
        self.assertTrue(all(not value.requires_grad and value.grad_fn is None for value in memory.values()))
        torch.testing.assert_close(mean, model.test_bn.running_mean, atol=0, rtol=0)
        torch.testing.assert_close(count, model.test_bn.num_batches_tracked, atol=0, rtol=0)
        self.assertEqual(head_flags, [m.training for m in model.modules()])
        model.build_history_memory([[None] * 4] * 2, [metadata(0), metadata(0, 'b')], cmd)
        again = model.build_history_memory(history, [metadata(4), metadata(1, 'b')], cmd)
        for key in memory:
            torch.testing.assert_close(again[key], memory[key], atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'same-scene'):
            model.build_history_memory(history, [metadata(4, 'wrong'), metadata(1, 'b')], cmd)

    def test_history_selection_uses_prev_chain_and_scene_boundary(self):
        dataset = SAWorldMemoryNuScenesDataset.__new__(SAWorldMemoryNuScenesDataset)
        dataset.memory_frames, dataset.memory_max_gap = 4, 1.
        dataset.data_infos = [dict(token=f'a{i}', prev=f'a{i-1}' if i else '',
                                   scene_token='a', timestamp=i*500000) for i in range(6)]
        dataset.data_infos.append(dict(token='b0', prev='a5', scene_token='b', timestamp=3000000))
        dataset._token_to_index = {info['token']: i for i, info in enumerate(dataset.data_infos)}
        self.assertEqual(dataset._history_indices(0), [None] * 4)
        self.assertEqual(dataset._history_indices(2), [None, None, 0, 1])
        self.assertEqual(dataset._history_indices(5), [1, 2, 3, 4])
        self.assertEqual(dataset._history_indices(6), [None] * 4)
        with patch.object(ResWorldCustomNuScenesDataset, 'prepare_train_data', return_value=None):
            with patch.object(dataset, '_history_indices', side_effect=AssertionError('rejected anchor loaded history')):
                self.assertIsNone(dataset.prepare_train_data(5))

    def test_cpu_only_history_collates_independent_clips(self):
        from mmcv.parallel import DataContainer as DC
        clips = [[None, None, None, self.history_entry(0)],
                 [self.history_entry(i, 'b') for i in range(4)]]
        result = collate([dict(history_memory=DC(clip, cpu_only=True)) for clip in clips], samples_per_gpu=2)
        actual = result['history_memory'].data[0]
        self.assertEqual(len(actual), 2)
        self.assertIsNone(actual[0][0])
        self.assertEqual(actual[1][3]['img_metas']['sample_idx'], 'b-3')

    def test_distributed_test_unwraps_cpu_only_history(self):
        import tempfile
        import torch.distributed as dist
        from functools import partial
        from mmcv.parallel import DataContainer as DC, MMDistributedDataParallel
        from projects.mmdet3d_plugin.saworld.apis import test as test_api

        clips = [[None, None, None, self.history_entry(0)],
                 [self.history_entry(i, 'b') for i in range(4)]]
        metas = [metadata(1), metadata(4, 'b')]
        samples = [dict(history_memory=DC(clip, cpu_only=True),
                        img_metas=[DC(meta, cpu_only=True)],
                        ego_fut_cmd=[DC(torch.ones(1, 1, 3), stack=True)])
                   for clip, meta in zip(clips, metas)]
        loader = torch.utils.data.DataLoader(
            samples, batch_size=2, collate_fn=partial(collate, samples_per_gpu=2))
        model = self.fake_detector()

        def forward(instance, return_loss, rescale, history_memory, img_metas, ego_fut_cmd):
            self.assertFalse(return_loss)
            self.assertTrue(rescale)
            self.assertIsInstance(history_memory, list)
            self.assertEqual(img_metas[0], metas)
            self.assertEqual(ego_fut_cmd[0].shape, (2, 1, 1, 3))
            self.assertIs(history_memory[0][3], clips[0][3])
            self.assertEqual(history_memory[0][3]['img_inputs'][0].device.type, 'cpu')
            memory = instance.build_history_memory(history_memory, img_metas[0], ego_fut_cmd[0])
            return memory['valid'].sum(1).tolist()

        model.forward = types.MethodType(forward, model)
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group('gloo', init_method=f'file://{directory}/init', rank=0, world_size=1)
            try:
                wrapped = MMDistributedDataParallel(model, broadcast_buffers=False)
                with patch.object(test_api.time, 'sleep'), patch.object(
                        test_api, 'collect_results_cpu', side_effect=lambda results, size, tmpdir: results):
                    self.assertEqual(test_api.custom_multi_gpu_test(wrapped, loader), [1, 4])
            finally:
                dist.destroy_process_group()

    def test_config_matches_random_anchor_training(self):
        cfg = Config.fromfile(str(CONFIG))
        self.assertEqual(cfg.data.shuffler_sampler.type, 'DistributedGroupSampler')
        self.assertTrue(cfg.data.train.filter_empty_gt)
        self.assertEqual(cfg.model.type, 'SAWorldV2')
        self.assertEqual(cfg.model.pts_bbox_head.type, 'SAWorldHeadV2Sample')
        self.assertEqual(cfg.model.pts_bbox_head.memory_frames, cfg.data.train.memory_frames)

    def test_legacy_registry_retains_streaming_forward(self):
        config = head_config()
        config.type = 'SAWorldHeadV2'
        legacy = build_head(config).eval()
        self.assertTrue(type(legacy).__module__.endswith('saworld_head_v2_streaming'))
        self.assertTrue(type(self.head).__module__.endswith('saworld_head_v2'))
        with torch.no_grad():
            for frame in (0, 1):
                inputs = self.inputs(frame=frame)
                inputs['bev_inputs'] = inputs['bev_inputs'][:2]
                output = legacy(**inputs)
                self.assertTrue(torch.isfinite(output['ego_fut_preds']).all())
                self.assertEqual(legacy.last_memory_size.tolist(), [frame, frame])


if __name__ == '__main__':
    unittest.main()
