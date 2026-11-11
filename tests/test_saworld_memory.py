"""CPU checks for independently rebuilt, detached SAWorld history memory."""
import copy
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate
from mmcv.runner import BaseModule
from mmdet.models import build_head

import projects.mmdet3d_plugin  # register project components
from projects.mmdet3d_plugin.saworld.saworld_v2 import SAWorldV2
from projects.mmdet3d_plugin.datasets.nuscenes_saworld_memory_dataset import SAWorldMemoryNuScenesDataset
from projects.mmdet3d_plugin.datasets.nuscenes_resworld_dataset import ResWorldCustomNuScenesDataset
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
                            for name, p in self.head.named_parameters() if name.startswith('memory_')))

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
