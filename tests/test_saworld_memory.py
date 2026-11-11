"""Run from SAWorld: python -m unittest discover -s tests -p test_saworld_memory.py."""
import copy
import unittest
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import BaseModule
from mmdet.models import build_head

import projects.mmdet3d_plugin  # noqa: F401, registers the head and sampler
from projects.mmdet3d_plugin.datasets.pipelines.memory import SAWorldMemoryMetadata
from projects.mmdet3d_plugin.datasets.samplers.scene_sampler import SAWorldSceneSampler


CONFIG = Path(__file__).resolve().parents[1] / 'projects/configs/saworld/saworld_config_v2_memory.py'


def head_config():
    config = copy.deepcopy(Config.fromfile(str(CONFIG)).model.pts_bbox_head)
    config.bev_h, config.bev_w = 4, 4
    config.positional_encoding.row_num_embed = 4
    config.positional_encoding.col_num_embed = 4
    return config


def metadata(frame, scene='scene-a', pose=None, timestamp=None):
    return dict(scene_token=scene, sample_idx=f'{scene}-{frame}',
                prev_idx=f'{scene}-{frame-1}' if frame else '',
                memory_timestamp=1_500_000_000.0 + frame * .5 if timestamp is None else timestamp,
                memory_ego2global=np.eye(4) if pose is None else pose)


class MemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)
        self.head = build_head(head_config()).eval()

    def tensors(self, batch=2):
        return (torch.randn(batch, 16, 256), torch.randn(batch, 16, 256),
                torch.randn(batch, 16, 3))

    def update(self, metas, scene, residual, points):
        times, poses = self.head._memory_metadata(metas, residual)
        relative, ages = self.head.pre_update_memory(metas, residual, times, poses)
        self.head.post_update_memory(metas, scene, residual, points, times, poses)
        return relative, ages

    def test_fifo_capacity_and_detachment(self):
        self.assertEqual(type(self.head).__bases__, (BaseModule,))
        scene, residual, points = self.tensors()
        residual.requires_grad_(True)
        for frame in range(7):
            self.update([metadata(frame), metadata(frame, 'scene-b')],
                        scene, residual + frame, points)
            self.assertEqual(self.head.last_memory_size.tolist(), [min(frame, 4)] * 2)
            self.assertEqual(tuple(self.head.memory_embedding.shape), (2, 4, 16, 256))
            self.assertIsNone(self.head.memory_embedding.grad_fn)
            self.assertFalse(self.head.memory_embedding.requires_grad)
        torch.testing.assert_close(self.head.memory_embedding[:, 0], residual.detach() + 6)
        torch.testing.assert_close(self.head.memory_embedding[:, -1], residual.detach() + 3)
        self.assertFalse(any(key.startswith(('memory_embedding', 'memory_scene',
                                             'memory_timestamp', 'memory_valid',
                                             'memory_reference_point')) for key in self.head.state_dict()))

    def test_reset_is_per_slot_and_causal(self):
        scene, residual, points = self.tensors()
        self.update([metadata(0), metadata(0, 'scene-b')], scene, residual, points)
        self.update([metadata(1), metadata(0, 'scene-c')], scene, residual, points)
        self.assertEqual(self.head.last_memory_size.tolist(), [1, 0])
        # A repeat/backward timestamp and a wrong predecessor each reset.
        metas = [metadata(1), metadata(2, 'scene-c')]
        self.update(metas, scene, residual, points)
        self.assertEqual(self.head.last_memory_size.tolist(), [0, 0])
        metas = [metadata(2, timestamp=1_500_000_010.), metadata(3, 'scene-c')]
        self.update(metas, scene, residual, points)
        self.assertEqual(self.head.last_memory_size.tolist(), [0, 1])
        self.head.epoch = 1
        self.update([metadata(3), metadata(4, 'scene-c')], scene, residual, points)
        self.assertEqual(self.head.last_memory_size.tolist(), [0, 0])
        self.head.train()
        self.assertIsNone(self.head.memory_embedding)

    def test_pose_alignment_and_real_time_age(self):
        scene, residual, points = self.tensors(batch=1)
        points[:] = torch.tensor([3., 2., 0.])
        old_pose = np.eye(4)
        old_pose[0, 3] = 10
        self.update([metadata(0, pose=old_pose)], scene, residual, points)
        pose = np.array([[0., -1., 0., 11.], [1., 0., 0., 0.],
                         [0., 0., 1., 0.], [0., 0., 0., 1.]])
        metas = [metadata(1, pose=pose)]
        times, poses = self.head._memory_metadata(metas, residual)
        relative, age = self.head.pre_update_memory(metas, residual, times, poses)
        torch.testing.assert_close(relative[0, 0, 0], torch.tensor([2., -2., 0.]))
        self.assertEqual(float(age[0, 0]), .5)
        self.assertEqual(self.head.memory_timestamp.dtype, torch.float64)

    def test_empty_bank_and_all_action_slots(self):
        scene, residual, points = self.tensors(batch=1)
        metas = [metadata(0)]
        times, poses = self.head._memory_metadata(metas, residual)
        relative, ages = self.head.pre_update_memory(metas, residual, times, poses)
        action = torch.randn(1, 3, 256, requires_grad=True)
        output = self.head.read_memory(scene, residual, action, points, relative, ages)
        torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)
        output.sum().backward()
        self.assertTrue(all(p.grad is not None for name, p in self.head.named_parameters()
                            if name.startswith('memory_')))
        self.head.zero_grad(set_to_none=True)
        self.head.post_update_memory(metas, scene, residual, points, times, poses)
        times, poses = self.head._memory_metadata([metadata(1)], residual)
        relative, ages = self.head.pre_update_memory([metadata(1)], residual, times, poses)
        action = torch.randn(1, 3, 256, requires_grad=True)
        output = self.head.read_memory(scene, residual, action, points, relative, ages)
        output.square().mean().backward()
        self.assertTrue((action.grad.norm(dim=-1) > 0).all())
        changed = action.detach().clone()
        changed[:, 2] += 3
        other = self.head.read_memory(scene, residual, changed, points, relative, ages)
        self.assertFalse(torch.allclose(output, other))

    def inputs(self, frame):
        return dict(bev_inputs=[torch.randn(6, 256, 4, 4), torch.randn(2, 3, 18)],
                    img_metas=[metadata(frame), metadata(frame, 'scene-b')],
                    cmd=torch.tensor([[[[1., 0., 0.]]], [[[0., 0., 1.]]]]),
                    ego_his_trajs=torch.zeros(2, 1, 2, 2),
                    ego_lcf_feat=torch.zeros(2, 1, 1, 9))

    def test_first_frame_matches_baseline_and_zero_memory_ablation(self):
        config = head_config()
        config.type = 'SAWorldHead'
        baseline = build_head(config).eval()
        common = baseline.state_dict()
        self.head.load_state_dict(common, strict=False)
        inputs = self.inputs(0)
        with torch.no_grad():
            expected = baseline(**inputs)
            actual = self.head(**inputs)
        for key in ('init_ego_fut_preds', 'ego_fut_preds', 'pred_bev'):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        config.type, config.memory_frames = 'SAWorldHeadV2', 0
        no_memory = build_head(config).eval()
        no_memory.load_state_dict(common, strict=True)
        with torch.no_grad():
            actual = no_memory(**inputs)
        torch.testing.assert_close(actual['ego_fut_preds'], expected['ego_fut_preds'], atol=0, rtol=0)

    def test_training_loss_and_second_frame_backward(self):
        self.head.train()
        for frame in (0, 1):
            inputs = self.inputs(frame)
            output = self.head(**inputs)
            losses = self.head.loss(None, None, None, None, output,
                                    torch.randn(2, 1, 6, 2), torch.ones(2, 1, 1, 6),
                                    inputs['cmd'], None)
            self.assertEqual(set(losses), {'loss_plan_reg', 'loss_plan_reg_init', 'loss_action_flow'})
            total = sum(losses.values())
            self.assertTrue(torch.isfinite(total))
            total.backward()
            memory_grads = [p.grad for name, p in self.head.named_parameters() if name.startswith('memory_')]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in memory_grads))
            if frame:
                self.assertGreater(sum(float(g.abs().sum()) for g in memory_grads), 0)
            self.head.zero_grad(set_to_none=True)

    def test_metadata_uses_camera_ego_and_inverse_bda(self):
        camera_ego = torch.eye(4).unsqueeze(0)
        camera_ego[0, 0, 3] = 10
        bda = torch.diag(torch.tensor([-2., 2., 2.]))
        results = dict(img_inputs=[None, None, camera_ego, None, None, None, bda], timestamp=123.5)
        output = SAWorldMemoryMetadata()(results)
        np.testing.assert_allclose(output['memory_ego2global'] @ np.array([2., 4., 0., 1.]),
                                   np.array([9., 2., 0., 1.]))
        self.assertEqual(output['memory_timestamp'], 123.5)

    def test_scene_sampler_preserves_slot_order_and_coverage(self):
        infos = [dict(scene_token=f'scene-{s}', timestamp=t)
                 for t in range(9) for s in range(5) if t <= 4 + s]
        class Dataset:
            data_infos = infos
            def __len__(self):
                return len(self.data_infos)
        dataset = Dataset()
        rank_samples = []
        for rank in (0, 1):
            sampler = SAWorldSceneSampler(dataset, samples_per_gpu=2, num_replicas=2, rank=rank)
            samples = list(sampler)
            self.assertEqual(len(samples), len(sampler))
            rank_samples.extend(samples)
            for slot in (samples[::2], samples[1::2]):
                for i, j in zip(slot, slot[1:]):
                    if infos[i]['scene_token'] == infos[j]['scene_token']:
                        self.assertEqual(infos[j]['timestamp'], infos[i]['timestamp'] + 1)
            sampler.set_epoch(1)
            self.assertNotEqual(samples, list(sampler))
        self.assertEqual(set(rank_samples), set(range(len(dataset))))


if __name__ == '__main__':
    unittest.main()
