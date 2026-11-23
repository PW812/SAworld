"""Behavioral checks for early history conditioning and shared residual dynamics."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from mmcv import Config
from mmcv.runner import BaseModule
from mmdet.models import build_head

import projects.mmdet3d_plugin
from projects.mmdet3d_plugin.datasets.nuscenes_saworld_memory_dataset import DeferredTestHistory
from projects.mmdet3d_plugin.saworld.saworld_head_v3 import ResidualDynamics
from projects.mmdet3d_plugin.saworld.saworld_v2 import SAWorldV2


CONFIG = Path(__file__).resolve().parents[1] / 'projects/configs/saworld/saworld_config_v3_dynamics.py'


def head_config():
    config = copy.deepcopy(Config.fromfile(str(CONFIG)).model.pts_bbox_head)
    config.bev_h = config.bev_w = 4
    config.positional_encoding.row_num_embed = 4
    config.positional_encoding.col_num_embed = 4
    return config


def metadata():
    return dict(scene_token='a', sample_idx='a4', prev_idx='a3',
                memory_timestamp=1_500_000_002., memory_ego2global=np.eye(4))


class DynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(17)
        self.head = build_head(head_config()).eval()

    def memory(self, batch=2):
        shape = (batch, 4, 16, 256)
        return dict(scene=torch.randn(shape), residual=torch.randn(shape) * .1,
                    points=torch.randn(batch, 4, 16, 3, dtype=torch.float64),
                    timestamp=torch.tensor([[1_500_000_000. + i * .5 for i in range(4)]] * batch,
                                           dtype=torch.float64),
                    valid=torch.ones(batch, 4, dtype=torch.bool))

    def inputs(self):
        return dict(bev_inputs=[torch.randn(6, 256, 4, 4), torch.randn(2, 3, 18), self.memory()],
                    img_metas=[metadata(), metadata()],
                    cmd=torch.tensor([[[[1., 0., 0.]]], [[[0., 0., 1.]]]]),
                    ego_his_trajs=torch.zeros(2, 1, 2, 2), ego_lcf_feat=torch.zeros(2, 1, 1, 9))

    def test_history_changes_action_prior_and_future_without_persistent_state(self):
        inputs = self.inputs()
        changed = copy.deepcopy(inputs)
        changed['bev_inputs'][2]['residual'] += 1.
        with torch.no_grad():
            first = self.head(**inputs)
            second = self.head(**changed)
            again = self.head(**inputs)
        for key in ('action_query', 'init_ego_fut_preds', 'ego_fut_preds', 'future_residuals'):
            self.assertFalse(torch.allclose(first[key], second[key]), key)
            torch.testing.assert_close(first[key], again[key], rtol=0, atol=0)
        self.assertEqual(type(self.head).__bases__, (BaseModule,))
        self.assertFalse(hasattr(self.head, 'memory_fusion_gate'))
        self.assertFalse(hasattr(self.head, 'memory_action_attn'))
        self.assertFalse(hasattr(self.head, 'res_tokenlearner'))

    def test_prior_and_flow_gradients_reach_dynamics_but_not_history_encoder(self):
        self.head.train()
        inputs = self.inputs()
        memory = inputs['bev_inputs'][2]
        memory['scene'].requires_grad_(True)
        memory['residual'].requires_grad_(True)
        inputs['bev_inputs'][0].requires_grad_(True)
        outputs = self.head(**inputs)
        losses = self.head.loss(None, None, None, None, outputs,
                                torch.randn(2, 1, 6, 2), torch.ones(2, 1, 1, 6), inputs['cmd'], None)
        self.assertEqual(set(losses), {'loss_plan_reg', 'loss_plan_reg_init',
                                       'loss_action_flow', 'loss_memory_dynamics'})
        for key in ('loss_plan_reg_init', 'loss_action_flow'):
            self.head.zero_grad(set_to_none=True)
            losses[key].backward(retain_graph=True)
            self.assertGreater(self.head.dynamics.update.weight_ih.grad.abs().sum().item(), 0)
            self.assertGreater(self.head.memory_query[1].weight.grad.abs().sum().item(), 0)
        self.head.zero_grad(set_to_none=True)
        sum(losses.values()).backward()
        self.assertIsNone(memory['scene'].grad)
        self.assertIsNone(memory['residual'].grad)
        self.assertGreater(inputs['bev_inputs'][0].grad.abs().sum().item(), 0)
        for name, parameter in self.head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_empty_and_partial_history_masks_ignore_padding_values(self):
        inputs = self.inputs()
        memory = inputs['bev_inputs'][2]
        memory['valid'][0] = False
        memory['valid'][1, :2] = False
        with torch.no_grad():
            expected = self.head(**inputs)
            for key in ('scene', 'residual', 'points', 'timestamp'):
                memory[key][~memory['valid']] = float('nan')
            actual = self.head(**inputs)
        self.assertEqual(actual['memory_valid_frames'].tolist(), [0, 2])
        for key in ('ego_fut_preds', 'init_ego_fut_preds', 'memory_dynamics_loss'):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.head.train()
        memory['valid'][:] = False
        self.head.zero_grad(set_to_none=True)
        output = self.head(**inputs)
        (output['ego_fut_preds'].square().mean() + output['memory_dynamics_loss']).backward()
        for name, p in self.head.named_parameters():
            if name.startswith('memory_'):
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_no_history_configuration_runs_current_observation_and_future_rollout(self):
        config = head_config()
        config.memory_frames = 0
        head = build_head(config).eval()
        inputs = self.inputs()
        inputs['bev_inputs'] = inputs['bev_inputs'][:2]
        with torch.no_grad():
            output = head(**inputs)
        self.assertEqual(output['memory_valid_frames'].tolist(), [0, 0])
        self.assertEqual(output['memory_dynamics_loss'].item(), 0)
        self.assertEqual(output['future_residuals'].shape, (2, 6, 16, 256))
        self.assertTrue(torch.isfinite(output['ego_fut_preds']).all())

    def test_cached_visual_history_matches_independent_clips_across_commands_and_scenes(self):
        detector = SAWorldV2.__new__(SAWorldV2)
        nn.Module.__init__(detector)
        detector.pts_bbox_head = self.head
        detector.eval()
        observations = {}
        for scene in ('a', 'b'):
            for t in range(8):
                meta = dict(scene_token=scene, sample_idx=str(t), prev_idx=str(t - 1),
                            memory_timestamp=t * .5, memory_ego2global=np.eye(4))
                meta['memory_ego2global'][0, 3] = t
                observations[scene, t] = dict(img_metas=meta, can_bus=torch.randn(3, 18),
                                              img_inputs=[torch.randn(3, 256, 4, 4)] + [torch.zeros(1)] * 6)

        def extract(inputs, metas, **kwargs):
            return [inputs[0][0]], None

        def plan(bev, metas, *args, ego_fut_cmd, **kwargs):
            preds = self.head(bev, metas, cmd=ego_fut_cmd, ego_lcf_feat=torch.zeros(1, 1, 1, 9))
            return [preds['ego_fut_preds']], {}

        # Cold shard start, warm command change, a different scene, a gap, and replay.
        sequence = [('a', 4, [0, 1, 2, 3], 5), ('a', 5, [1, 2, 3, 4], 1),
                    ('b', 0, [], 1), ('b', 1, [0], 1), ('b', 6, [], 1),
                    ('a', 4, [0, 1, 2, 3], 5)]
        with torch.no_grad(), patch.object(detector, 'extract_img_feat', side_effect=extract) as encoder, \
                patch.object(detector, 'simple_test_pts', side_effect=plan):
            for scene, t, indices, expected_calls in sequence:
                current = observations[scene, t]
                entries = [None] * (4 - len(indices)) + [observations[scene, j] for j in indices]
                cmd = torch.eye(3)[t % 3].view(1, 1, 1, 3)
                args = dict(img_metas=[current['img_metas']], ego_fut_cmd=cmd,
                            img_inputs=[v[None] for v in current['img_inputs']],
                            can_bus=current['can_bus'][None], gt_bboxes_3d=None, gt_labels_3d=None,
                            map_gt_bboxes_3d=None, map_gt_labels_3d=None)
                reference = detector.simple_test(history_memory=[entries], **args)
                clip = DeferredTestHistory('dataset', {})
                for entry in entries:
                    clip.append(None if entry is None else dict(img_metas=entry['img_metas'],
                                                               can_bus=entry['can_bus']))

                def load(entry):
                    source = observations[scene, int(entry['img_metas']['sample_idx'])]
                    return dict(img_inputs=source['img_inputs'], memory_ego2global=source['img_metas']['memory_ego2global'])

                encoder.reset_mock()
                with patch.object(clip, 'load_observation', side_effect=load) as reads:
                    actual = detector.simple_test(history_memory=[clip], **args)
                torch.testing.assert_close(actual[0]['pts_bbox'], reference[0]['pts_bbox'], rtol=0, atol=0)
                self.assertEqual(encoder.call_count, expected_calls)
                self.assertEqual(reads.call_count, expected_calls - 1)
                self.assertLessEqual(len(detector._test_bevs), 4)
        detector.train()
        self.assertFalse(detector._test_bevs)

    def test_spatial_weights_are_shared_only_in_eval(self):
        inputs = self.inputs()
        with torch.no_grad(), patch.object(self.head.tokenfuser, 'spatial_mix',
                                           wraps=self.head.tokenfuser.spatial_mix) as mix:
            fast = self.head(**inputs)
            self.assertEqual(mix.call_count, 1)
            # Recompute the spatial weights at each horizon, as in the original decoder.
            with patch.object(self.head.tokenfuser, 'forward',
                              side_effect=lambda tokens, bev, mix: type(self.head.tokenfuser).forward(
                                  self.head.tokenfuser, tokens, bev)):
                reference = self.head(**inputs)
            torch.testing.assert_close(fast['ego_fut_preds'], reference['ego_fut_preds'], rtol=0, atol=0)
            torch.testing.assert_close(fast['pred_bev'], reference['pred_bev'], rtol=0, atol=0)
            mix.reset_mock()
            self.head.train()
            self.head(**inputs)
            self.assertEqual(mix.call_count, self.head.fut_ts)

    def test_shared_prediction_uses_physical_time_and_fits_a_known_trend(self):
        dynamics = ResidualDynamics(4)
        with torch.no_grad():
            dynamics.rate.weight.zero_()
            dynamics.rate.bias.fill_(.2)
        residual, state = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
        _, half_steps = dynamics.rollout(residual, state, 6, .5)
        _, full_steps = dynamics.rollout(residual, state, 3, 1.)
        torch.testing.assert_close(half_steps[:, 1::2], full_steps)
        torch.testing.assert_close(half_steps[:, -1], residual + .6)

        # The exact same rate operator is trained on observed transitions.
        with torch.no_grad():
            self.head.dynamics.rate.weight.zero_()
            self.head.dynamics.rate.bias.fill_(.2)
        memory = self.memory(batch=1)
        times = torch.tensor([0., .25, .75, 1.5], dtype=torch.float64)
        memory['timestamp'][0] = 1_500_000_000. + times
        memory['residual'][:] = (times.float() * .2).view(1, 4, 1, 1)
        current = torch.full((1, 16, 256), .4)
        selection = torch.full((1, 16, 16), 1/16.)
        with patch.object(self.head, 'read_history', return_value=memory['residual']):
            _, loss = self.head.encode_dynamics(current, current, selection, memory, [metadata()])
        self.assertLess(loss.item(), 1e-7)
        with patch.object(self.head, 'read_history', return_value=memory['residual'].flip(1)):
            _, wrong_order = self.head.encode_dynamics(current, current, selection, memory, [metadata()])
        self.assertGreater(wrong_order.item(), .01)

    def test_future_and_nonchronological_history_are_rejected(self):
        inputs = self.inputs()
        memory = inputs['bev_inputs'][2]
        memory['timestamp'][:, -1] = metadata()['memory_timestamp']
        with self.assertRaisesRegex(ValueError, 'strictly before'):
            self.head(**inputs)
        memory['timestamp'] = memory['timestamp'].flip(1) - .5
        with self.assertRaisesRegex(ValueError, 'chronological'):
            self.head(**inputs)

    def test_each_waypoint_uses_its_own_time_and_prior_reference(self):
        class Condition(nn.Module):
            def forward(self, query, prior):
                return query

        class Decoder(nn.Module):
            def forward(self, query, **kwargs):
                return query

        class Fuser(nn.Module):
            def spatial_mix(self, original):
                return original

            def forward(self, query, original, mix=None):
                return query[:, :1].expand_as(original)

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.references = []

            def forward(self, query, value, reference_points, **kwargs):
                self.references.append(reference_points.clone())
                return query + value[:1]

        self.head.action_mln, self.head.res_latent_decoder = Condition(), Decoder()
        self.head.tokenfuser, self.head.col_attn = Fuser(), Attention()
        query = torch.arange(18.).view(18, 1, 1).expand(18, 2, 256)
        states = torch.arange(1., 7.).view(1, 6, 1, 1).expand(2, 6, 16, 256)
        references = torch.randn(2, 3, 6, 1, 2)
        refined, _ = self.head.refine_waypoints(
            query, references, states, torch.zeros(16, 2, 256),
            torch.zeros(2, 16, 256), torch.zeros(1, 2, 12))
        expected = query.reshape(3, 6, 2, 256) + torch.arange(1., 7.).view(1, 6, 1, 1)
        torch.testing.assert_close(refined.reshape_as(expected), expected)
        for t, reference in enumerate(self.head.col_attn.references):
            torch.testing.assert_close(reference, references[:, :, t])


if __name__ == '__main__':
    unittest.main()
