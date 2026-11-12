"""SAWorld V3: history-conditioned planning with shared residual dynamics.

A single recurrent state summarizes observed residuals before action generation
and is rolled forward to construct the world at each waypoint time. This head
is standalone; the existing per-sample detached history protocol is reused.
"""
from math import isfinite

import torch
from torch import nn
from torch.nn import functional as F
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn import Linear
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.cnn.bricks.transformer import build_positional_encoding, build_transformer_layer_sequence
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmdet.core import reduce_mean
from mmdet.models import HEADS, build_loss

from .tokenlearner import TokenLearner, TokenFuser
from .action_dit import ActionDiT
from .rcsample import Mlp


class MLN(nn.Module):
    '''
    from "https://github.com/exiawsh/StreamPETR"
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256, use_ln=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.use_ln = use_ln

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        if self.use_ln:
            self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.init_weight()

    def init_weight(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        if self.use_ln:
            x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out

class SELayerMLP(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.mlp_reduce = nn.Linear(channels, channels)
        self.act1 = act_layer()
        self.mlp_expand = nn.Linear(channels, channels)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.mlp_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.mlp_expand(x_se)
        return x * self.gate(x_se)


class ResidualDynamics(nn.Module):
    """Observe residuals, then use the same state update during open-loop rollout.

    r_next = r + dt * rate(h). During history encoding, the next observation
    corrects this prediction; during rollout, the prediction is fed back.
    Time intervals are physical seconds, not sequence indices.
    """

    def __init__(self, channels):
        super().__init__()
        self.update = nn.GRUCell(channels + 1, channels)
        self.rate = nn.Linear(channels, channels)

    def observe(self, residual, state, dt):
        elapsed = dt[:, None, None].expand(*residual.shape[:-1], 1)
        inputs = torch.cat((residual, elapsed), dim=-1)
        return self.update(inputs.flatten(0, 1), state.flatten(0, 1)).view_as(state)

    def predict(self, residual, state, dt):
        return residual + dt[:, None, None] * self.rate(state)

    def rollout(self, residual, state, steps, dt):
        elapsed = residual.new_full((residual.shape[0],), dt)
        states, residuals = [], []
        for _ in range(steps):
            residual = self.predict(residual, state, elapsed)
            state = self.observe(residual, state, elapsed)
            states.append(state)
            residuals.append(residual)
        return torch.stack(states, dim=1), torch.stack(residuals, dim=1)


@HEADS.register_module()
class SAWorldHeadV3(BaseModule):
    """Use one causal residual state for decisions and future-world evolution."""

    def __init__(self,
                #  *args,
                 grid_config,
                 num_frames=3,
                 embed_dims=256,
                 in_channels=256,
                 num_reg_fcs=2,
                 positional_encoding=None,
                 bev_h=30,
                 bev_w=30,
                 fut_ts=6,
                 fut_mode=6,
                 num_scenes=16,
                 latent_decoder=None,
                 res_latent_decoder=None,
                 way_decoder=None,
                 ego_fut_mode=3,
                 loss_plan_reg=dict(type='L1Loss', loss_weight=0.25),
                 ego_lcf_feat_idx=None,
                 valid_fut_ts=6,
                 use_action=True,
                 action_num_tokens=3,
                 action_dit_layers=2,
                 action_dit_heads=8,
                 action_noise_scale=0.0,
                 action_ego_status_dim=9,
                 action_dropout=0.0,
                 action_flow_steps=None,
                 action_ffn_ratio=2,
                 action_flow_loss_weight=0.1,
                 action_source_std=1.0,
                 action_fixed_noise_seed=0,
                 action_behavior_scale=(10.0, 5.0),
                 action_traj_dt=None,
                 action_traj_is_displacement=True,
                 action_velocity_indices=None,
                 action_velocity_from_history=True,
                 action_history_dt=0.5,
                 action_history_is_displacement=True,
                 memory_frames=4,
                 memory_heads=8,
                 memory_max_gap=1.0,
                 dynamics_loss_weight=1.0,
                 **kwargs):
        super().__init__()
        if kwargs.get('explicit_time_condition', False) or kwargs.get('use_action_residual', False):
            raise ValueError('V3 uses shared residual dynamics; disable separate time/action-residual ablations')
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.embed_dims = embed_dims
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.latent_decoder = latent_decoder
        self.res_latent_decoder = res_latent_decoder
        self.way_decoder = way_decoder
        self.positional_encoding = positional_encoding
        self.ego_fut_mode = ego_fut_mode
        self.ego_lcf_feat_idx = ego_lcf_feat_idx
        self.valid_fut_ts = valid_fut_ts
        self.num_scenes = num_scenes
        self.num_frames = num_frames
        self.grid_min = torch.tensor([grid_config['x'][0], grid_config['y'][0]])
        self.grid_max = torch.tensor([grid_config['x'][1], grid_config['y'][1]])
        self.grid_size = torch.tensor([grid_config['x'][2], grid_config['y'][2]])
        self.use_action = use_action
        if not self.use_action:
            raise ValueError('SAWorldHeadV3 requires use_action=True')
        # ActionDiT 只读取观测场景；专家轨迹在 loss() 中单独监督 flow。
        if self.use_action :
            self.action_dit_cfg = dict(
                embed_dims=embed_dims,
                num_tokens=action_num_tokens,
                num_layers=action_dit_layers,
                num_heads=action_dit_heads,
                noise_scale=action_noise_scale,
                ego_status_dim=action_ego_status_dim,
                dropout=action_dropout,
                ffn_ratio=action_ffn_ratio,
                flow_steps=action_flow_steps,
                source_std=action_source_std,
                fixed_noise_seed=action_fixed_noise_seed,
                behavior_scale=action_behavior_scale)
            self.action_flow_loss_weight = float(action_flow_loss_weight)
            self.action_traj_dt = action_traj_dt
            self.action_traj_is_displacement = bool(action_traj_is_displacement)
            self.action_velocity_indices = action_velocity_indices
            self.action_velocity_from_history = bool(action_velocity_from_history)
            self.action_history_dt = action_history_dt
            self.action_history_is_displacement = bool(action_history_is_displacement)
            if not isfinite(self.action_flow_loss_weight) or self.action_flow_loss_weight < 0:
                raise ValueError('action_flow_loss_weight must be finite and nonnegative')
            if self.action_flow_loss_weight > 0 and action_num_tokens > fut_ts:
                raise ValueError('action_num_tokens must not exceed fut_ts for behavior targets')
            if action_traj_dt is not None and (not isfinite(action_traj_dt) or action_traj_dt <= 0):
                raise ValueError('action_traj_dt must be positive, in seconds')
            if not isfinite(action_history_dt) or action_history_dt <= 0:
                raise ValueError('action_history_dt must be positive, in seconds')
            if action_velocity_indices is not None and (
                    len(action_velocity_indices) != 2 or
                    any(not isinstance(index, int) or index < 0 for index in action_velocity_indices) or
                    action_velocity_indices[0] == action_velocity_indices[1]):
                raise ValueError('action_velocity_indices must contain two distinct nonnegative indices')
        else:
            self.action_flow_loss_weight = 0
        self._init_layers()
        self.loss_plan_reg = build_loss(loss_plan_reg)
        self.loss_plan_reg_init = build_loss(loss_plan_reg)
        if not isinstance(memory_frames, int) or memory_frames < 0:
            raise ValueError('memory_frames must be a nonnegative integer')
        if not isfinite(memory_max_gap) or memory_max_gap <= 0:
            raise ValueError('memory_max_gap must be positive, in seconds')
        if not isfinite(dynamics_loss_weight) or dynamics_loss_weight < 0:
            raise ValueError('dynamics_loss_weight must be finite and nonnegative')
        if action_traj_dt is None or not isfinite(action_traj_dt) or action_traj_dt <= 0:
            raise ValueError('V3 requires action_traj_dt in seconds for the future rollout')
        if num_frames < 2:
            raise ValueError('Residual dynamics requires at least two observed frames')
        self.memory_frames = memory_frames
        self.memory_max_gap = float(memory_max_gap)
        self.dynamics_loss_weight = float(dynamics_loss_weight)
        self.dynamics = ResidualDynamics(embed_dims)
        self.scene_state_fusion = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims), nn.LayerNorm(embed_dims))
        if memory_frames:
            if memory_heads < 1 or embed_dims % memory_heads:
                raise ValueError('memory_heads must divide embed_dims')
            self.memory_heads = memory_heads
            self.memory_query = nn.Sequential(nn.LayerNorm(embed_dims), nn.Linear(embed_dims, embed_dims))
            self.memory_key = nn.Sequential(nn.LayerNorm(embed_dims), nn.Linear(embed_dims, embed_dims))
            self.memory_position = nn.Sequential(
                nn.Linear(3, embed_dims), nn.ReLU(), nn.Linear(embed_dims, embed_dims))
        self.last_memory_size = None


    def _init_layers(self):
        ego_fut_decoder = []
        ego_fut_dec_in_dim = self.embed_dims + len(self.ego_lcf_feat_idx) \
            if self.ego_lcf_feat_idx is not None else self.embed_dims
        for _ in range(self.num_reg_fcs):
            ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            ego_fut_decoder.append(nn.ReLU())
        ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, 2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)
        init_ego_fut_decoder = []
        for _ in range(self.num_reg_fcs):
            init_ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            init_ego_fut_decoder.append(nn.ReLU())
        init_ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, 2))
        self.init_ego_fut_decoder = nn.Sequential(*init_ego_fut_decoder)

        self.navi_embedding = nn.Embedding(3, self.embed_dims)
        self.navi_se = SELayerMLP(self.embed_dims)
        self.canbus_mlp = Mlp(18, self.embed_dims, self.embed_dims)
        self.canbus_se = SELayerMLP(self.embed_dims)
        self.bev_fusion_conv = ConvModule(self.in_channels * self.num_frames, self.in_channels,
                                          kernel_size=3, padding=1)

        self.way_point = nn.Embedding(self.ego_fut_mode*self.fut_ts, self.embed_dims * 2)
        self.tokenlearner = TokenLearner(self.num_scenes, self.embed_dims * 2)
        self.tokenfuser = TokenFuser(self.num_scenes, self.embed_dims)
        self.latent_decoder = build_transformer_layer_sequence(self.latent_decoder)
        self.way_decoder = build_transformer_layer_sequence(self.way_decoder)
        self.res_latent_decoder = build_transformer_layer_sequence(self.res_latent_decoder)
        if self.use_action:
            self.action_dit = ActionDiT(**self.action_dit_cfg)
        self.col_attn = MultiScaleDeformableAttention(self.embed_dims,
                                            num_points=8, num_levels=1)
        self.action_mln = MLN(self.fut_ts * 2, self.embed_dims)
        self.positional_encoding = build_positional_encoding(
            self.positional_encoding)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        if self.latent_decoder is not None:
            for p in self.latent_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if self.way_decoder is not None:
            for p in self.way_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if self.res_latent_decoder is not None:
            for p in self.res_latent_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    @force_fp32(apply_to=('bev_inputs',))
    def forward(self, bev_inputs, img_metas, prev_bev=None, only_bev=False,
                ego_his_trajs=None, ego_lcf_feat=None, cmd=None, ego_velocity=None):
        memory = bev_inputs[2] if len(bev_inputs) > 2 else None
        (bev_embed, bev_navi_embed, scene_tokens,
         per_frame, selection, residuals) = self._encode_scene(bev_inputs, cmd)
        scene, position = scene_tokens.split(self.embed_dims, dim=-1)
        current_residual = residuals[0, ..., :self.embed_dims]
        current_scene = per_frame[0, ..., :self.embed_dims]
        state, dynamics_loss = self.encode_dynamics(
            current_scene, current_residual, selection, memory, img_metas)

        # Memory defines the current decision state, before either action or prior.
        latent_query = self.scene_state_fusion(torch.cat((scene, state), dim=-1))
        latent_query, latent_pos = latent_query.transpose(0, 1), position.transpose(0, 1)
        latent_query = self.latent_decoder(
            query=latent_query, key=latent_query, value=latent_query,
            query_pos=latent_pos, key_pos=latent_pos)
        action_scene = (latent_query + latent_pos).transpose(0, 1).contiguous()
        action_query, action_aux = self.action_dit(
            action_scene, cmd=cmd, ego_status=ego_lcf_feat, return_aux=True)
        scene_action_query = torch.cat((latent_query, action_query.transpose(0, 1)), dim=0)
        scene_action_pos = torch.cat((latent_pos, torch.zeros_like(action_query.transpose(0, 1))), dim=0)

        batch = scene.shape[0]
        wp_pos, way_point = self.way_point.weight.to(scene).split(self.embed_dims, dim=-1)
        wp_pos = wp_pos[:, None].expand(-1, batch, -1)
        way_point = way_point[:, None].expand(-1, batch, -1)
        way_point = self.way_decoder(
            query=way_point, key=scene_action_query, value=scene_action_query,
            query_pos=wp_pos, key_pos=scene_action_pos)
        init_trajs = self.init_ego_fut_decoder(way_point).transpose(0, 1).reshape(
            batch, self.ego_fut_mode, self.fut_ts, 2)
        command = self.action_dit._command_indices(cmd, batch, scene.device)
        batch_index = torch.arange(batch, device=scene.device)
        init_wp_vector = init_trajs[batch_index, command].reshape(batch, -1).unsqueeze(0)
        reference_points = init_trajs.cumsum(dim=2)
        reference_points = ((reference_points - self.grid_min.to(scene)) /
                            self.grid_size.to(scene) / 200).unsqueeze(-2)

        # No future observations enter this rollout. All horizons share parameters.
        future_states, future_residuals = self.dynamics.rollout(
            current_residual, state, self.fut_ts, self.action_traj_dt)
        way_point, pred_bev = self.refine_waypoints(
            way_point, reference_points, future_states, latent_pos,
            bev_navi_embed, init_wp_vector)
        trajectories = self.ego_fut_decoder(way_point).transpose(0, 1).reshape(
            batch, self.ego_fut_mode, self.fut_ts, 2)
        outputs = dict(
            bev_embed=bev_embed, pred_bev=pred_bev,
            scene_query=latent_query, scene_action_query=scene_action_query,
            action_query=action_query,
            action_behavior_coefficients=action_aux['behavior_coefficients'],
            wp_vector=trajectories[batch_index, command].reshape(batch, -1).unsqueeze(0),
            ego_fut_preds=trajectories, init_ego_fut_preds=init_trajs,
            dynamics_state=state, future_residuals=future_residuals,
            memory_dynamics_loss=dynamics_loss, memory_valid_frames=self.last_memory_size)
        if self.training and self.action_flow_loss_weight > 0:
            outputs.update(
                action_flow_scene_tokens=action_scene, action_flow_cmd=command,
                action_flow_ego_status=ego_lcf_feat,
                action_flow_initial_velocity=self._action_initial_velocity(
                    batch, action_scene, ego_his_trajs, ego_lcf_feat, ego_velocity))
        return outputs

    def refine_waypoints(self, way_point, reference_points, future_states,
                         position, bev_navi_embed, prior):
        """Each waypoint samples its matching future BEV at its prior location."""
        batch, steps = future_states.shape[:2]
        query = way_point.reshape(self.ego_fut_mode, steps, batch, self.embed_dims)
        spatial_shapes = torch.tensor([[self.bev_h, self.bev_w]], device=way_point.device)
        level_start_index = torch.zeros(1, dtype=torch.long, device=way_point.device)
        refined = []
        for step in range(steps):
            # Existing action/trajectory conditioning belongs to the world decoder.
            world_query = self.action_mln(future_states[:, step].transpose(0, 1), prior)
            world_query = self.res_latent_decoder(
                query=world_query, key=world_query, value=world_query,
                query_pos=position, key_pos=position)
            pred_bev = self.tokenfuser(world_query.transpose(0, 1), bev_navi_embed) + bev_navi_embed
            refined.append(self.col_attn(
                query=query[:, step], key=pred_bev.transpose(0, 1), value=pred_bev.transpose(0, 1),
                reference_points=reference_points[:, :, step],
                spatial_shapes=spatial_shapes, level_start_index=level_start_index))
        # Stack by mode, then time: exactly the prior/label ordering.
        return torch.stack(refined, dim=1).reshape_as(way_point), pred_bev

    def _action_initial_velocity(self, batch_size, reference, ego_his_trajs,
                                 ego_lcf_feat, ego_velocity):
        """从显式速度或已观测历史获取当前轨迹坐标系下的速度，单位 m/s。"""
        if ego_velocity is not None:
            if ego_velocity.shape != (batch_size, 2):
                raise ValueError('ego_velocity must be [B, 2], in the trajectory frame, m/s')
            return ego_velocity.detach().to(reference)
        if self.action_velocity_from_history:
            if (ego_his_trajs is None or ego_his_trajs.dim() < 3 or
                    ego_his_trajs.shape[0] != batch_size or ego_his_trajs.shape[-1] != 2):
                raise ValueError('action velocity requires current-frame ego_his_trajs [B,...,H,2]')
            history_steps = ego_his_trajs.shape[-2]
            if history_steps < 1 or ego_his_trajs.numel() != batch_size * history_steps * 2:
                raise ValueError('ego_his_trajs must contain one observed history per sample')
            history = ego_his_trajs.detach().reshape(batch_size, history_steps, 2).to(reference)
            if not torch.isfinite(history).all():
                raise ValueError('observed trajectory history must be finite')
            if self.action_history_is_displacement:
                displacement = history[:, -1]
            else:
                if history_steps < 2:
                    raise ValueError('position history needs two points ending at current time')
                displacement = history[:, -1] - history[:, -2]
            return displacement / self.action_history_dt
        if self.action_velocity_indices is None or ego_lcf_feat is None:
            raise ValueError('provide ego_velocity, observed history, or action_velocity_indices')
        status = ego_lcf_feat.reshape(batch_size, -1)
        if max(self.action_velocity_indices) >= status.shape[-1]:
            raise ValueError('action_velocity_indices exceed ego_lcf_feat width')
        return status[:, list(self.action_velocity_indices)].detach().to(reference)

    def _action_flow_loss(self, preds_dicts, future_traj, future_mask, command):
        """独立的行为空间 flow matching；只有完整有效的未来参与监督。"""
        if self.action_traj_dt is None:
            raise ValueError('set action_traj_dt to the future trajectory sampling interval')
        if 'action_flow_scene_tokens' not in preds_dicts:
            raise ValueError('action flow loss requires a training-mode head.forward()')
        scene_tokens = preds_dicts['action_flow_scene_tokens']
        batch_size = scene_tokens.shape[0]
        command_indices = self.action_dit._command_indices(command, batch_size, scene_tokens.device)
        if not torch.equal(command_indices, preds_dicts['action_flow_cmd']):
            raise ValueError('forward cmd and loss ego_fut_cmd must agree')
        if not ((future_mask == 0) | (future_mask == 1)).all():
            raise ValueError('ego_fut_masks must contain only valid=1 / invalid=0')
        target, valid = self.action_dit.build_behavior_targets(
            future_traj=future_traj,
            initial_velocity=preds_dicts['action_flow_initial_velocity'],
            dt=self.action_traj_dt,
            future_mask=future_mask.bool(),
            trajectory_is_displacement=self.action_traj_is_displacement)
        raw_loss = self.action_dit.flow_matching_loss(
            scene_tokens=scene_tokens,
            behavior_targets=target,
            cmd=command_indices,
            ego_status=preds_dicts['action_flow_ego_status'],
            behavior_valid=valid)
        # 对齐各 rank 有效样本数，避免 DDP 对局部均值等权平均造成偏差。
        local_count = valid.float().sum()
        mean_count = reduce_mean(local_count.detach())
        ddp_weight = local_count / mean_count.clamp_min(1e-6)
        return self.action_flow_loss_weight * ddp_weight * raw_loss

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             map_gt_bboxes_list,
             map_gt_labels_list,
             preds_dicts,
             ego_fut_gt,
             ego_fut_masks,
             ego_fut_cmd,
             gt_attr_labels,
             gt_bboxes_ignore=None,
             map_gt_bboxes_ignore=None,
             img_metas=None):

        ego_fut_preds = preds_dicts['ego_fut_preds']
        init_ego_fut_preds = preds_dicts['init_ego_fut_preds']
        loss_dict = dict()

        # Planning Loss
        ego_fut_gt = ego_fut_gt.squeeze(1)
        ego_fut_masks = ego_fut_masks.squeeze(1).squeeze(1)
        ego_fut_cmd = ego_fut_cmd.squeeze(1).squeeze(1)

        ego_fut_gt = ego_fut_gt.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
        loss_plan_l1_weight = ego_fut_cmd[..., None, None] * ego_fut_masks[:, None, :, None]
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1, 1, 1, 2)

        loss_plan_l1 = self.loss_plan_reg(
            ego_fut_preds,
            ego_fut_gt,
            loss_plan_l1_weight
        )

        loss_plan_l1_init = self.loss_plan_reg_init(
            init_ego_fut_preds,
            ego_fut_gt,
            loss_plan_l1_weight
        )

        loss_dict['loss_plan_reg'] = loss_plan_l1
        loss_dict['loss_plan_reg_init'] = loss_plan_l1_init

        if self.action_flow_loss_weight > 0:
            loss_dict['loss_action_flow'] = self._action_flow_loss(
                preds_dicts, ego_fut_gt[:, 0], ego_fut_masks, ego_fut_cmd)

        if self.dynamics_loss_weight > 0:
            loss_dict['loss_memory_dynamics'] = (
                preds_dicts['memory_dynamics_loss'] * self.dynamics_loss_weight)

        return loss_dict

    def _encode_scene(self, bev_inputs, cmd):
        """Shared observed-scene extraction for the anchor and its past frames."""
        bev_feats, can_bus_infos = bev_inputs[:2]
        bt, c, h, w = bev_feats.shape
        bs = bt // self.num_frames
        dtype = bev_feats[0].dtype
        device = bev_feats[0].device
        can_bus_infos = self.canbus_mlp(can_bus_infos.permute(1, 0, 2)).view(bt, 1, self.in_channels)
        bev_embed = self.canbus_se(bev_feats.view(bt, c, h*w).permute(0, 2, 1), can_bus_infos)
        bev_embed_single = bev_embed.clone()
        bev_embed = bev_embed.permute(0, 2, 1).view(self.num_frames, bs, c, h, w).permute(1, 0, 2, 3, 4)
        bev_embed = self.bev_fusion_conv(bev_embed.reshape(bs, self.num_frames * c, h, w))
        bev_feats = bev_feats.view(self.num_frames, bs, c, h, w)


        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_feats.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        pos_embd = bev_pos.flatten(2).permute(0, 2, 1)
        bev_embed = bev_embed.reshape(bs, c, h * w).permute(0, 2, 1)

        navi_embed = []
        for bidx in range(bs):
            cmd_idx = torch.nonzero(cmd[bidx, 0, 0])[0, 0]
            navi_embed.append(self.navi_embedding.weight[cmd_idx][None, None])
        navi_embed = torch.cat(navi_embed, dim=0)

        bev_navi_embed = self.navi_se(bev_embed, navi_embed)

        bev_query = torch.cat((bev_navi_embed, pos_embd), -1)

        learned_latent_query, selected = self.tokenlearner(bev_query)
        res_selected = selected  # One spatial basis for scene, residuals and memory.
        bev_embed_single = torch.cat((bev_embed_single, pos_embd.repeat(self.num_frames,1,1)), -1)
        bev_embed_single = torch.einsum('bsi,bic->bsc', res_selected.repeat(self.num_frames,1,1), bev_embed_single)\
                            .view(self.num_frames, bs, learned_latent_query.shape[1],learned_latent_query.shape[2])
        res_latent_query_all = bev_embed_single[:-1] - bev_embed_single[1:]

        return (bev_embed, bev_navi_embed, learned_latent_query,
                bev_embed_single, res_selected, res_latent_query_all)

    @torch.no_grad()
    def encode_memory(self, bev_inputs, cmd, img_metas):
        """Encode observations only; no action rollout or world prediction."""
        _, _, _, per_frame, selection, residuals = self._encode_scene(bev_inputs, cmd)
        scene = per_frame[0, ..., :self.embed_dims]
        residual = residuals[0, ..., :self.embed_dims]
        points = self._reference_points(selection)
        times, poses = self._memory_metadata(img_metas, residual)
        global_points = torch.einsum('bij,bnj->bni', poses[:, :3, :3], points.double())
        global_points = global_points + poses[:, None, :3, 3]
        return dict(scene=scene.detach(), residual=residual.detach(),
                    points=global_points.detach(), timestamp=times.detach())

    def _memory_metadata(self, img_metas, reference):
        required = ('scene_token', 'sample_idx', 'prev_idx',
                    'memory_timestamp', 'memory_ego2global')
        if len(img_metas) != reference.shape[0]:
            raise ValueError('memory requires one metadata dict per batch sample')
        for meta in img_metas:
            missing = [key for key in required if key not in meta]
            if missing:
                raise ValueError('V3 memory metadata missing {}. Use the memory data pipeline.'.format(missing))
        times = torch.tensor([meta['memory_timestamp'] for meta in img_metas],
                             device=reference.device, dtype=torch.float64)
        poses = torch.stack([torch.as_tensor(meta['memory_ego2global'],
                                             device=reference.device, dtype=torch.float64)
                             for meta in img_metas])
        if poses.shape != (len(img_metas), 4, 4) or not torch.isfinite(poses).all() or not torch.isfinite(times).all():
            raise ValueError('memory needs finite timestamps and [B,4,4] BEV-to-global poses')
        return times, poses

    def align_memory(self, memory, img_metas, residual):
        if memory is None:
            raise ValueError('V3 requires sample history; use SAWorldV2 and the memory dataset')
        times, poses = self._memory_metadata(img_metas, residual)
        valid = memory['valid']
        expected = (residual.shape[0], self.memory_frames, self.num_scenes, self.embed_dims)
        if memory['residual'].shape != expected or valid.shape != expected[:2]:
            raise ValueError('dataset/head memory_frames or token shapes do not match')
        ages = times[:, None] - memory['timestamp']
        if ((ages <= 0) & valid).any():
            raise ValueError('Memory must contain only observations strictly before the anchor')
        inverse = torch.linalg.inv(poses)
        points = torch.einsum('bij,bknj->bkni', inverse[:, :3, :3], memory['points'])
        points = points + inverse[:, None, None, :3, 3]
        self.last_memory_size = valid.sum(dim=1).detach()
        return points.to(residual), ages.masked_fill(~valid, 0).to(residual)

    def _reference_points(self, selection):
        """Coarse token anchors in the existing RCSample BEV convention.

        RCSample.create_grid_infos uses x-bound rows and y-bound columns,
        then stacks (column, row, 0). Keep that convention for memory geometry.
        """
        lower, upper = self.grid_min.to(selection), self.grid_max.to(selection)
        rows = (torch.arange(self.bev_h, device=selection.device, dtype=selection.dtype) + .5)
        cols = (torch.arange(self.bev_w, device=selection.device, dtype=selection.dtype) + .5)
        rows = lower[0] + rows * (upper[0] - lower[0]) / self.bev_h
        cols = lower[1] + cols * (upper[1] - lower[1]) / self.bev_w
        yy, xx = torch.meshgrid(rows, cols, indexing='ij')
        grid = torch.stack((xx, yy, torch.zeros_like(xx)), dim=-1).flatten(0, 1)
        return torch.einsum('bnp,pc->bnc', selection, grid)

    def read_history(self, current_scene, points, history_points, memory):
        """Associate each past frame to current slots without mixing time steps.

        Attention estimates correspondence; values stay in the observed residual
        feature space so consecutive residuals have a common prediction target.
        The reference points are coarse region centers, not tracked objects.
        """
        valid = memory['valid'][:, :, None, None]
        scenes = memory['scene'].detach().masked_fill(~valid, 0)
        residuals = memory['residual'].detach().masked_fill(~valid, 0)
        history_points = history_points.masked_fill(~valid, 0)
        query = self.memory_query(current_scene) + self.memory_position(points / 50.)
        key = self.memory_key(scenes) + self.memory_position(history_points / 50.)
        batch, frames, tokens, channels = key.shape
        heads, width = self.memory_heads, channels // self.memory_heads
        query = query.reshape(batch, tokens, heads, width).permute(0, 2, 1, 3).unsqueeze(1)
        key = key.reshape(batch, frames, tokens, heads, width).permute(0, 1, 3, 2, 4)
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) / width ** .5
        weights = scores.softmax(dim=-1).mean(dim=2).to(residuals)
        return torch.matmul(weights, residuals).masked_fill(~valid, 0)

    def encode_dynamics(self, current_scene, residual, selection, memory, img_metas):
        """Fit observed transitions and assimilate the current residual last."""
        batch = residual.shape[0]
        if self.memory_frames:
            points, ages = self.align_memory(memory, img_metas, residual)
            history = self.read_history(
                current_scene, self._reference_points(selection), points, memory)
            observations = torch.cat((history, residual[:, None]), dim=1)
            times = torch.cat((-ages, ages.new_zeros(batch, 1)), dim=1)
            valid = torch.cat((memory['valid'], memory['valid'].new_ones(batch, 1)), dim=1)
        else:
            observations = residual[:, None]
            times = residual.new_zeros(batch, 1)
            valid = torch.ones(batch, 1, dtype=torch.bool, device=residual.device)
            self.last_memory_size = torch.zeros(batch, dtype=torch.long, device=residual.device)

        state, previous = torch.zeros_like(residual), torch.zeros_like(residual)
        previous_time = residual.new_zeros(batch)
        seen = valid.new_zeros(batch)
        error_sum, pair_count = residual.sum() * 0., residual.new_zeros(())
        for index in range(observations.shape[1]):
            keep = valid[:, index]
            has_pair = keep & seen
            dt = torch.where(has_pair, times[:, index] - previous_time, previous_time.new_zeros(batch))
            if ((dt <= 0) & has_pair).any() or (dt > self.memory_max_gap).any():
                raise ValueError('Residual history must be chronological with gaps <= memory_max_gap')
            observed = observations[:, index].masked_fill(~keep[:, None, None], 0)
            predicted = self.dynamics.predict(previous, state, dt)
            error = F.l1_loss(predicted, observed.detach(), reduction='none').mean(dim=(1, 2))
            error_sum = error_sum + (error * has_pair.to(error)).sum()
            pair_count = pair_count + has_pair.sum()
            updated = self.dynamics.observe(observed, state, dt)
            state = torch.where(keep[:, None, None], updated, state)
            previous = torch.where(keep[:, None, None], observed, previous)
            previous_time = torch.where(keep, times[:, index], previous_time)
            seen = seen | keep
        # DDP averages gradients; normalize by mean per-rank count for a global
        # mean over valid observed transitions, including heterogeneous clips.
        denominator = reduce_mean(pair_count.detach()) if self.training else pair_count
        return state, error_sum / denominator.clamp_min(1)

