
import math
import warnings
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _modulate(x: torch.Tensor, shift: torch.Tensor,
              scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ActionQueryBlock(nn.Module):
    

    def __init__(self, embed_dims: int, num_heads: int = 8,
                 dropout: float = 0.0, ffn_ratio: int = 2):
        super().__init__()
        if num_heads < 1 or embed_dims % num_heads:
            raise ValueError('embed_dims must be divisible by positive num_heads')
        hidden_dims = int(embed_dims * ffn_ratio)
        if hidden_dims < 1 or not 0.0 <= dropout < 1.0:
            raise ValueError('invalid ffn_ratio or dropout')
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.dropout = dropout
        self.norm_attn = nn.LayerNorm(embed_dims, elementwise_affine=False)
        self.scene_norm = nn.LayerNorm(embed_dims)
        self.query_proj = nn.Linear(embed_dims, embed_dims)
        self.action_kv = nn.Linear(embed_dims, embed_dims * 2)
        self.scene_kv = nn.Linear(embed_dims, embed_dims * 2)
        self.attn_out = nn.Linear(embed_dims, embed_dims)
        self.norm_ffn = nn.LayerNorm(embed_dims, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, hidden_dims), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dims, embed_dims),
            nn.Dropout(dropout))
        self.ada_ln = nn.Sequential(
            nn.SiLU(), nn.Linear(embed_dims, embed_dims * 4))
      
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dims).transpose(1, 2)

    def prepare_scene(self, scene_tokens: torch.Tensor,
                      padding_mask: Optional[torch.Tensor] = None
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        if padding_mask is not None:
            # Mask before projection so padding NaNs cannot contaminate values.
            scene_tokens = scene_tokens.masked_fill(padding_mask.unsqueeze(-1), 0)
        key, value = self.scene_kv(self.scene_norm(scene_tokens)).chunk(2, -1)
        return self._heads(key), self._heads(value)

    def forward(self, action_tokens: torch.Tensor,
                scene_kv: Tuple[torch.Tensor, torch.Tensor],
                condition: torch.Tensor,
                scene_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        shift_a, scale_a, shift_f, scale_f = self.ada_ln(condition).chunk(4, -1)
        hidden = _modulate(self.norm_attn(action_tokens), shift_a, scale_a)
        query = self._heads(self.query_proj(hidden))
        action_key, action_value = self.action_kv(hidden).chunk(2, -1)
        key = torch.cat((self._heads(action_key), scene_kv[0]), dim=2)
        value = torch.cat((self._heads(action_value), scene_kv[1]), dim=2)
       
        autocast_off = (torch.autocast(device_type=query.device.type, enabled=False)
                        if hasattr(torch, 'autocast') else
                        torch.cuda.amp.autocast(enabled=False))
        with autocast_off:
            score = torch.matmul(query.float(), key.float().transpose(-1, -2))
            score = score * (self.head_dims ** -0.5)
            if scene_padding_mask is not None:
                action_mask = scene_padding_mask.new_zeros(
                    scene_padding_mask.shape[0], action_tokens.shape[1])
                mask = torch.cat((action_mask, scene_padding_mask), dim=1)
                score = score.masked_fill(mask[:, None, None, :], float('-inf'))
            weights = F.dropout(score.softmax(-1), self.dropout, self.training)
        readout = torch.matmul(weights.to(value.dtype), value)
        readout = readout.transpose(1, 2).reshape_as(action_tokens)
        action_tokens = action_tokens + self.attn_out(readout)
        hidden = _modulate(self.norm_ffn(action_tokens), shift_f, scale_f)
        return action_tokens + self.ffn(hidden)


class ActionDiT(nn.Module):
    """Generate action queries using behavior-grounded conditional flow.

    The first two output channels are normalized DCT motion-change coefficients;
    the other channels are learned scene-conditioned features. The full query
    has embed_dims channels. Coefficients do not directly prescribe the final
    trajectory, which is decoded jointly with scene tokens by the caller.
    """

    def __init__(self, embed_dims: int, num_tokens: int,
                 num_layers: int = 2, num_heads: int = 8,
                 noise_scale: float = 0.0, ego_status_dim: int = 9,
                 dropout: float = 0.0, ffn_ratio: int = 2,
                 num_commands: int = 3, *,
                 flow_steps: Optional[int] = None,
                 source_std: float = 1.0, fixed_noise_seed: int = 0,
                 behavior_scale: Tuple[float, float] = (10.0, 5.0)):
        super().__init__()
        if embed_dims < 4 or num_heads < 1 or embed_dims % num_heads:
            raise ValueError('embed_dims >= 4 must be divisible by num_heads')
        if num_tokens < 1 or num_layers < 1:
            raise ValueError('num_tokens and num_layers must be positive')
        if num_commands < 1 or ego_status_dim < 1:
            raise ValueError('num_commands and ego_status_dim must be positive')
        if source_std <= 0 or not math.isfinite(source_std):
            raise ValueError('source_std must be finite and positive')
        self.flow_steps = num_layers if flow_steps is None else flow_steps
        if not isinstance(self.flow_steps, int) or self.flow_steps < 1:
            raise ValueError('flow_steps must be a positive integer')
        if noise_scale != 0:
            warnings.warn('noise_scale query jitter is ignored; configure '
                          'source_std for flow-source noise instead.',
                          FutureWarning, stacklevel=2)
        self.embed_dims = embed_dims
        self.num_tokens = num_tokens
        self.num_commands = num_commands
        self.ego_status_dim = ego_status_dim
        self.noise_scale = float(noise_scale)
        self.source_std = float(source_std)
        scales = torch.as_tensor(behavior_scale, dtype=torch.float32)
        if scales.shape != (2,) or not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError('behavior_scale must contain two finite positive m/s scales')
        self.register_buffer('behavior_scale', scales)
        generator = torch.Generator(device='cpu')
        generator.manual_seed(fixed_noise_seed)
        self.register_buffer('fixed_noise', torch.randn(
            1, num_tokens, 2, generator=generator) * source_std)
        frequency = torch.exp(-math.log(10000.0) * torch.arange(
            embed_dims // 2, dtype=torch.float32) / max(embed_dims // 2 - 1, 1))
        self.register_buffer('time_frequency', frequency, persistent=False)
        self.action_query = nn.Parameter(torch.randn(1, num_tokens, embed_dims) * 0.02)
        self.behavior_embed = nn.Linear(2, embed_dims)
        self.command_embed = nn.Embedding(num_commands, embed_dims)
        self.ego_status_proj = nn.Sequential(
            nn.Linear(ego_status_dim, embed_dims), nn.LayerNorm(embed_dims),
            nn.SiLU(), nn.Linear(embed_dims, embed_dims))
        self.condition_norm = nn.LayerNorm(embed_dims)
        self.block = ActionQueryBlock(embed_dims, num_heads, dropout, ffn_ratio)
        self.out_norm = nn.LayerNorm(embed_dims)
        self.velocity_head = nn.Linear(embed_dims, 2)
        nn.init.normal_(self.velocity_head.weight, std=0.02)
        nn.init.zeros_(self.velocity_head.bias)

    def _command_indices(self, command: Optional[torch.Tensor],
                         batch_size: int, device: torch.device) -> torch.Tensor:
        if command is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        command = command.to(device=device)
        if command.numel() % batch_size:
            raise ValueError('cmd must provide one entry/vector per sample')
        command = command.reshape(batch_size, -1)
        if command.shape[1] == 1:
            indices = command[:, 0].long()
            if ((indices < 0) | (indices >= self.num_commands)).any():
                raise ValueError('command index outside [0, num_commands)')
            if command.is_floating_point() and (command[:, 0] != indices).any():
                raise ValueError('scalar commands must be integer indices')
            return indices
        if command.shape[1] < self.num_commands:
            raise ValueError('command vectors need at least num_commands entries')
        return command[:, :self.num_commands].argmax(dim=-1).long()

    def _ego_status(self, ego_status: Optional[torch.Tensor], batch_size: int,
                    device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if ego_status is None:
            return torch.zeros(batch_size, self.ego_status_dim, device=device, dtype=dtype)
        status = ego_status.reshape(batch_size, -1).to(device=device, dtype=dtype)
        # Preserve the original padding/truncation convention.
        if status.shape[1] < self.ego_status_dim:
            status = F.pad(status, (0, self.ego_status_dim - status.shape[1]))
        return status[:, :self.ego_status_dim]

    def _prepare(self, scene: torch.Tensor, cmd: Optional[torch.Tensor],
                 ego: Optional[torch.Tensor], mask: Optional[torch.Tensor]):
        if scene.dim() != 3 or scene.shape[-1] != self.embed_dims:
            raise ValueError('scene_tokens must be [B, N, embed_dims]')
        b, n, _ = scene.shape
        if b < 1 or n < 1 or not scene.is_floating_point():
            raise ValueError('scene_tokens need nonempty batch/scene and floating dtype')
        if mask is not None:
            if mask.shape != (b, n) or mask.dtype != torch.bool:
                raise ValueError('scene_key_padding_mask must be bool [B, N], True=ignore')
            mask = mask.to(scene.device)
        indices = self._command_indices(cmd, b, scene.device)
        status = self._ego_status(ego, b, scene.device, scene.dtype)
        condition = self.command_embed(indices) + self.ego_status_proj(status)
        condition = self.condition_norm(condition).to(scene.dtype)
        return condition, self.block.prepare_scene(scene, mask), mask

    def _field(self, z: torch.Tensor, tau: torch.Tensor,
               condition: torch.Tensor, scene_kv: Tuple[torch.Tensor, torch.Tensor],
               mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        angles = tau.float().unsqueeze(-1) * self.time_frequency.float() * 1000.0
        time_embed = torch.cat((angles.cos(), angles.sin()), dim=-1)
        if time_embed.shape[-1] < self.embed_dims:
            time_embed = F.pad(time_embed, (0, self.embed_dims - time_embed.shape[-1]))
        step_condition = condition + time_embed.to(condition.dtype)
        query = self.behavior_embed(z) + condition.unsqueeze(1)
        query = query + self.action_query.to(query.dtype)
        hidden = self.out_norm(self.block(query, scene_kv, step_condition, mask))
        return self.velocity_head(hidden), hidden

    def _rollout(self, condition: torch.Tensor,
                 scene_kv: Tuple[torch.Tensor, torch.Tensor],
                 mask: Optional[torch.Tensor],
                 initial_noise: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = condition.shape[0]
        if initial_noise is None:
            z = self.fixed_noise.to(condition).expand(b, -1, -1)
        else:
            if initial_noise.shape != (b, self.num_tokens, 2):
                raise ValueError('initial_noise must be [B, K, 2], already scaled')
            z = initial_noise.to(condition)
        step = 1.0 / self.flow_steps
        for index in range(self.flow_steps):
            tau = condition.new_full((b,), index * step)
            velocity, _ = self._field(z, tau, condition, scene_kv, mask)
            z = z + step * velocity.to(z.dtype)
        # Re-evaluate at the generated endpoint; the last integration evaluation
        # was at tau < 1 and its context would be stale for the final behavior.
        _, hidden = self._field(z, condition.new_ones(b), condition, scene_kv, mask)
        action = torch.cat((z.to(hidden.dtype), hidden[..., 2:]), dim=-1)
        return action, z

    def forward(self, scene_tokens: torch.Tensor,
                cmd: Optional[torch.Tensor] = None,
                ego_status: Optional[torch.Tensor] = None, *,
                scene_key_padding_mask: Optional[torch.Tensor] = None,
                initial_noise: Optional[torch.Tensor] = None,
                return_aux: bool = False
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        condition, scene_kv, mask = self._prepare(
            scene_tokens, cmd, ego_status, scene_key_padding_mask)
        action, behavior = self._rollout(condition, scene_kv, mask, initial_noise)
        if return_aux:
            return action, {'behavior_coefficients': behavior}
        return action

    def _flow_loss(self, condition: torch.Tensor,
                   scene_kv: Tuple[torch.Tensor, torch.Tensor],
                   mask: Optional[torch.Tensor], target: torch.Tensor,
                   valid: Optional[torch.Tensor]) -> torch.Tensor:
        b = condition.shape[0]
        if target.shape != (b, self.num_tokens, 2):
            raise ValueError('behavior_targets must be normalized [B, K, 2]')
        if valid is None:
            valid = torch.ones(b, device=condition.device, dtype=torch.bool)
        elif valid.shape != (b,) or valid.dtype != torch.bool:
            raise ValueError('behavior_valid must be bool [B]')
        valid = valid.to(condition.device)
        target = target.detach().to(device=condition.device, dtype=torch.float32)
        target = torch.where(valid[:, None, None], target, torch.zeros_like(target))
        if not torch.isfinite(target).all():
            raise ValueError('valid behavior targets must be finite')
        noise = torch.randn_like(target) * self.source_std
        tau = torch.rand(b, device=condition.device, dtype=torch.float32)
        z_tau = (1 - tau[:, None, None]) * noise + tau[:, None, None] * target
        predicted, _ = self._field(z_tau.to(condition.dtype), tau,
                                   condition, scene_kv, mask)
        error = (predicted.float() - (target - noise)).square().mean(dim=(1, 2))
        weights = valid.to(error.dtype)
        # A differentiable zero when every sample is invalid (DDP-safe graph).
        return (error * weights).sum() / weights.sum().clamp_min(1)

    def flow_matching_loss(self, scene_tokens: torch.Tensor,
                           behavior_targets: torch.Tensor,
                           cmd: Optional[torch.Tensor] = None,
                           ego_status: Optional[torch.Tensor] = None, *,
                           behavior_valid: Optional[torch.Tensor] = None,
                           scene_key_padding_mask: Optional[torch.Tensor] = None
                           ) -> torch.Tensor:
        """Standalone loss when the existing head calls forward separately.

        forward_train is more efficient: it shares scene K/V and condition
        preparation between the target-free query rollout and this loss.
        """
        condition, scene_kv, mask = self._prepare(
            scene_tokens, cmd, ego_status, scene_key_padding_mask)
        return self._flow_loss(condition, scene_kv, mask, behavior_targets, behavior_valid)

    def forward_train(self, scene_tokens: torch.Tensor,
                      cmd: Optional[torch.Tensor] = None,
                      ego_status: Optional[torch.Tensor] = None, *,
                      behavior_targets: torch.Tensor,
                      behavior_valid: Optional[torch.Tensor] = None,
                      scene_key_padding_mask: Optional[torch.Tensor] = None
                      ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return target-free action queries and one unweighted flow loss.

        Keep the returned action attached to the existing trajectory loss.
        Targets influence the auxiliary loss ONLY, never the returned queries.
        No detach/no_grad is used on the rollout or cached scene projections.
        """
        condition, scene_kv, mask = self._prepare(
            scene_tokens, cmd, ego_status, scene_key_padding_mask)
        action, _ = self._rollout(condition, scene_kv, mask)
        loss = self._flow_loss(condition, scene_kv, mask, behavior_targets, behavior_valid)
        return action, {'loss_action_flow': loss}

    @torch.no_grad()
    def build_behavior_targets(self, future_traj: torch.Tensor,
                               initial_velocity: torch.Tensor, dt: float, *,
                               future_mask: Optional[torch.Tensor] = None,
                               trajectory_is_displacement: bool = False
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert complete expert motion into fixed low-frequency coefficients.

        future_traj: [B,H,2], meters, current ego frame; positions by default.
        initial_velocity: [B,2], m/s, SAME coordinate frame, supplied explicitly.
        dt: positive uniform interval between consecutive future trajectory points.
        future_mask: bool [B,H] or [B,H,1], True=valid. Masked NaNs are allowed.
        Returns normalized [B,K,2] targets and bool [B] supervision validity.
        behavior_scale is a fixed per-coordinate m/s scale, not learned statistics.
        """
        if future_traj.dim() != 3 or future_traj.shape[-1] != 2:
            raise ValueError('future_traj must be [B, H, 2]')
        b, h, _ = future_traj.shape
        if h < self.num_tokens or b < 1:
            raise ValueError('need nonempty batch and H >= num_tokens for DCT basis')
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        if initial_velocity.shape != (b, 2):
            raise ValueError('initial_velocity must be [B, 2] in current ego frame')
        positions = future_traj.detach().float()
        v0 = initial_velocity.detach().to(positions)
        if future_mask is None:
            mask = torch.ones(b, h, dtype=torch.bool, device=positions.device)
        else:
            if future_mask.shape == (b, h, 1):
                future_mask = future_mask.squeeze(-1)
            if future_mask.shape != (b, h) or future_mask.dtype != torch.bool:
                raise ValueError('future_mask must be bool [B, H] or [B, H, 1]')
            mask = future_mask.to(positions.device)
        valid = mask.all(dim=1)
        if not torch.isfinite(positions[mask]).all() or not torch.isfinite(v0).all():
            raise ValueError('valid trajectory labels and initial_velocity must be finite')
        positions = torch.where(valid[:, None, None], positions, torch.zeros_like(positions))
        if trajectory_is_displacement:
            displacements = positions
        else:
            origin = positions.new_zeros(b, 1, 2)
            displacements = positions - torch.cat((origin, positions[:, :-1]), dim=1)
        delta_velocity = displacements / dt - v0[:, None, :]
        delta_velocity = delta_velocity / self.behavior_scale.to(delta_velocity)
        grid = torch.arange(h, device=positions.device, dtype=torch.float32) + 0.5
        orders = torch.arange(self.num_tokens, device=positions.device, dtype=torch.float32)
        basis = torch.cos(math.pi / h * grid[:, None] * orders[None, :]) * math.sqrt(2.0 / h)
        basis[:, 0] = 1.0 / math.sqrt(h)
        target = torch.einsum('hk,bhd->bkd', basis, delta_velocity)
        target = torch.where(valid[:, None, None], target, torch.zeros_like(target))
        return target, valid


BehaviorQueryFlow = ActionDiT


class MathMultiheadAttention(nn.Module):
    """Multi-head attention without PyTorch's fused SDP CUDA kernels.

    PyTorch 2.0.1+cu118 can select an invalid backward kernel on H100 GPUs.
    Keeping the same parameter names/shapes as ``nn.MultiheadAttention`` makes
    checkpoints compatible while using explicit matmul-softmax-matmul math.
    """

    def __init__(self, embed_dims: int, num_heads: int = 8,
                 dropout: float = 0.0, bias: bool = True):
        super().__init__()
        if num_heads < 1 or embed_dims % num_heads:
            raise ValueError('embed_dims must be divisible by positive num_heads')
        if not 0.0 <= dropout < 1.0:
            raise ValueError('dropout must be in [0, 1)')
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.dropout = float(dropout)
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dims, embed_dims))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dims))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dims, embed_dims, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.zeros_(self.in_proj_bias)
            nn.init.zeros_(self.out_proj.bias)

    def _heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, tokens, _ = tensor.shape
        return tensor.reshape(
            batch_size, tokens, self.num_heads,
            self.head_dims).transpose(1, 2)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, need_weights: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            raise ValueError('query, key and value must be [B, N, C]')
        if (query.shape[0] != key.shape[0] or key.shape[:2] != value.shape[:2] or
                query.shape[-1] != self.embed_dims or
                key.shape[-1] != self.embed_dims or
                value.shape[-1] != self.embed_dims):
            raise ValueError('attention batch, token or channel shapes differ')
        weight_q, weight_k, weight_v = self.in_proj_weight.chunk(3, dim=0)
        if self.in_proj_bias is None:
            bias_q = bias_k = bias_v = None
        else:
            bias_q, bias_k, bias_v = self.in_proj_bias.chunk(3, dim=0)
        projected_q = self._heads(F.linear(query, weight_q, bias_q))
        projected_k = self._heads(F.linear(key, weight_k, bias_k))
        projected_v = self._heads(F.linear(value, weight_v, bias_v))
        with torch.autocast(
                device_type=query.device.type, enabled=False):
            scores = torch.matmul(
                projected_q.float(), projected_k.float().transpose(-1, -2))
            scores = scores * (self.head_dims ** -0.5)
            attention = F.dropout(
                scores.softmax(dim=-1), self.dropout, self.training)
        output = torch.matmul(attention.to(projected_v.dtype), projected_v)
        output = output.transpose(1, 2).reshape(
            query.shape[0], query.shape[1], self.embed_dims)
        output = self.out_proj(output)
        if need_weights:
            return output, attention.mean(dim=1)
        return output, None


class LatentTransitionBlock(nn.Module):
    """Predict a scene change whose values come only from control tokens."""

    def __init__(self, embed_dims: int, num_heads: int = 8,
                 dropout: float = 0.1, ffn_ratio: int = 4):
        super().__init__()
        self.query_norm = nn.LayerNorm(embed_dims)
        self.control_norm = nn.LayerNorm(embed_dims)
        self.control_attn = MathMultiheadAttention(
            embed_dims, num_heads, dropout=dropout)
        self.update_norm = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dims * ffn_ratio, embed_dims),
            nn.Dropout(dropout))

    def forward(self, query_tokens: torch.Tensor,
                control_tokens: torch.Tensor) -> torch.Tensor:
        """Return a control-derived update without a scene-value shortcut."""
        query = self.query_norm(query_tokens)
        control = self.control_norm(control_tokens)
        update = self.control_attn(
            query, control, control, need_weights=False)[0]
        return update + self.ffn(self.update_norm(update))


class SARLatentTransition(nn.Module):
    """Trajectory-aware transition from current to future scene queries."""

    def __init__(self,
                 embed_dims: int,
                 num_trajectory_steps: int,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 ffn_ratio: int = 4):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_trajectory_steps = num_trajectory_steps
        self.state_norm = nn.LayerNorm(embed_dims)
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(2, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.GELU(),
            nn.Linear(embed_dims, embed_dims))
        self.trajectory_pos = nn.Parameter(
            torch.randn(1, num_trajectory_steps, embed_dims) * 0.02)
        self.layers = nn.ModuleList([
            LatentTransitionBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                dropout=dropout,
                ffn_ratio=ffn_ratio)
            for _ in range(num_layers)
        ])
        self.delta_head = nn.Sequential(
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims),
            nn.GELU(),
            nn.Linear(embed_dims, embed_dims))
        nn.init.normal_(self.delta_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.delta_head[-1].bias)

    def normalize_state(self, state_tokens: torch.Tensor) -> torch.Tensor:
        """Map current and target Scene Queries to one transition space."""
        return self.state_norm(state_tokens)

    def forward(self,
                state_tokens: torch.Tensor,
                action_tokens: torch.Tensor,
                trajectory: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state_tokens.dim() != 3 or action_tokens.dim() != 3:
            raise ValueError(
                "state_tokens and action_tokens must be [B, N, C]")
        if state_tokens.shape[0] != action_tokens.shape[0]:
            raise ValueError("state/action batch sizes must match")
        if trajectory.dim() != 3 or trajectory.shape[-1] != 2:
            raise ValueError("trajectory must have shape [B, H, 2]")
        if trajectory.shape[0] != state_tokens.shape[0]:
            raise ValueError("state/trajectory batch sizes must match")
        if trajectory.shape[1] != self.num_trajectory_steps:
            raise ValueError(
                "Expected {} trajectory steps, got {}".format(
                    self.num_trajectory_steps, trajectory.shape[1]))

        trajectory = trajectory.to(
            dtype=state_tokens.dtype, device=state_tokens.device)
        trajectory_tokens = self.trajectory_encoder(trajectory)
        trajectory_tokens = trajectory_tokens + self.trajectory_pos.to(
            trajectory_tokens.dtype)
        # Only Action Query and trajectory tokens provide values to the state
        # transition. Scene Query is used as a query, not duplicated in memory.
        control_tokens = torch.cat((action_tokens, trajectory_tokens), dim=1)
        state_base = self.normalize_state(state_tokens)
        hidden = state_base
        for layer in self.layers:
            hidden = layer(hidden, control_tokens)
        delta = self.delta_head(hidden)
        # Detaching only the identity base prevents the future objective from
        # being solved through S_t -> S_hat directly. Gradients still reach
        # Scene Query through its role as the transition query above.
        future_state = state_base.detach() + delta
        return future_state, delta, state_base


class FutureFeedbackBlock(nn.Module):
    """Correct current Scene-Action Query using predicted scene innovation."""

    def __init__(self, embed_dims: int, num_heads: int = 8,
                 dropout: float = 0.1, ffn_ratio: int = 4):
        super().__init__()
        self.current_norm = nn.LayerNorm(embed_dims)
        # Affine-free normalization and bias-free projections ensure that a
        # zero future change produces a zero feedback update.
        self.future_delta_norm = nn.LayerNorm(
            embed_dims, elementwise_affine=False)
        self.cross_attn = MathMultiheadAttention(
            embed_dims, num_heads, dropout=dropout, bias=False)
        self.update_norm = nn.LayerNorm(
            embed_dims, elementwise_affine=False)
        self.update_ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * ffn_ratio, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dims * ffn_ratio, embed_dims, bias=False),
            nn.Dropout(dropout))
        self.feedback_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, current_tokens: torch.Tensor,
                future_delta_tokens: torch.Tensor,
                current_pos: Optional[torch.Tensor] = None,
                future_pos: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        if current_tokens.dim() != 3 or future_delta_tokens.dim() != 3:
            raise ValueError("current and future-delta tokens must be [B, N, C]")
        query = self.current_norm(current_tokens)
        future_delta = self.future_delta_norm(future_delta_tokens)
        key = future_delta
        if current_pos is not None:
            query = query + current_pos.to(query.dtype)
        if future_pos is not None:
            key = key + future_pos.to(key.dtype)
        update, attention = self.cross_attn(
            query, key, future_delta, need_weights=True)
        correction = update + self.update_ffn(self.update_norm(update))
        corrected = current_tokens + \
            torch.tanh(self.feedback_scale) * correction
        return corrected, attention


class ActionInterventionWorldRefiner(nn.Module):
    """Decode trajectory residuals from an action-intervention world response.

    The caller predicts the factual world innovation and a behavior-neutral
    intervention with the same transition. Their finite difference isolates
    the part of the future that the generated Action Query can change. Waypoint
    queries read both the factual innovation and this controllability residual.

    Action features alter attention queries, never multiply a learned gate.
    Sparse world innovations are the only attention values and every downstream
    projection is bias-free, so zero factual/intervention evidence gives an
    exactly zero trajectory correction. Affine-free normalization prevents a
    small latent magnitude from automatically collapsing the correction.
    """

    def __init__(self, embed_dims: int, num_trajectory_steps: int,
                 num_heads: int = 8, residual_scale: float = 1.0,
                 dropout: float = 0.0, ffn_ratio: int = 2):
        super().__init__()
        if embed_dims < 4 or num_heads < 1 or embed_dims % num_heads:
            raise ValueError('embed_dims must be divisible by positive num_heads')
        if num_trajectory_steps < 1:
            raise ValueError('num_trajectory_steps must be positive')
        if residual_scale <= 0 or not math.isfinite(residual_scale):
            raise ValueError('residual_scale must be finite and positive')
        if not 0.0 <= dropout < 1.0 or ffn_ratio <= 0:
            raise ValueError('invalid dropout or ffn_ratio')
        hidden_dims = int(embed_dims * ffn_ratio)
        self.embed_dims = embed_dims
        self.num_trajectory_steps = num_trajectory_steps
        self.residual_scale = float(residual_scale)

        self.waypoint_norm = nn.LayerNorm(embed_dims)
        self.position_encoder = nn.Sequential(
            nn.Linear(2, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.GELU(),
            nn.Linear(embed_dims, embed_dims))
        self.time_embed = nn.Parameter(
            torch.randn(1, num_trajectory_steps, embed_dims) * 0.02)
        self.action_norm = nn.LayerNorm(embed_dims)
        self.action_attn = MathMultiheadAttention(
            embed_dims, num_heads, dropout=dropout)

        self.factual_norm = nn.LayerNorm(
            embed_dims, elementwise_affine=False)
        self.intervention_norm = nn.LayerNorm(
            embed_dims, elementwise_affine=False)
        # Branch identity and sparse position affect only keys. Values remain
        # pure world evidence, which preserves the zero-evidence invariant.
        self.world_branch_embed = nn.Parameter(
            torch.randn(1, 2, embed_dims) * 0.02)
        self.world_attn = MathMultiheadAttention(
            embed_dims, num_heads, dropout=dropout, bias=False)
        self.response_norm = nn.LayerNorm(
            embed_dims, elementwise_affine=False)
        self.response_ffn = nn.Sequential(
            nn.Linear(embed_dims, hidden_dims, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, embed_dims, bias=False),
            nn.Dropout(dropout))
        self.correction_head = nn.Sequential(
            nn.LayerNorm(embed_dims, elementwise_affine=False),
            nn.Linear(embed_dims, embed_dims, bias=False),
            nn.GELU(),
            nn.Linear(embed_dims, 2, bias=False))
        nn.init.normal_(self.correction_head[-1].weight, std=0.02)

    def forward(self, waypoint_tokens: torch.Tensor,
                prior_positions: torch.Tensor,
                factual_delta_tokens: torch.Tensor,
                neutral_delta_tokens: torch.Tensor,
                action_tokens: torch.Tensor,
                world_position_tokens: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if waypoint_tokens.dim() != 3:
            raise ValueError('waypoint_tokens must be [B, H, C]')
        batch_size, steps, channels = waypoint_tokens.shape
        if steps != self.num_trajectory_steps or channels != self.embed_dims:
            raise ValueError('waypoint token shape differs from refiner configuration')
        if prior_positions.shape != (batch_size, steps, 2):
            raise ValueError('prior_positions must be [B, H, 2]')
        if (factual_delta_tokens.dim() != 3 or
                factual_delta_tokens.shape[0] != batch_size or
                factual_delta_tokens.shape[-1] != channels):
            raise ValueError('factual_delta_tokens must be [B, N, C]')
        if neutral_delta_tokens.shape != factual_delta_tokens.shape:
            raise ValueError('neutral and factual world deltas must have equal shape')
        if (action_tokens.dim() != 3 or
                action_tokens.shape[0] != batch_size or
                action_tokens.shape[-1] != channels):
            raise ValueError('action_tokens must be [B, K, C]')
        if (world_position_tokens is not None and
                world_position_tokens.shape != factual_delta_tokens.shape):
            raise ValueError('world_position_tokens must match world deltas')

        query = self.waypoint_norm(waypoint_tokens)
        query = query + self.position_encoder(prior_positions.to(query))
        query = query + self.time_embed.to(query)
        action_values = self.action_norm(action_tokens)
        action_context, action_attention = self.action_attn(
            query, action_values, action_values, need_weights=True)
        query = query + action_context

        factual_values = self.factual_norm(factual_delta_tokens)
        intervention_values = self.intervention_norm(
            factual_delta_tokens - neutral_delta_tokens)
        world_values = torch.cat((factual_values, intervention_values), dim=1)
        world_keys = world_values.reshape(
            batch_size, 2, factual_values.shape[1], channels)
        world_keys = world_keys + self.world_branch_embed.to(world_keys).unsqueeze(2)
        if world_position_tokens is not None:
            positions = world_position_tokens.to(world_keys)
            world_keys = world_keys + positions.unsqueeze(1)
        world_keys = world_keys.reshape(batch_size, -1, channels)

        response, world_attention = self.world_attn(
            query, world_keys, world_values, need_weights=True)
        response = response + self.response_ffn(self.response_norm(response))
        correction = torch.tanh(self.correction_head(response))
        correction = correction * self.residual_scale
        return correction, world_attention, action_attention


__all__ = [
    "ActionDiT", "BehaviorQueryFlow", "SARLatentTransition",
    "FutureFeedbackBlock", "ActionInterventionWorldRefiner"
]
