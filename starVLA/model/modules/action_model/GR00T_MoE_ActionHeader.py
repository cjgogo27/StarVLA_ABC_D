# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0.
"""
GR00T-compatible Flow-Matching action head with a lightweight MoE decoder.

The module intentionally keeps the original GR00T submodule names for the
shared path. In particular, ``action_decoder`` is still the first expert, so a
trained QwenGR00T checkpoint can initialize this head through the existing
``trainer.pretrained_checkpoint`` loading path. Additional experts are copied
from the loaded base decoder lazily before the first forward/predict call.
"""

import torch
import torch.nn.functional as F
from torch import nn

from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, MLP


class MoEDecoderMixin:
    def _init_moe_decoder(self, config):
        moe_cfg = config.get("moe", {}) or {}
        self.num_experts = int(moe_cfg.get("num_experts", 4))
        self.top_k = int(moe_cfg.get("top_k", 2))
        self.router_noise_std = float(moe_cfg.get("router_noise_std", 0.0))
        self.load_balance_loss_coeff = float(moe_cfg.get("load_balance_loss_coeff", 0.01))
        self.router_z_loss_coeff = float(moe_cfg.get("router_z_loss_coeff", 0.001))
        self.initialize_experts_from_base = bool(moe_cfg.get("initialize_experts_from_base", True))
        self.expert_init_noise_std = float(moe_cfg.get("expert_init_noise_std", 0.0))
        self.balance_on_action_tokens_only = bool(moe_cfg.get("balance_on_action_tokens_only", True))

        if self.num_experts < 1:
            raise ValueError("framework.action_model.moe.num_experts must be >= 1")
        if self.top_k < 1 or self.top_k > self.num_experts:
            raise ValueError("framework.action_model.moe.top_k must be in [1, num_experts]")

        # FlowmatchingActionHead already created self.action_decoder. Keep that
        # name as expert 0 so GR00T checkpoints load it without remapping.
        self.expert_decoders = nn.ModuleList([self.action_decoder])
        for _ in range(self.num_experts - 1):
            self.expert_decoders.append(
                MLP(
                    input_dim=self.model.config.output_dim,
                    hidden_dim=self.hidden_size,
                    output_dim=self.action_dim,
                )
            )

        self.router = nn.Linear(self.model.config.output_dim, self.num_experts)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        self._moe_experts_synced = False
        self._last_moe_losses = {}
        self._last_moe_metrics = {}

    def _sync_extra_experts_from_base(self):
        if self._moe_experts_synced or not self.initialize_experts_from_base:
            return
        base_state = self.expert_decoders[0].state_dict()
        for expert in self.expert_decoders[1:]:
            expert.load_state_dict(base_state, strict=True)
            if self.expert_init_noise_std > 0:
                with torch.no_grad():
                    for param in expert.parameters():
                        param.add_(torch.randn_like(param) * self.expert_init_noise_std)
        self._moe_experts_synced = True

    def _moe_decode(self, hidden_states: torch.Tensor, training: bool, num_action_tokens: int | None = None) -> torch.Tensor:
        self._sync_extra_experts_from_base()

        logits = self.router(hidden_states)
        if training and self.router_noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.router_noise_std

        router_probs = F.softmax(logits, dim=-1)
        if self.top_k < self.num_experts:
            top_values, top_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
            gates = torch.zeros_like(router_probs).scatter_(-1, top_indices, top_values)
            gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        else:
            gates = router_probs

        expert_outputs = torch.stack([expert(hidden_states) for expert in self.expert_decoders], dim=-2)
        decoded = (expert_outputs * gates.unsqueeze(-1)).sum(dim=-2)

        if self.balance_on_action_tokens_only and num_action_tokens is not None:
            balance_logits = logits[:, -num_action_tokens:]
            balance_probs = router_probs[:, -num_action_tokens:]
            balance_gates = gates[:, -num_action_tokens:]
        else:
            balance_logits = logits
            balance_probs = router_probs
            balance_gates = gates

        # Switch-Transformer style load-balancing term, computed on supervised
        # action tokens by default so the auxiliary objective matches the loss.
        mean_prob = balance_probs.mean(dim=(0, 1))
        mean_gate = balance_gates.mean(dim=(0, 1))
        load_balance_loss = self.num_experts * torch.sum(mean_prob * mean_gate)
        router_z_loss = torch.logsumexp(balance_logits.float(), dim=-1).pow(2).mean()

        top1 = balance_probs.argmax(dim=-1)
        usage = F.one_hot(top1, num_classes=self.num_experts).float().mean(dim=(0, 1))
        entropy = -(balance_probs.float() * balance_probs.float().clamp_min(1e-9).log()).sum(dim=-1).mean()

        self._last_moe_losses = {
            "moe_load_balance_loss": load_balance_loss,
            "moe_router_z_loss": router_z_loss.to(load_balance_loss.dtype),
        }
        self._last_moe_metrics = {
            "moe/router_entropy": entropy.detach(),
            "moe/router_top1_max": usage.max().detach(),
            "moe/router_top1_min": usage.min().detach(),
        }
        for idx, value in enumerate(usage.detach()):
            self._last_moe_metrics[f"moe/expert_{idx}_top1_usage"] = value
        return decoded

    def get_moe_losses(self):
        return self._last_moe_losses

    def get_moe_loss_weights(self):
        return {
            "moe_load_balance_loss": self.load_balance_loss_coeff,
            "moe_router_z_loss": self.router_z_loss_coeff,
        }

    def get_moe_metrics(self):
        return self._last_moe_metrics


class MoEFlowmatchingActionHead(MoEDecoderMixin, FlowmatchingActionHead):
    def __init__(self, full_config):
        super().__init__(full_config=full_config)
        self._init_moe_decoder(full_config.framework.action_model)

    def forward(
        self, vl_embs: torch.Tensor, actions: torch.Tensor, state: torch.Tensor = None, encoder_attention_mask=None
    ):
        device = vl_embs.device

        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        state_features = self.state_encoder(state) if state is not None else None

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        pred = self._moe_decode(model_output, training=self.training, num_action_tokens=actions.shape[1])
        pred_actions = pred[:, -actions.shape[1] :]

        loss = ((pred_actions - velocity) ** 2).mean()
        return loss

    @torch.no_grad()
    def predict_action(self, vl_embs: torch.Tensor, state: torch.Tensor = None) -> torch.Tensor:
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor)

            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )

            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self._moe_decode(model_output, training=False, num_action_tokens=self.action_horizon)

            pred_velocity = pred[:, -self.action_horizon :]
            actions = actions + dt * pred_velocity
        return actions


def get_action_model(config=None):
    return MoEFlowmatchingActionHead(full_config=config)
