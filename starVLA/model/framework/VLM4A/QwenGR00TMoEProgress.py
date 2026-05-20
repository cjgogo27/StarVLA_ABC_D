# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0.
"""
Qwen-GR00T-MoE with task-progress conditioning.

This is an isolated framework variant. It keeps the GR00T-MoE action head
unchanged and appends one learned progress token to the VLM hidden sequence.
Old GR00T/MoE checkpoints load non-strictly; only the small progress encoder is
new.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_MoE_ActionHeader import MoEFlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


class TaskProgressEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x[:, None, :]
        return self.net(x)


@dataclass
class QwenGR00TMoEProgressDefaultConfig:
    name: str = "QwenGR00TMoEProgress"

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2048,
        }
    )

    task_progress: dict = field(
        default_factory=lambda: {
            "enabled": True,
            "dim": 5,
            "use_zero_when_missing": True,
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "DiT-B",
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 8,
            "repeated_diffusion_steps": 8,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            "num_target_vision_tokens": 32,
            "moe": {
                "num_experts": 4,
                "top_k": 2,
                "router_noise_std": 0.0,
                "load_balance_loss_coeff": 0.003,
                "router_z_loss_coeff": 0.001,
                "initialize_experts_from_base": True,
                "expert_init_noise_std": 0.0001,
                "balance_on_action_tokens_only": True,
            },
            "diffusion_model_cfg": {
                "cross_attention_dim": 2048,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenGR00TMoEProgress")
@FRAMEWORK_REGISTRY.register("QwenMoEProgress")
class Qwen_GR00T_MoE_Progress(baseframework):
    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenGR00TMoEProgressDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        vl_hidden_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = vl_hidden_dim

        progress_cfg = self.config.framework.get("task_progress", {})
        self.use_task_progress = bool(progress_cfg.get("enabled", True))
        self.task_progress_dim = int(progress_cfg.get("dim", 5))
        self.use_zero_progress_when_missing = bool(progress_cfg.get("use_zero_when_missing", True))
        self.task_progress_encoder = (
            TaskProgressEncoder(self.task_progress_dim, vl_hidden_dim) if self.use_task_progress else None
        )

        self.action_model: MoEFlowmatchingActionHead = get_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def _append_task_progress_token(self, last_hidden: torch.Tensor, examples: List[dict]) -> torch.Tensor:
        if not self.use_task_progress:
            return last_hidden

        if "task_progress" in examples[0]:
            task_progress = np.array([example["task_progress"] for example in examples], dtype=np.float32)
        elif self.use_zero_progress_when_missing:
            task_progress = np.zeros((len(examples), self.task_progress_dim), dtype=np.float32)
        else:
            return last_hidden

        task_progress = torch.tensor(task_progress, device=last_hidden.device, dtype=last_hidden.dtype)
        progress_token = self.task_progress_encoder(task_progress)
        return torch.cat([last_hidden, progress_token], dim=1)

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        last_hidden = self._append_task_progress_token(last_hidden, examples)

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(last_hidden_repeated, actions_target_repeated, state_repeated)

        output = {"action_loss": action_loss}
        output.update(self.action_model.get_moe_losses())
        output["loss_weights"] = self.action_model.get_moe_loss_weights()
        output.update(self.action_model.get_moe_metrics())
        return output

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        last_hidden = self._append_task_progress_token(last_hidden, examples)

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}
