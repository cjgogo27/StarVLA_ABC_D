# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025].
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""

import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenGR00T
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenGR00TDefaultConfig:
    """QwenGR00T framework default parameters.

    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.  Extra YAML keys not listed here are kept
    as-is (Config-as-API flexibility).
    """

    # --- Registry identifier ---
    name: str = "QwenGR00T"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (local or HF hub id)
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
            # VLM hidden dimension (used for cross-attention alignment)
            "vl_hidden_dim": 2048,
        }
    )

    # # === DINO encoder (optional multi-view spatial tokens) === Dino is not used in this QwenGR00T version, we can add it later when we want to use it
    # dino: dict = field(default_factory=lambda: {
    #     # DINO backbone variant: "dinov2_vits14" | "dinov2_vitb14" | ...
    #     "dino_backbone": "dinov2_vits14",
    # })

    # === Action head (Flow-matching / DiT diffusion) ===
    action_model: dict = field(
        default_factory=lambda: {
            # DiT model size: "DiT-B" | "DiT-L" | "DiT-XL"
            "action_model_type": "DiT-B",
            # Hidden dim for action model (auto-aligned at runtime)
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            # Whether to add positional embeddings in the action head
            "add_pos_embed": True,
            "max_seq_len": 1024,
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # State dimension (proprioception input)
            "state_dim": 7,
            # Canonical chunk length (number of action steps the head predicts).
            # Legacy YAMLs may use future_action_window_size = action_horizon - 1;
            # apply_config_compat normalises both directions.
            "action_horizon": 8,
            # Repeat factor for flow-matching loss (more noise samples per batch)
            "repeated_diffusion_steps": 8,
            # Beta distribution params for noise schedule
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            # Inference denoising steps
            "num_inference_timesteps": 4,
            # Number of vision tokens fed to action head
            "num_target_vision_tokens": 32,
            # === DiT Transformer sub-config ===
            "diffusion_model_cfg": {
                # Cross-attention dim (aligned to VLM hidden_size at runtime)
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

    # === Motion-Oriented Transformer adapter ===
    # Disabled by default. When enabled, it builds a compact motion token from
    # recent proprio/action history and appends it to the VLM tokens before the
    # GR00T action head cross-attention.
    motion_adapter: dict = field(
        default_factory=lambda: {
            "enabled": False,
            "history_window": 4,
            "hidden_dim": 512,
            "num_layers": 2,
            "num_heads": 8,
            "dropout": 0.1,
            "num_motion_tokens": 1,
            "use_state": True,
            "use_action": True,
            "pass_state_to_action_head": True,
        }
    )

    # # === Training precision flag === This is unnecessary, unused parameter
    # reduce_in_full_precision: bool = True


class MotionTokenAdapter(nn.Module):
    """Small MOT adapter that turns recent state/action history into tokens."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        output_dim: int,
        history_window: int = 4,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_motion_tokens: int = 1,
        use_state: bool = True,
        use_action: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"motion_adapter.hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.history_window = int(history_window)
        self.num_motion_tokens = int(num_motion_tokens)
        self.use_state = bool(use_state) and self.state_dim > 0
        self.use_action = bool(use_action) and self.action_dim > 0

        self.state_proj = nn.Linear(self.state_dim, hidden_dim) if self.use_state else None
        self.action_proj = nn.Linear(self.action_dim, hidden_dim) if self.use_action else None
        self.state_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.action_type = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embedding = nn.Embedding(max(1, self.history_window * 2), hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.motion_queries = nn.Parameter(torch.randn(self.num_motion_tokens, hidden_dim) * 0.02)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    @staticmethod
    def _match_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
        if x.shape[-1] == dim:
            return x
        if x.shape[-1] > dim:
            return x[..., :dim]
        pad = x.new_zeros(*x.shape[:-1], dim - x.shape[-1])
        return torch.cat([x, pad], dim=-1)

    def _prepare_history(
        self,
        x: Optional[torch.Tensor],
        *,
        batch_size: int,
        feature_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if x is None:
            x = torch.zeros(batch_size, 0, feature_dim, device=device, dtype=dtype)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self._match_dim(x[:, -self.history_window :, :], feature_dim)
        if x.shape[1] < self.history_window:
            pad = x.new_zeros(batch_size, self.history_window - x.shape[1], feature_dim)
            x = torch.cat([pad, x], dim=1)
        return x

    def forward(
        self,
        *,
        state_history: Optional[torch.Tensor],
        action_history: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tokens = []

        if self.use_state:
            state_history = self._prepare_history(
                state_history,
                batch_size=batch_size,
                feature_dim=self.state_dim,
                device=device,
                dtype=dtype,
            )
            state_tokens = self.state_proj(state_history.to(dtype=self.state_proj.weight.dtype)) + self.state_type
            tokens.append(state_tokens)

        if self.use_action:
            action_history = self._prepare_history(
                action_history,
                batch_size=batch_size,
                feature_dim=self.action_dim,
                device=device,
                dtype=dtype,
            )
            action_tokens = self.action_proj(action_history.to(dtype=self.action_proj.weight.dtype)) + self.action_type
            tokens.append(action_tokens)

        if not tokens:
            motion = self.motion_queries.unsqueeze(0).expand(batch_size, -1, -1)
            return self.out_proj(motion).to(dtype=dtype)

        history_tokens = torch.cat(tokens, dim=1)
        pos_ids = torch.arange(history_tokens.shape[1], device=device).clamp(max=self.pos_embedding.num_embeddings - 1)
        history_tokens = history_tokens + self.pos_embedding(pos_ids).unsqueeze(0)

        encoded = self.encoder(history_tokens)
        pooled = encoded.mean(dim=1, keepdim=True) + self.motion_queries.unsqueeze(0)
        return self.out_proj(pooled).to(dtype=dtype)


@FRAMEWORK_REGISTRY.register("QwenGR00T")
class Qwen_GR00T(baseframework):
    """
    Multimodal vision-language-action model (GR00T variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - Flow-matching (DiT) diffusion head for continuous action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenGR00TDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.action_dim = int(self.config.framework.action_model.action_dim)
        self.state_dim = int(self.config.framework.action_model.get("state_dim", 0) or 0)

        motion_cfg = self.config.framework.get("motion_adapter", {})
        self.motion_adapter_enabled = bool(motion_cfg.get("enabled", False))
        self.motion_state_dim = int(motion_cfg.get("state_dim", self.state_dim) or 0)
        self.motion_action_dim = int(motion_cfg.get("action_dim", self.action_dim) or 0)
        self.motion_history_window = int(motion_cfg.get("history_window", 4))
        self.pass_state_to_action_head = bool(motion_cfg.get("pass_state_to_action_head", True))
        self.motion_adapter = None
        self._motion_state_cache = []
        self._motion_action_cache = []
        if self.motion_adapter_enabled:
            self.motion_adapter = MotionTokenAdapter(
                state_dim=self.motion_state_dim,
                action_dim=self.motion_action_dim,
                output_dim=self.qwen_vl_interface.model.config.hidden_size,
                history_window=self.motion_history_window,
                hidden_dim=int(motion_cfg.get("hidden_dim", 512)),
                num_layers=int(motion_cfg.get("num_layers", 2)),
                num_heads=int(motion_cfg.get("num_heads", 8)),
                dropout=float(motion_cfg.get("dropout", 0.1)),
                num_motion_tokens=int(motion_cfg.get("num_motion_tokens", 1)),
                use_state=bool(motion_cfg.get("use_state", True)),
                use_action=bool(motion_cfg.get("use_action", True)),
            )

    def reset_motion_cache(self) -> None:
        self._motion_state_cache = []
        self._motion_action_cache = []

    @staticmethod
    def _tensor_from_examples(examples: List[dict], key: str, *, device: torch.device, dtype: torch.dtype):
        values = [example[key] for example in examples if key in example and example[key] is not None]
        if len(values) != len(examples):
            return None
        return torch.tensor(np.array(values), device=device, dtype=dtype)

    def _append_motion_tokens(
        self,
        last_hidden: torch.Tensor,
        *,
        state_history: Optional[torch.Tensor] = None,
        action_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.motion_adapter_enabled or self.motion_adapter is None:
            return last_hidden
        motion_tokens = self.motion_adapter(
            state_history=state_history,
            action_history=action_history,
            batch_size=last_hidden.shape[0],
            device=last_hidden.device,
            dtype=last_hidden.dtype,
        )
        return torch.cat([last_hidden, motion_tokens], dim=1)

    @staticmethod
    def _match_feature_dim(x: Optional[torch.Tensor], dim: int) -> Optional[torch.Tensor]:
        if x is None or dim <= 0:
            return None
        if x.shape[-1] == dim:
            return x
        if x.shape[-1] > dim:
            return x[..., :dim]
        pad = x.new_zeros(*x.shape[:-1], dim - x.shape[-1])
        return torch.cat([x, pad], dim=-1)

    def _cached_motion_history(self, state: Optional[torch.Tensor], batch_size: int):
        if batch_size != 1:
            return state, None

        state_history = None
        if state is not None:
            current_state = state[:, -1:, :].detach()
            self._motion_state_cache.append(current_state)
            self._motion_state_cache = self._motion_state_cache[-self.motion_history_window :]
            state_history = torch.cat(self._motion_state_cache, dim=1)

        action_history = None
        if self._motion_action_cache:
            action_history = torch.cat(self._motion_action_cache[-self.motion_history_window :], dim=1).to(
                state.device if state is not None else self.action_model.device
            )
        return state_history, action_history

    def _update_action_cache(self, pred_actions: torch.Tensor) -> None:
        if pred_actions.shape[0] != 1:
            return
        self._motion_action_cache.append(pred_actions[:, :1, :].detach())
        self._motion_action_cache = self._motion_action_cache[-self.motion_history_window :]

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """ """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        state_history = None
        action_history = None

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)
            action_history = self._tensor_from_examples(
                examples, "action_history", device=last_hidden.device, dtype=last_hidden.dtype
            )
            if action_history is None:
                action_history = actions[:, : self.motion_history_window, :]

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_history = self._tensor_from_examples(
                    examples, "state_history", device=last_hidden.device, dtype=last_hidden.dtype
                )
                if state_history is None:
                    state_history = state
                action_head_state = self._match_feature_dim(state, self.state_dim)
                state_repeated = action_head_state.repeat(repeated_diffusion_steps, 1, 1)
                if not self.pass_state_to_action_head:
                    state_repeated = None

            last_hidden = self._append_motion_tokens(
                last_hidden,
                state_history=state_history,
                action_history=action_history,
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(
                last_hidden_repeated, actions_target_repeated, state_repeated
            )  # (B, chunk_len, action_dim)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        if self.motion_adapter_enabled and int(kwargs.get("step", -1)) == 0:
            self.reset_motion_cache()
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )
        state_history, action_history = self._cached_motion_history(state, last_hidden.shape[0])
        last_hidden = self._append_motion_tokens(
            last_hidden,
            state_history=state_history,
            action_history=action_history,
        )
        action_head_state = state if self.pass_state_to_action_head else None
        action_head_state = self._match_feature_dim(action_head_state, self.state_dim)

        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, action_head_state)  # (B, chunk_len, action_dim)
            if self.motion_adapter_enabled:
                self._update_action_cache(pred_actions)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "This is a fake instruction for testing.",
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    predict_output = model.predict_action(examples=[sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    print("Finished")
