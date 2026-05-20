# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
CosmoPredict2-PI Framework — World Model + Layer-wise Cross-DiT Flow-Matching.

Uses Cosmos-Predict2 DiT as the perception backbone with a layer-wise
cross-DiT flow-matching action head, inspired by π₀ (Physical Intelligence).

Architecture:
  T5 (text) + VAE (image) → DiT Transformer (28 blocks)
    → Multi-layer hidden_states [28 × (B, N, 2048)]
    → LayerwiseFlowmatchingActionHead (cross-DiT)
    → action predictions [B, chunk_len, action_dim]

Key differences from CosmoPredict2GR00T:
  - Action head: Layer-wise cross-DiT (not single-layer flow-matching)
  - Uses ALL transformer layers (not just last hidden state)
  - More expressive multi-scale feature fusion
"""

import sys
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead, get_action_model
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class CosmoPredict2PIDefaultConfig:
    """CosmoPredict2-PI default parameters."""

    name: str = "CosmoPredict2PI"

    # === World Model backbone (Cosmos-Predict2) ===
    world_model: dict = field(
        default_factory=lambda: {
            "base_wm": "./playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World",
            "extract_layers": [-1],
        }
    )

    # LayerwiseFM reads qwenvl.vl_hidden_dim and qwenvl.num_vl_layers
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World",
            "vl_hidden_dim": 2048,
            "num_vl_layers": 28,
        }
    )

    # === Action head (Layer-wise Flow-matching / cross-DiT) ===
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "LayerwiseFM",
            "action_dim": 7,
            "state_dim": 7,
            "future_action_window_size": 15,
            "past_action_window_size": 0,
            "repeated_diffusion_steps": 2,
            "num_inference_timesteps": 4,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "num_target_vision_tokens": 32,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "diffusion_model_cfg": {},
        }
    )


@FRAMEWORK_REGISTRY.register("CosmoPredict2PI")
class CosmoPredict2_PI(baseframework):
    """
    World-Model-for-Action framework using Cosmos-Predict2 + Layer-wise cross-DiT.

    Components:
      - Cosmos-Predict2 DiT (T5 + VAE + Transformer) for spatiotemporal features
      - Layer-wise cross-DiT flow-matching head fed by all transformer layers
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(CosmoPredict2PIDefaultConfig, config)

        self.backbone = get_world_model(config=self.config)

        # Auto-detect hidden size and num layers from world model
        wm_hidden = self.backbone.model.config.hidden_size
        # Cosmos uses transformer_blocks, Wan uses blocks
        if hasattr(self.backbone.transformer, "transformer_blocks"):
            num_blocks = len(self.backbone.transformer.transformer_blocks)
        else:
            num_blocks = len(self.backbone.transformer.blocks)

        self.config.framework.qwenvl.vl_hidden_dim = wm_hidden
        self.config.framework.qwenvl.num_vl_layers = num_blocks

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)
        self.flare_cfg = self._get_flare_cfg()
        self.flare_enabled = bool(self.flare_cfg.get("enabled", False))
        if self.flare_enabled:
            projection_dim = int(self.flare_cfg.get("projection_dim", wm_hidden))
            self.flare_online_proj = nn.Sequential(
                nn.LayerNorm(wm_hidden),
                nn.Linear(wm_hidden, projection_dim),
            )
            self.flare_target_proj = nn.Sequential(
                nn.LayerNorm(wm_hidden),
                nn.Linear(wm_hidden, projection_dim),
            )
            for param in self.flare_target_proj.parameters():
                param.requires_grad = False
        else:
            self.flare_online_proj = None
            self.flare_target_proj = None

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        # Register hooks for ALL transformer blocks (not just extract_layers)
        self._all_hidden_states = []
        self._all_hooks = []
        self._register_all_hooks()

    def _get_flare_cfg(self):
        if not hasattr(self.config, "datasets"):
            return {}
        if not hasattr(self.config.datasets, "vla_data"):
            return {}
        return getattr(self.config.datasets.vla_data, "future_latent_alignment", {}) or {}

    def _register_all_hooks(self):
        """Register forward hooks on ALL transformer blocks for layerwise features."""
        for hook in self._all_hooks:
            hook.remove()
        self._all_hooks.clear()

        if hasattr(self.backbone.transformer, "transformer_blocks"):
            blocks = self.backbone.transformer.transformer_blocks
        else:
            blocks = self.backbone.transformer.blocks
        for block in blocks:
            hook = block.register_forward_hook(self._capture_all_hook)
            self._all_hooks.append(hook)

    def _capture_all_hook(self, module, input, output):
        if isinstance(output, tuple):
            self._all_hidden_states.append(output[0])
        else:
            self._all_hidden_states.append(output)

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        state = [example["state"] for example in examples] if "state" in examples[0] else None

        wm_inputs = self.backbone.build_inputs(images=batch_images, instructions=instructions)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            self._all_hidden_states.clear()
            wm_outputs = self.backbone(
                **wm_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            # Collect all layer hidden states
            vl_embs_list = list(self._all_hidden_states)
            base_hidden = vl_embs_list[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 2)
                if self.config and hasattr(self.config, "framework")
                else 2
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(vl_embs_list_repeated, actions_target_repeated, state_repeated)

        output = {"action_loss": action_loss}
        if self.flare_enabled and "future_image" in examples[0]:
            future_images = [example["future_image"] for example in examples]
            flare_loss = self._compute_future_latent_alignment_loss(
                current_hidden=base_hidden,
                future_images=future_images,
                instructions=instructions,
            )
            output["flare_loss"] = flare_loss
            output["loss_weights"] = {"flare_loss": float(self.flare_cfg.get("weight", 0.05))}

        return output

    def _pool_latents(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.mean(dim=1)

    def _encode_alignment_target(self, images: List, instructions: List[str]) -> torch.Tensor:
        future_inputs = self.backbone.build_inputs(images=images, instructions=instructions)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            self._all_hidden_states.clear()
            self.backbone(
                **future_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            future_hidden = self._all_hidden_states[-1].detach()
        return future_hidden

    def _compute_future_latent_alignment_loss(
        self,
        current_hidden: torch.Tensor,
        future_images: List,
        instructions: List[str],
    ) -> torch.Tensor:
        future_hidden = self._encode_alignment_target(future_images, instructions)

        online = self.flare_online_proj(self._pool_latents(current_hidden.float()))
        target = self.flare_target_proj(self._pool_latents(future_hidden.float())).detach()

        loss_type = str(self.flare_cfg.get("loss_type", "cosine")).lower()
        if loss_type == "mse":
            return F.mse_loss(online, target)

        online = F.normalize(online, dim=-1)
        target = F.normalize(target, dim=-1)
        return 1.0 - (online * target).sum(dim=-1).mean()

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        wm_inputs = self.backbone.build_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self._all_hidden_states.clear()
            wm_outputs = self.backbone(
                **wm_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            vl_embs_list = list(self._all_hidden_states)

        state = (
            torch.from_numpy(np.array(state)).to(vl_embs_list[-1].device, dtype=vl_embs_list[-1].dtype)
            if state is not None
            else None
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(vl_embs_list, state)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf
    from PIL import Image

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.name = "CosmoPredict2PI"
    cfg.framework.world_model = {
        "base_wm": "./playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World",
        "extract_layers": [-1],
    }
    cfg.framework.qwenvl = {
        "base_vlm": "./playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World",
        "vl_hidden_dim": 2048,
        "num_vl_layers": 28,
    }

    model: CosmoPredict2_PI = CosmoPredict2_PI(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image, image],
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),
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
    print(f"Predicted Action: {normalized_actions}")
    print("Finished")
