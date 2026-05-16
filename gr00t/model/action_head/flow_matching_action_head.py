# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.action_encoder import SinusoidalPositionalEncoding, swish

from .cross_attention_dit import DiT, SelfAttentionTransformer


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,) or (B, T) -- per-batch or per-position timesteps
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # Accept both (B,) and (B, T) timesteps.
        # (B,) is the standard case; (B, T) is used for training-time RTC
        # where prefix positions have tau=1.0 (clean) and postfix positions
        # have the sampled noise level.
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B, T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        elif timesteps.dim() == 2 and timesteps.shape == (B, T):
            pass  # already per-position
        else:
            raise ValueError(
                f"Expected `timesteps` to have shape (B,) or (B, T), got {timesteps.shape}."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    # Training-time Real-Time Chunking (RTC) configuration.
    # When max_rtc_delay > 0, the training loop simulates inference delays by
    # randomly choosing a prefix length d ~ Uniform(0, max_rtc_delay) per sample.
    # Prefix positions (0..d-1) are set to clean (t=1.0) and the loss is computed
    # only on the postfix (d..T-1).  At d=0 the sample reduces to standard
    # flow-matching training, so the model stays compatible with non-RTC inference.
    max_rtc_delay: int = field(
        default=0,
        metadata={
            "help": "Maximum prefix delay for training-time RTC. "
            "0 disables RTC (standard training). "
            "When > 0, each training sample randomly picks a delay d in [0, max_rtc_delay] "
            "and only computes loss on the postfix actions."
        },
    )

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg)
            if config.use_vlln
            else nn.Identity()
        )

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        device = vl_embs.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Embed noised action trajectory.
        actions = action_input.action  # (B, T, action_dim)
        B, T, action_d = actions.shape
        if action_d != self.action_dim:
            raise ValueError(
                f"Batch action width {action_d} != model action_dim {self.action_dim} "
                "(check GR00TTransform max_action_dim matches checkpoint action_head_cfg "
                "action_dim, not max_action_dim)."
            )
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(B, device=actions.device, dtype=actions.dtype)  # (B,)

        # ---- Training-time Real-Time Chunking (RTC) ----
        # Reference: "Training-Time Action Conditioning for Efficient Real-Time
        # Chunking" (arXiv:2512.05964), Algorithm 1.
        #
        # When max_rtc_delay > 0 we simulate an inference delay:
        #   - Sample delay d ~ Uniform(0, max_rtc_delay) per batch element.
        #   - Prefix positions (0..d-1) get t=1.0 (clean action), so
        #     noisy_trajectory[prefix] = actions (no noise).
        #   - Postfix positions (d..T-1) get the normally-sampled t.
        #   - Loss is only computed on postfix positions.
        #
        # When d=0 for a given sample (always the case when max_rtc_delay==0),
        # the behaviour is identical to standard flow-matching training.
        use_rtc = getattr(self.config, "max_rtc_delay", 0) > 0
        if use_rtc:
            # Cap at T-1 so every sample has at least one postfix position
            # (otherwise the loss is zero for that sample and no gradient flows).
            max_d = min(self.config.max_rtc_delay, T - 1)
            support_size = max_d + 1  # delay values 0, 1, ..., max_d

            # Exponentially-weighted sampling biased toward small delays, matching
            # Physical Intelligence's TT-RTC: w[k] ∝ exp(max_d - k), so delay=0
            # gets the highest probability. The intuition is that real-world
            # inference latency is usually short, so the model should see "no/short
            # prefix" far more often than "long prefix" during training.
            w = torch.exp(
                torch.arange(support_size, device=device, dtype=torch.float32).flip(0)
            )
            w = w / w.sum()
            delay = torch.multinomial(w, B, replacement=True)  # (B,) in [0, support_size)

            # Build per-position continuous timesteps: (B, T)
            pos_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # (B, T)
            prefix_mask = pos_idx < delay.unsqueeze(1)  # (B, T) bool

            # Per-position t: prefix => 1.0 (clean), postfix => sampled t
            t_per_pos = torch.where(
                prefix_mask,
                torch.ones(B, T, device=device, dtype=t.dtype),
                t.unsqueeze(1).expand(-1, T),
            )  # (B, T)

            # Noisy trajectory with per-position interpolation:
            #   x_t = t * action + (1 - t) * noise
            # For prefix (t=1.0): x_t = action (clean)
            # For postfix: standard flow matching interpolation
            t_expanded = t_per_pos.unsqueeze(-1)  # (B, T, 1)
            noisy_trajectory = (1 - t_expanded) * noise + t_expanded * actions

            # Velocity target is the same for all positions
            velocity = actions - noise

            # Discretize per-position timesteps for the action encoder: (B, T)
            t_discretized_per_pos = (t_per_pos * self.num_timestep_buckets).long()

            # For the DiT's global AdaLN conditioning we use the postfix
            # timestep (same as the sampled t), since that reflects the noise
            # level being denoised.
            t_discretized_global = (t * self.num_timestep_buckets).long()  # (B,)

            # Encode actions with per-position timesteps
            action_features = self.action_encoder(
                noisy_trajectory, t_discretized_per_pos, embodiment_id
            )

            # Build postfix-only loss mask: (B, T, 1) -- expanded to action_dim later
            postfix_mask = (~prefix_mask).unsqueeze(-1).float()  # (B, T, 1)
        else:
            # Standard flow-matching training (no RTC)
            # Explicit (B,1,1) broadcast avoids ambiguous (B,) mixing with (B,T,D).
            t_mix = t.reshape(B, 1, 1)
            noisy_trajectory = (1.0 - t_mix) * noise + t_mix * actions
            velocity = actions - noise
            # Per-batch scalar timestep -> discrete buckets (t is shape (B,) from sample_time).
            t_discretized_global = (t.reshape(B) * self.num_timestep_buckets).long()
            action_features = self.action_encoder(noisy_trajectory, t_discretized_global, embodiment_id)
            postfix_mask = None

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        vl_attn_mask = backbone_output.backbone_attention_mask

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized_global,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Compute loss.
        action_mask = action_input.action_mask  # (B, T, action_dim)
        loss_per_element = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask

        if postfix_mask is not None:
            # RTC: only compute loss on postfix positions
            combined_mask = action_mask * postfix_mask  # (B, T, action_dim)
            loss = (loss_per_element * postfix_mask).sum() / combined_mask.sum().clamp(min=1.0)
        else:
            loss = loss_per_element.sum() / action_mask.sum()

        output_dict = {
            "loss": loss,
        }
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        prefix_actions: torch.Tensor = None,
        num_prefix_steps: int = 0,
    ) -> BatchFeature:
        """
        Generate action predictions via flow-matching denoising.

        When ``prefix_actions`` is provided the method implements the inference
        algorithm from "Training-Time Action Conditioning for Efficient
        Real-Time Chunking" (arXiv:2512.05964, Algorithm 2):

        * Prefix positions (0 .. num_prefix_steps-1) are clamped to the known
          clean actions and their per-token timestep is set to
          ``num_timestep_buckets`` (i.e. t=1.0, fully clean).
        * Postfix positions follow the normal Euler denoising schedule.
        * The DiT's global AdaLN timestep uses the postfix schedule.

        This matches the training-time conditioning (``forward()`` with
        ``max_rtc_delay > 0``) so the model sees the same per-token timestep
        pattern it was trained on.

        Args:
            backbone_output: Output from the vision-language backbone.
            action_input: Action input features (state, embodiment_id, etc.).
            prefix_actions: Optional tensor of shape (B, K, action_dim) containing
                known actions to fix during denoising (inpainting). These are
                typically carried over from a previous prediction for temporal
                consistency in real-time action chunking.
            num_prefix_steps: Number of leading action steps to clamp to
                prefix_actions at each denoising iteration. Must be <= action_horizon.
                Ignored if prefix_actions is None.
        """

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        T = self.config.action_horizon
        actions = torch.randn(
            size=(batch_size, T, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        # Inpainting setup: initialize prefix positions with known actions
        # so the denoising process starts from a better initial point.
        use_inpainting = (
            prefix_actions is not None
            and num_prefix_steps > 0
            and num_prefix_steps <= T
        )
        if use_inpainting:
            # Shape validation: prefix_actions must cover the requested prefix length.
            expected = (batch_size, num_prefix_steps, self.config.action_dim)
            if (
                prefix_actions.shape[0] != batch_size
                or prefix_actions.shape[1] < num_prefix_steps
                or prefix_actions.shape[2] != self.config.action_dim
            ):
                raise ValueError(
                    f"prefix_actions shape {tuple(prefix_actions.shape)} incompatible with "
                    f"(batch={batch_size}, >={num_prefix_steps}, action_dim={self.config.action_dim}); "
                    f"expected at least {expected}."
                )

            # Warn if the model was not trained with TT-RTC: inpainting will be OOD.
            trained_max_delay = getattr(self.config, "max_rtc_delay", 0)
            if trained_max_delay == 0:
                warnings.warn(
                    "prefix_actions provided but the model was trained with "
                    "max_rtc_delay=0 (no TT-RTC). Inpainting will be out-of-distribution "
                    "and results may degrade.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            elif num_prefix_steps > trained_max_delay:
                warnings.warn(
                    f"num_prefix_steps={num_prefix_steps} exceeds the training-time "
                    f"max_rtc_delay={trained_max_delay}; the model has never seen this "
                    f"many clamped prefix tokens and results may degrade.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            prefix_actions = prefix_actions.to(dtype=actions.dtype, device=device)
            actions[:, :num_prefix_steps, :] = prefix_actions[:, :num_prefix_steps, :]

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            if use_inpainting:
                # Per-position timesteps: prefix = num_timestep_buckets (t=1.0,
                # clean), postfix = current denoising step.
                t_per_pos = torch.full(
                    (batch_size, T), fill_value=t_discretized, dtype=torch.long, device=device,
                )
                t_per_pos[:, :num_prefix_steps] = self.num_timestep_buckets

                # Prefix is already clean here: it was seeded at line ~518 before the loop,
                # and re-clamped after every Euler step (line ~577 below). No clamp needed.
                action_features = self.action_encoder(actions, t_per_pos, embodiment_id)
            else:
                # Standard: single timestep for all positions
                timesteps_tensor = torch.full(
                    size=(batch_size,), fill_value=t_discretized, device=device
                )
                action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)

            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            # DiT global timestep: always the postfix denoising timestep
            global_timestep = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=global_timestep,
            )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -T:]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity

            # Clamp prefix positions back after the Euler step.
            if use_inpainting:
                actions[:, :num_prefix_steps, :] = prefix_actions[:, :num_prefix_steps, :]

        return BatchFeature(data={"action_pred": actions})

    @staticmethod
    def build_prefix_weights(
        d: int,
        K: int,
        H: int,
        schedule: str,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Per-position guidance weights for the soft-mask region [d, K).

        Hard region [0, d) and free region [K, H) both get weight=0; the hard
        region is handled separately by overwriting, so guidance there is
        redundant. The soft region's weight decays from 1 at k=d toward 0 at
        k=K-1 according to ``schedule``.

        Returns a tensor of shape (H,).
        """
        w = torch.zeros(H, device=device, dtype=dtype)
        if K <= d:
            return w
        soft_len = K - d
        if soft_len == 1:
            w[d] = 1.0
            return w
        # norm goes 0 at k=d, 1 at k=K-1.
        norm = torch.linspace(0.0, 1.0, soft_len, device=device, dtype=dtype)
        if schedule == "exp":
            w[d:K] = torch.exp(-3.0 * norm)
        elif schedule == "linear":
            w[d:K] = 1.0 - norm
        elif schedule == "ones":
            w[d:K] = 1.0
        elif schedule == "zeros":
            w[d:K] = 0.0
        else:
            raise ValueError(f"unknown prefix_attention_schedule: {schedule!r}")
        return w

    def get_action_with_guidance(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        prefix_actions: torch.Tensor,
        num_prefix_steps: int,
        prefix_attention_horizon: int,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 5.0,
    ) -> BatchFeature:
        """Denoise with hard-clamp prefix + pinv-corrected soft guidance.

        Hard region (positions [0, num_prefix_steps)): overwritten to match
        ``prefix_actions`` exactly after every Euler step, same as the
        Phase-1 path. Per-position timestep is set to ``num_timestep_buckets``
        so the action encoder sees "clean" for these positions.

        Soft region (positions [num_prefix_steps, prefix_attention_horizon)):
        the velocity field is nudged via PI's pinv-corrected guidance toward
        the same ``prefix_actions`` tensor (the model's prediction is pulled
        toward y but not forced). Weights decay according to
        ``prefix_attention_schedule``.

        Free region (positions [prefix_attention_horizon, H)): pure model.

        Reference: PI's ``realtime_action`` in pi-rtc-kinetix/src/model.py,
        adapted to PyTorch + GR00T's DiT.
        """
        backbone_output = self.process_backbone_output(backbone_output)
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        state_features = self.state_encoder(action_input.state, embodiment_id)

        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        T = self.config.action_horizon
        action_dim = self.config.action_dim

        # Validate inputs (server-side contract).
        if prefix_actions.shape[1] < prefix_attention_horizon:
            raise RuntimeError(
                f"guidance: prefix_actions has {prefix_actions.shape[1]} positions "
                f"but prefix_attention_horizon={prefix_attention_horizon}; need >="
            )
        if prefix_attention_horizon > T:
            raise RuntimeError(
                f"guidance: prefix_attention_horizon={prefix_attention_horizon} "
                f"exceeds chunk horizon T={T}"
            )

        prefix_actions = prefix_actions.to(dtype=vl_embs.dtype, device=device)

        # Build the per-position target y. We align the new chunk to the
        # prior chunk's tail: y[k] corresponds to prefix_actions[k]. Only the
        # first prefix_attention_horizon positions matter (the rest have
        # weight=0 anyway), but we zero-pad to length T for shape consistency.
        y = torch.zeros(batch_size, T, action_dim, dtype=vl_embs.dtype, device=device)
        K = min(prefix_attention_horizon, prefix_actions.shape[1])
        y[:, :K, :] = prefix_actions[:, :K, :]

        # Per-position guidance weights. Hard region [0, d) gets weight 0
        # because the hard-clamp overwrite handles that exactly already.
        W = self.build_prefix_weights(
            d=num_prefix_steps,
            K=prefix_attention_horizon,
            H=T,
            schedule=prefix_attention_schedule,
            device=device,
            dtype=vl_embs.dtype,
        )  # (T,)

        # Seed actions from noise; hard region overwritten immediately so the
        # first Euler step sees clean prefix tokens.
        actions = torch.randn(
            size=(batch_size, T, action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )
        actions[:, :num_prefix_steps, :] = prefix_actions[:, :num_prefix_steps, :]

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # We need autograd through the model for the pinv-correction. Make
        # sure model params don't accumulate grads (they shouldn't anyway,
        # since we're in inference, but be defensive).
        with torch.no_grad():
            future_tokens_const = self.future_tokens.weight.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            if self.config.add_pos_embed:
                pos_ids = torch.arange(T, dtype=torch.long, device=device)
                pos_embs_const = self.position_embedding(pos_ids).unsqueeze(0)
            else:
                pos_embs_const = None

        for step in range(num_steps):
            t_cont = step / float(num_steps)
            t_disc = int(t_cont * self.num_timestep_buckets)

            # Per-position discrete time: clean (=num_timestep_buckets) for
            # the hard region, current step otherwise.
            t_per_pos = torch.full(
                (batch_size, T), fill_value=t_disc, dtype=torch.long, device=device
            )
            t_per_pos[:, :num_prefix_steps] = self.num_timestep_buckets

            # ----- Forward with autograd to compute pinv correction -----
            with torch.enable_grad():
                actions_grad = actions.detach().requires_grad_(True)

                action_features = self.action_encoder(
                    actions_grad, t_per_pos, embodiment_id
                )
                if pos_embs_const is not None:
                    action_features = action_features + pos_embs_const

                sa_embs = torch.cat(
                    (state_features, future_tokens_const, action_features), dim=1
                )
                global_timestep = torch.full(
                    size=(batch_size,), fill_value=t_disc, device=device
                )
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=global_timestep,
                )
                pred = self.action_decoder(model_output, embodiment_id)
                v_t = pred[:, -T:]

                # Predicted clean: x_1 = x_t + (1 - t) * v_t
                x_1 = actions_grad + (1.0 - t_cont) * v_t
                # Cotangent: (y - x_1) weighted per-position.
                err = (y - x_1).detach() * W.view(1, T, 1)
                # VJP through (x_t -> x_1): L = <err, x_1>, ∂L/∂x_t = correction.
                # Stop gradient on err so it's just a cotangent.
                L = (err * x_1).sum()
                correction = torch.autograd.grad(L, actions_grad)[0]

            # Detach for the Euler step.
            v_t_d = v_t.detach()

            # Guidance weight λ(t) — PI's formula, capped at max_guidance_weight.
            if t_cont > 1e-6:
                c_t = (1.0 - t_cont) / t_cont
            else:
                c_t = float(max_guidance_weight)
            denom = max((1.0 - t_cont) ** 2, 1e-8)
            inv_r2 = (t_cont ** 2 + (1.0 - t_cont) ** 2) / denom
            lam = min(c_t * inv_r2, float(max_guidance_weight))

            v_corrected = v_t_d + lam * correction

            # Euler step.
            actions = actions.detach() + dt * v_corrected

            # Hard re-clamp: prefix positions must remain exactly prefix_actions
            # regardless of what the Euler step did (guidance has weight 0 there
            # so correction is already ~0, but defensive overwrite is cheap).
            actions[:, :num_prefix_steps, :] = prefix_actions[:, :num_prefix_steps, :]

        return BatchFeature(data={"action_pred": actions})

    @staticmethod
    def build_prefix_weights_it_rtc(
        d: int,
        K: int,
        H: int,
        schedule: str,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """PI-style guidance weights, ported verbatim from PI's
        ``get_prefix_weights`` (pi-rtc-kinetix/src/model.py:40-63).

        Args ``d`` and ``K`` correspond to PI's ``start`` and ``end``.

        With d=2, end=6, total=10, the output is:
            [1, 1, 4/5, 3/5, 2/5, 1/5, 0, 0, 0, 0]
                 ^                ^
               start             end
        (then optionally passed through the exp transform w * (e^w-1)/(e-1)).
        """
        # PI: start = min(start, end)
        start = min(d, K)

        if schedule == "ones":
            w = torch.ones(H, device=device, dtype=dtype)
        elif schedule == "zeros":
            w = (torch.arange(H, device=device) < start).to(dtype)
        elif schedule in ("linear", "exp"):
            arange = torch.arange(H, device=device, dtype=dtype)
            # PI: w = clip((start - 1 - arange) / (end - start + 1) + 1, 0, 1)
            w = torch.clamp((start - 1 - arange) / (K - start + 1) + 1, 0.0, 1.0)
            if schedule == "exp":
                # PI: w = w * expm1(w) / (e - 1)
                e_minus_1 = float(torch.e) - 1.0
                w = w * torch.expm1(w) / e_minus_1
        else:
            raise ValueError(f"unknown prefix_attention_schedule: {schedule!r}")

        # PI: zero past `end`
        w = torch.where(torch.arange(H, device=device) >= K, torch.zeros_like(w), w)
        return w

    def get_action_pure_it_rtc(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        prior_chunk_slice: torch.Tensor,
        inference_delay: int,
        prefix_attention_horizon: int,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 5.0,
    ) -> BatchFeature:
        """Pure PI IT-RTC: gradient-guided denoising, no prefix overwriting.

        Mirrors ``pi-rtc-kinetix/src/model.py`` ``realtime_action`` when
        ``simulated_delay is None`` — the published IT-RTC algorithm.

        Differences vs ``get_action_with_guidance`` (the hybrid):
        * No prefix overwriting at any point (init from pure noise, stay free).
        * Uniform discrete timestep across all positions (no t=1.0 per-position).
        * Weights cover [0, K) with PI's schedule (W[0:d]=1.0).
        * The model is fed in-distribution inputs throughout, so the autograd
          VJP through the DiT gives meaningful gradients.

        Trade-off: positions [0, d) are no longer bit-exact equal to
        prior_chunk_slice — they are *strongly pulled* via weight=1.0 but
        the constraint is finite. For the safety property in async RTC, this
        is acceptable because the ROS client slices off the first d_actual
        positions before queueing, so the robot never executes them anyway.

        Args:
            prior_chunk_slice: (B, L, action_dim) with L >= prefix_attention_horizon.
                Already sliced by the policy to start at c+1 of the cached
                prior chunk. Only the first ``prefix_attention_horizon``
                positions are used (the rest are ignored).
        """
        backbone_output = self.process_backbone_output(backbone_output)
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        state_features = self.state_encoder(action_input.state, embodiment_id)

        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        T = self.config.action_horizon
        action_dim = self.config.action_dim
        d = inference_delay
        K = prefix_attention_horizon

        if K > T:
            raise RuntimeError(
                f"pure IT-RTC: prefix_attention_horizon={K} exceeds T={T}"
            )
        if prior_chunk_slice.shape[1] < K:
            raise RuntimeError(
                f"pure IT-RTC: prior_chunk_slice has {prior_chunk_slice.shape[1]} "
                f"positions but K={K}"
            )

        prior_chunk_slice = prior_chunk_slice.to(dtype=vl_embs.dtype, device=device)

        # Target y of shape (B, T, action_dim). y[k] = prior[k] for k in [0, K).
        # Past K the weight is zero so y[K:] is don't-care; we set it to 0.
        y = torch.zeros(batch_size, T, action_dim, dtype=vl_embs.dtype, device=device)
        y[:, :K, :] = prior_chunk_slice[:, :K, :]

        # PI's weights: 1.0 in [0, d), decay in [d, K), 0 in [K, T).
        W = self.build_prefix_weights_it_rtc(
            d=d, K=K, H=T,
            schedule=prefix_attention_schedule,
            device=device, dtype=vl_embs.dtype,
        )

        # Seed from pure noise. No prefix overwriting at any point in this method.
        actions = torch.randn(
            size=(batch_size, T, action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Precompute frozen tensors used inside the autograd region.
        with torch.no_grad():
            future_tokens_const = self.future_tokens.weight.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            if self.config.add_pos_embed:
                pos_ids = torch.arange(T, dtype=torch.long, device=device)
                pos_embs_const = self.position_embedding(pos_ids).unsqueeze(0)
            else:
                pos_embs_const = None

        for step in range(num_steps):
            t_cont = step / float(num_steps)
            t_disc = int(t_cont * self.num_timestep_buckets)

            # Uniform timestep across all positions — no per-position trick.
            timesteps_tensor = torch.full(
                (batch_size,), fill_value=t_disc, dtype=torch.long, device=device
            )

            with torch.enable_grad():
                actions_grad = actions.detach().requires_grad_(True)

                action_features = self.action_encoder(
                    actions_grad, timesteps_tensor, embodiment_id
                )
                if pos_embs_const is not None:
                    action_features = action_features + pos_embs_const

                sa_embs = torch.cat(
                    (state_features, future_tokens_const, action_features), dim=1
                )
                global_timestep = torch.full(
                    (batch_size,), fill_value=t_disc, device=device
                )
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=global_timestep,
                )
                pred = self.action_decoder(model_output, embodiment_id)
                v_t = pred[:, -T:]

                # Predicted clean: x_1 = x_t + (1 - t) * v_t.
                x_1 = actions_grad + (1.0 - t_cont) * v_t
                err = (y - x_1).detach() * W.view(1, T, 1)
                L = (err * x_1).sum()
                correction = torch.autograd.grad(L, actions_grad)[0]

            v_t_d = v_t.detach()

            # Guidance weight λ(t) — PI's formula, capped at max_guidance_weight.
            if t_cont > 1e-6:
                c_t = (1.0 - t_cont) / t_cont
            else:
                c_t = float(max_guidance_weight)
            denom = max((1.0 - t_cont) ** 2, 1e-8)
            inv_r2 = (t_cont ** 2 + (1.0 - t_cont) ** 2) / denom
            lam = min(c_t * inv_r2, float(max_guidance_weight))

            v_corrected = v_t_d + lam * correction

            actions = actions.detach() + dt * v_corrected

        # NO overwriting — purely guided. [0, d) is strongly pulled but not exact.
        return BatchFeature(data={"action_pred": actions})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
