# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Part of this implementation is adapted from https://github.com/facebookresearch/DiT
# which is released under NonCommercial-4.0 license
# Part of this implementation is adapted from https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
# which is released under MIT license
# Part of this implementation is adapted from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
# which is released under MIT license

# Modifications copyright (c) 2025 aiOla
# adapted from https://github.com/facebookresearch/flow_matching/blob/main/examples/text/model/transformer.py
# changes: added support for audio branch via cross attention and CFG

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch import Tensor, nn

from . import rotary


def bias_dropout_add_scale(x: Tensor, scale: Tensor, residual: Tensor | None, prob: float, training: bool) -> Tensor:
    return residual + scale * F.dropout(x, p=prob, training=training)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            x = F.layer_norm(x.float(), [self.dim])

        return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(time: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        """Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=time.device)
        args = time[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, time: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(time=time, dim=self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        cond_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be devisable by n_heads"

        self.n_heads = n_heads
        self.dim = dim
        self.dropout = dropout
        self.head_dim = self.dim // self.n_heads

        # Self attention
        self.norm1 = LayerNorm(dim=dim)
        self.qw = nn.Linear(dim, dim, bias=False)
        self.kw = nn.Linear(dim, dim, bias=False)
        self.vw = nn.Linear(dim, dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        # Cross attention (always enabled)
        self.norm_cross = LayerNorm(dim=dim)
        self.norm_audio = LayerNorm(dim=dim)
        self.q_cross = nn.Linear(dim, dim, bias=False)
        self.k_cross = nn.Linear(dim, dim, bias=False)
        self.v_cross = nn.Linear(dim, dim, bias=False)
        self.cross_out = nn.Linear(dim, dim, bias=False)

        # MLP
        self.norm2 = LayerNorm(dim=dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )

        # AdaLN modulation (9 parameters for msa/cross/mlp)
        n_params = 9
        self.adaLN_modulation = nn.Linear(cond_dim, n_params * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(
        self,
        x: Tensor,
        rotary_cos_sin: Tensor,
        c: Tensor,
        audio: Tensor,
        audio_k: Tensor | None = None,
        audio_v: Tensor | None = None,
    ) -> Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Get modulation parameters for all layers (9 or 6 total)
        modulation_params = self.adaLN_modulation(c)[:, None].chunk(9, dim=2)
        shift_msa, scale_msa, gate_msa, shift_cross, scale_cross, gate_cross, shift_mlp, scale_mlp, gate_mlp = modulation_params

        # Self attention
        x_skip = x
        x = modulate(x=self.norm1(x), shift=shift_msa, scale=scale_msa)

        q = self.qw(x)
        k = self.kw(x)
        v = self.vw(x)

        q, k, v = (item.view(batch_size, seq_len, self.n_heads, self.head_dim) for item in (q, k, v))

        with torch.amp.autocast("cuda", enabled=False):
            cos, sin = rotary_cos_sin
            original_dtype = q.dtype

            q = rotary.apply_rotary_emb_torch(x=q.float(), cos=cos.float(), sin=sin.float()).to(original_dtype)
            k = rotary.apply_rotary_emb_torch(x=k.float(), cos=cos.float(), sin=sin.float()).to(original_dtype)

        q, k, v = (item.transpose(1, 2) for item in (q, k, v))
        x = F.scaled_dot_product_attention(query=q, key=k, value=v)
        x = rearrange(x, "b h s d -> b s (h d)", b=batch_size)
        x = bias_dropout_add_scale(
            x=self.attn_out(x),
            scale=gate_msa,
            residual=x_skip,
            prob=self.dropout,
            training=self.training,
        )

        # Cross attention (always)
        x_skip = x
        x = modulate(x=self.norm_cross(x), shift=shift_cross, scale=scale_cross)
        q = self.q_cross(x)
        if (audio_k is not None) and (audio_v is not None):
            k = audio_k
            v = audio_v
        else:
            audio = self.norm_audio(audio)
            k = self.k_cross(audio)
            v = self.v_cross(audio)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.n_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.n_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.n_heads)

        x = F.scaled_dot_product_attention(query=q, key=k, value=v)
        x = rearrange(x, "b h s d -> b s (h d)")
        x = bias_dropout_add_scale(
            x=self.cross_out(x),
            scale=gate_cross,
            residual=x_skip,
            prob=self.dropout,
            training=self.training,
        )

        # MLP
        x = bias_dropout_add_scale(
            x=self.mlp(modulate(x=self.norm2(x), shift=shift_mlp, scale=scale_mlp)),
            scale=gate_mlp,
            residual=x,
            prob=self.dropout,
            training=self.training,
        )

        return x


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate(x=self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)

        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, masked: bool, config: DictConfig):
        super().__init__()

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        self.config = config
        self.vocab_size = vocab_size

        add_token = 1 if masked else 0

        self.vocab_embed = nn.Embedding(self.vocab_size + add_token, config.hidden_size)

        self.time_embedding = TimestepEmbedder(hidden_size=config.cond_dim)

        # Audio projection to match hidden size
        self.audio_proj = nn.Sequential(
            nn.Linear(config.whisper_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

        self.rotary_emb = rotary.Rotary(dim=config.hidden_size // config.n_heads)

        # Preserve embeddings
        self.preserve_embeddings = nn.Embedding(2, config.hidden_size)

        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    dim=config.hidden_size,
                    n_heads=config.n_heads,
                    cond_dim=config.cond_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.n_blocks)
            ]
        )

        self.output_layer = DDitFinalLayer(
            hidden_size=config.hidden_size,
            out_channels=vocab_size + add_token,
            cond_dim=config.cond_dim,
        )

    def forward(
        self,
        x_t: Tensor,
        time: Tensor,
        audio_embeddings: Tensor | None = None,
        preserve_mask: Tensor = None,
        audio_projected: Tensor | None = None,
        audio_k_all: Tensor | None = None,
        audio_v_all: Tensor | None = None,
        cfg_strength: float = 1.0,
        audio_drop_prob: float = 0.0,
        use_gradient_checkpointing: bool = False,
    ) -> Tensor:
        if audio_embeddings is None:
            assert (
                audio_projected is not None and audio_k_all is not None and audio_v_all is not None
            ), "audio_embeddings, audio_projected, audio_k_all, and audio_v_all must be provided if audio_embeddings is None"
            assert not self.training, "audio_embeddings must be provided in training mode"
            if audio_projected.shape[0] != x_t.shape[0]:
                raise ValueError(f"Batch size mismatch: x_t={x_t.shape[0]}, audio_projected={audio_projected.shape[0]}")
            if audio_k_all.shape[0] != x_t.shape[0]:
                raise ValueError(f"Batch size mismatch: x_t={x_t.shape[0]}, audio_k_all={audio_k_all.shape[0]}")
            if audio_v_all.shape[0] != x_t.shape[0]:
                raise ValueError(f"Batch size mismatch: x_t={x_t.shape[0]}, audio_v_all={audio_v_all.shape[0]}")

        elif audio_embeddings.shape[0] != x_t.shape[0]:
            raise ValueError(f"Batch size mismatch: x_t={x_t.shape[0]}, audio_embeddings={audio_embeddings.shape[0]}")

        if self.training:
            # Training mode with audio dropout
            if audio_drop_prob > 0:
                # Create dropout mask without modifying input tensor
                mask = torch.rand(x_t.shape[0], device=x_t.device) < audio_drop_prob
                audio_embeddings = torch.where(mask.unsqueeze(-1).unsqueeze(-1), torch.zeros_like(audio_embeddings), audio_embeddings)

            return self._forward(
                x_t=x_t,
                time=time,
                audio_embeddings=audio_embeddings,
                preserve_mask=preserve_mask,
                audio_projected=audio_projected,
                audio_k_all=audio_k_all,
                audio_v_all=audio_v_all,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )

        elif cfg_strength == 1.0:
            # Regular inference mode
            return self._forward(
                x_t=x_t,
                time=time,
                audio_embeddings=audio_embeddings,
                preserve_mask=preserve_mask,
                audio_projected=audio_projected,
                audio_k_all=audio_k_all,
                audio_v_all=audio_v_all,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )

        else:
            # CFG inference mode - not training and cfg_strength > 0
            # Create unconditioned batch (zeros for audio embeddings)
            # NOTE: we use a similar approach to LlamaGen:
            # https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/models/generate.py#L96-L97
            uncond_audio_embeddings = torch.zeros_like(audio_embeddings)

            # Concatenate conditioned and unconditioned batches
            double_time = torch.cat([time, time], dim=0)
            double_x_t = torch.cat([x_t, x_t], dim=0)
            double_preserve_mask = torch.cat([preserve_mask, preserve_mask], dim=0) if preserve_mask is not None else None

            # Build or combine caches
            with torch.no_grad():
                cond_cache = (
                    self.build_audio_cache(audio_embeddings)
                    if audio_projected is None
                    else {
                        "audio_projected": audio_projected,
                        "audio_k_all": audio_k_all,
                        "audio_v_all": audio_v_all,
                    }
                )
                uncond_cache = self.build_audio_cache(uncond_audio_embeddings)

            double_proj = torch.cat([cond_cache["audio_projected"], uncond_cache["audio_projected"]], dim=0)
            double_k_all = torch.cat([cond_cache["audio_k_all"], uncond_cache["audio_k_all"]], dim=0)
            double_v_all = torch.cat([cond_cache["audio_v_all"], uncond_cache["audio_v_all"]], dim=0)

            # Get logits for both conditioned and unconditioned using cache
            double_logits = self._forward(
                x_t=double_x_t,
                time=double_time,
                audio_embeddings=audio_embeddings,  # unused when projected tensors are provided
                preserve_mask=double_preserve_mask,
                audio_projected=double_proj,
                audio_k_all=double_k_all,
                audio_v_all=double_v_all,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )

            # Split results
            cond_logits, uncond_logits = double_logits.chunk(2, dim=0)

            # Apply CFG formula
            logits = uncond_logits + cfg_strength * (cond_logits - uncond_logits)

            return logits

    # Internal forward method for inference
    def _forward(
        self,
        x_t: Tensor,
        time: Tensor,
        audio_embeddings: Tensor,
        preserve_mask: Tensor = None,
        audio_projected: Tensor | None = None,
        audio_k_all: Tensor | None = None,
        audio_v_all: Tensor | None = None,
        use_gradient_checkpointing: bool = False,
    ) -> Tensor:
        # Handle both one-hot and index inputs
        if x_t.dim() == 3:  # one-hot input
            # Convert one-hot to embeddings directly
            x = torch.matmul(x_t, self.vocab_embed.weight)
        else:  # index input
            # Ensure indices are long type
            x = self.vocab_embed(x_t.long())

        # Add preserve embeddings
        if preserve_mask is not None:
            x = x + self.preserve_embeddings(preserve_mask.long())

        # Project audio to hidden dimension (or use provided tensors)
        if audio_projected is not None:
            audio = audio_projected
        else:
            audio = self.audio_proj(audio_embeddings)

        # Get time embeddings
        c = F.silu(self.time_embedding(time=time))

        rotary_cos_sin = self.rotary_emb(x=x)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                if use_gradient_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(self.blocks[i], x, rotary_cos_sin, c, audio, None, None, use_reentrant=False)
                else:
                    audio_k = audio_k_all[:, i] if audio_k_all is not None else None
                    audio_v = audio_v_all[:, i] if audio_v_all is not None else None
                    x = self.blocks[i](x=x, rotary_cos_sin=rotary_cos_sin, c=c, audio=audio, audio_k=audio_k, audio_v=audio_v)

        # Apply final layer with full precision
        with torch.amp.autocast("cuda", dtype=torch.float32):
            x = self.output_layer(x=x, c=c)

        return x
