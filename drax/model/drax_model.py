# Copyright (c) aiOla.ai
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file
from torch import nn
from transformers import WhisperConfig, WhisperModel

from .transformer import Transformer


class Drax(nn.Module):
    """DRAX model = Audio Encoder (HF) + Flow Decoder (`Transformer`).

    This wrapper provides a lightweight HF-style API to save and load the
    combined system without bundling the encoder weights. The saved config
    references the encoder by its HF identifier.
    """

    def __init__(
        self,
        audio_encoder: nn.Module,
        decoder: Transformer,
        decoder_config: dict[str, Any],
        whisper_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.audio_encoder = audio_encoder
        self.decoder = decoder
        self.decoder_config = decoder_config
        self.whisper_config = whisper_config

    @torch.no_grad()
    def encode_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Convert preprocessed Whisper input features to encoder hidden states.
        Assumes `audio_encoder` is a Whisper encoder.
        """
        return self.audio_encoder(audio_features).last_hidden_state

    @torch.no_grad()
    def build_audio_cache(self, audio_embeddings: torch.Tensor) -> dict:
        """Precompute audio projection and per-block cross-attention K/V for fixed audio.
        Returns a dict with three tensors suitable for replication by solvers:
          - audio_projected: [B, L, H]
          - audio_k_all: [B, num_blocks, L, H]
          - audio_v_all: [B, num_blocks, L, H]
        For blocks without cross attention, K/V entries are zeros.
        """
        B, L, H = audio_embeddings.shape
        projected_audio = self.decoder.audio_proj(audio_embeddings)
        num_blocks = len(self.decoder.blocks)
        # Derive feature dim from a sample projection
        feat_dim = projected_audio.size(-1)
        k_all = projected_audio.new_zeros((B, num_blocks, L, feat_dim))
        v_all = projected_audio.new_zeros((B, num_blocks, L, feat_dim))
        for i, blk in enumerate(self.decoder.blocks):
            norm_a = blk.norm_audio(projected_audio)
            k = blk.k_cross(norm_a)
            v = blk.v_cross(norm_a)
            k_all[:, i] = k
            v_all[:, i] = v

        return {"audio_projected": projected_audio, "audio_k_all": k_all, "audio_v_all": v_all}

    def to_dict_config(self) -> dict[str, Any]:
        return {
            "model_type": "drax",
            "decoder_config": self.decoder_config,
            "whisper_config": self.whisper_config,
        }

    def save_pretrained(self, save_directory: str) -> None:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Save config
        config_path = save_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict_config(), f, indent=2)

        # Save decoder weights (safetensors)
        decoder_weights = self.decoder.state_dict()
        safetensors_save_file(decoder_weights, str(save_dir / "decoder.safetensors"))

        # Save encoder weights (safetensors)
        encoder_weights = self.audio_encoder.state_dict()
        safetensors_save_file(encoder_weights, str(save_dir / "encoder.safetensors"))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        device: torch.device | None = None,
        torch_dtype: torch.dtype | None = None,
        cache_dir: str | None = None,
    ) -> "Drax":
        """Load Drax from a local directory or the Hugging Face Hub.

        Expects a repo or directory containing:
          - config.json with keys: decoder_config, whisper_config
          - decoder.safetensors (or decoder.bin)
          - encoder.safetensors (or encoder.bin)
        """
        base_dir = Path(pretrained_model_name_or_path)
        is_local = base_dir.is_dir()

        # Resolve config path (local or hub)
        if is_local:
            config_path = base_dir / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Missing config.json at {config_path}")
        else:
            config_path = Path(
                hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename="config.json",
                    cache_dir=cache_dir,
                )
            )

        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        if cfg.get("model_type") != "drax":
            raise ValueError("Config model_type != 'drax'")

        decoder_cfg = cfg["decoder_config"]
        whisper_cfg_dict = cfg["whisper_config"]

        # Build Whisper encoder locally from saved config and weights
        whisper_config = WhisperConfig(**whisper_cfg_dict)
        whisper_model = WhisperModel(whisper_config)
        audio_encoder = whisper_model.encoder
        if device is not None or torch_dtype is not None:
            audio_encoder = audio_encoder.to(device=device or torch.device("cpu"), dtype=torch_dtype)
        audio_encoder.eval()
        # Load encoder weights
        if is_local:
            enc_weights_path = base_dir / "encoder.safetensors"
            if enc_weights_path.exists():
                enc_state = safetensors_load_file(str(enc_weights_path))
            else:
                enc_weights_path = base_dir / "encoder.bin"
                if not enc_weights_path.exists():
                    raise FileNotFoundError(f"Missing encoder weights at {pretrained_model_name_or_path}")
                enc_state = torch.load(str(enc_weights_path), map_location=device or "cpu")
        else:
            # prefer safetensors
            try:
                enc_file = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename="encoder.safetensors",
                    cache_dir=cache_dir,
                )
                enc_state = safetensors_load_file(enc_file)
            except Exception:
                enc_file = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename="encoder.bin",
                    cache_dir=cache_dir,
                )
                enc_state = torch.load(enc_file, map_location=device or "cpu")
        audio_encoder.load_state_dict(enc_state, strict=True)

        # Build decoder Transformer from config
        vocab_size = decoder_cfg.get("vocab_size")
        masked = decoder_cfg.get("masked", False)
        decoder = Transformer(vocab_size=vocab_size, masked=masked, config=decoder_cfg)
        if device is not None or torch_dtype is not None:
            decoder = decoder.to(device=device or torch.device("cpu"), dtype=torch_dtype)
        decoder.eval()

        # Load decoder weights
        if is_local:
            dec_weights_path = base_dir / "decoder.safetensors"
            if dec_weights_path.exists():
                decoder_state = safetensors_load_file(str(dec_weights_path))
            else:
                dec_weights_path = base_dir / "decoder.bin"
                if not dec_weights_path.exists():
                    raise FileNotFoundError(f"Missing decoder weights at {pretrained_model_name_or_path}")

                decoder_state = torch.load(str(dec_weights_path), map_location=device or "cpu")
        else:
            try:
                dec_file = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename="decoder.safetensors",
                    cache_dir=cache_dir,
                )
                decoder_state = safetensors_load_file(dec_file)
            except Exception:
                dec_file = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename="decoder.bin",
                    cache_dir=cache_dir,
                )
                decoder_state = torch.load(dec_file, map_location=device or "cpu")
        decoder.load_state_dict(decoder_state, strict=False)

        return cls(
            audio_encoder=audio_encoder,
            decoder=decoder,
            decoder_config=decoder_cfg,
            whisper_config=whisper_cfg_dict,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        audio_features: torch.Tensor | None = None,
        preserve_mask: torch.Tensor | None = None,
        audio_projected: torch.Tensor | None = None,
        audio_k_all: torch.Tensor | None = None,
        audio_v_all: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if audio_projected is not None and audio_k_all is not None and audio_v_all is not None:
            audio_embeddings = None
        else:
            audio_embeddings = self.encode_audio(audio_features)
        logits = self.decoder(
            x_t=x,
            time=t,
            audio_embeddings=audio_embeddings,
            preserve_mask=preserve_mask,
            audio_projected=audio_projected,
            audio_k_all=audio_k_all,
            audio_v_all=audio_v_all,
        )
        return torch.softmax(logits.float() / temperature, -1)
