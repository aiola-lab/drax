# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

# Modifications copyright (c) 2025 aiOla
# adapted from https://github.com/facebookresearch/flow_matching/blob/main/examples/text/logic/flow.py
# changes: added abstract methods

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class SourceDistribution(ABC):
    """Base class for source distributions."""

    is_learnable: bool = False  # Whether the distribution has learnable parameters
    supports_audio_conditioning: bool = False  # Whether the distribution supports audio conditioning

    @abstractmethod
    def sample(self, *args, **kwargs):
        """Sample from the distribution; must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def sample_like(self, x):
        """Sample with the same shape/device as `x`; must be implemented by subclasses."""
        raise NotImplementedError


class MaskedSourceDistribution(SourceDistribution):
    def __init__(self, mask_token: int) -> None:
        self.mask_token = mask_token

    @property
    def masked(self) -> bool:
        return True

    def sample(self, tensor_size: tuple[int, ...], device: torch.device) -> Tensor:
        return torch.zeros(tensor_size, device=device).fill_(self.mask_token).long()

    def sample_like(self, tensor_like: Tensor) -> Tensor:
        return torch.zeros_like(tensor_like).fill_(self.mask_token).long()


class UniformSourceDistribution(SourceDistribution):
    """Uniform source distribution."""

    is_learnable = False
    supports_audio_conditioning = False

    def __init__(self, vocab_size: int, masked: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.masked = masked

    def sample(self, tensor_size: tuple[int, ...], device: torch.device) -> Tensor:
        return torch.randint(size=tensor_size, high=self.vocab_size, device=device)

    def sample_like(self, tensor_like: Tensor) -> Tensor:
        return torch.randint_like(tensor_like, high=self.vocab_size)
