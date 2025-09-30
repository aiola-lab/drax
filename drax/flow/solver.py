# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

# Modifications copyright (c) 2025 aiOla
# adapted from https://github.com/facebookresearch/flow_matching/blob/main/flow_matching/solver/discrete_solver.py
# changes: added support for preserving tokens

from collections.abc import Callable
from contextlib import nullcontext
from math import ceil

import torch
from flow_matching.path.scheduler import ConvexScheduler
from flow_matching.solver.solver import Solver
from flow_matching.solver.utils import get_nearest_times
from flow_matching.utils import ModelWrapper, categorical
from torch import Tensor
from tqdm import tqdm


class DraxMixtureDiscreteEulerSolver(Solver):
    r"""Similar to the original solver, but with support for preserving tokens"""

    def __init__(
        self,
        model: ModelWrapper,
        scheduler: ConvexScheduler,
        vocabulary_size: int,
        source_distribution_p: Tensor | None = None,
    ):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.vocabulary_size = vocabulary_size

        if source_distribution_p is not None:
            assert source_distribution_p.shape == torch.Size(
                [vocabulary_size]
            ), f"Source distribution p dimension must match the vocabulary size {vocabulary_size}. Got {source_distribution_p.shape}."

        self.source_distribution_p = source_distribution_p

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        step_size: float | None,
        div_free: float | Callable[[float], float] = 0.0,
        dtype_categorical: torch.dtype = torch.float32,
        time_grid: Tensor | None = None,
        return_intermediates: bool = False,
        verbose: bool = False,
        **model_extras,
    ) -> Tensor:
        preserve_mask = model_extras.get("preserve_mask", None)
        if preserve_mask is not None:
            preserve_mask = preserve_mask.to(device=x_init.device).bool()

        if not div_free == 0.0:
            assert self.source_distribution_p is not None, "Source distribution p must be specified in order to add a divergence-free term to the probability velocity."

        # Initialize the current state `x_t` with the initial state `X_0`.
        if time_grid is None:
            time_grid = torch.tensor([0.0, 1.0], device=x_init.device)
        else:
            time_grid = time_grid.to(device=x_init.device)

        if step_size is None:
            # If step_size is None then set the t discretization to time_grid.
            t_discretization = time_grid
            n_steps = len(time_grid) - 1
        else:
            # If step_size is float then t discretization is uniform with step size set by step_size.
            t_init = time_grid[0].item()
            t_final = time_grid[-1].item()
            assert (
                t_final - t_init
            ) > step_size, f"Time interval [time_grid[0], time_grid[-1]] must be larger than step_size. Got a time interval [{t_init}, {t_final}] and step_size {step_size}."

            n_steps = ceil((t_final - t_init) / step_size)
            t_discretization = torch.tensor(
                [t_init + step_size * i for i in range(n_steps)] + [t_final],
                device=x_init.device,
            )

            if return_intermediates:
                # get order of intermediate steps:
                order = torch.argsort(time_grid)
                # Compute intermediate steps to return via nearest points in t_discretization to time_grid.
                time_grid = get_nearest_times(time_grid=time_grid, t_discretization=t_discretization)

        x_t = x_init.clone()
        steps_counter = 0
        res = []

        if return_intermediates:
            res = [x_init.clone()]

        if verbose:
            ctx = tqdm(total=t_final, desc=f"NFE: {steps_counter}")
        else:
            ctx = nullcontext()

        with ctx:
            for i in range(n_steps):
                t = t_discretization[i : i + 1]
                h = t_discretization[i + 1 : i + 2] - t_discretization[i : i + 1]

                # Sample x_1 ~ p_1|t( \cdot |x_t)
                p_1t = self.model(x=x_t, t=t.repeat(x_t.shape[0]), **model_extras)
                x_1 = categorical(p_1t.to(dtype=dtype_categorical))
                if preserve_mask is not None:
                    # preseve_mask is a boolean mask of the same shape as x_t. We assume x_init contains the preserve_token_ids at the preserve_mask positions.
                    x_1 = torch.where(preserve_mask, x_init, x_1)
                    x_t = torch.where(preserve_mask, x_init, x_t)

                # Checks if final step
                if i == n_steps - 1:
                    x_t = x_1
                else:
                    # Compute u_t(x|x_t,x_1)
                    scheduler_output = self.scheduler(t=t)

                    k_t = scheduler_output.alpha_t
                    d_k_t = scheduler_output.d_alpha_t

                    delta_1 = torch.nn.functional.one_hot(x_1, num_classes=self.vocabulary_size).to(k_t.dtype)
                    u = d_k_t / (1 - k_t) * delta_1

                    # Add divergence-free part
                    div_free_t = div_free(t) if callable(div_free) else div_free

                    if div_free_t > 0:
                        p_0 = self.source_distribution_p[(None,) * x_t.dim()]
                        u = u + div_free_t * d_k_t / (k_t * (1 - k_t)) * ((1 - k_t) * p_0 + k_t * delta_1)

                    # Set u_t(x_t|x_t,x_1) = 0
                    delta_t = torch.nn.functional.one_hot(x_t, num_classes=self.vocabulary_size)
                    u = torch.where(delta_t.to(dtype=torch.bool), torch.zeros_like(u), u)

                    # Sample x_t ~ u_t( \cdot |x_t,x_1)
                    intensity = u.sum(dim=-1)  # Assuming u_t(xt|xt,x1) := 0
                    mask_jump = torch.rand(size=x_t.shape, device=x_t.device) < 1 - torch.exp(-h * intensity)

                    if mask_jump.sum() > 0:
                        x_t[mask_jump] = categorical(u[mask_jump].to(dtype=dtype_categorical))

                steps_counter += 1
                t = t + h

                if return_intermediates and (t in time_grid):
                    res.append(x_t.clone())

                if verbose:
                    ctx.n = t.item()
                    ctx.refresh()
                    ctx.set_description(f"NFE: {steps_counter}")

        if return_intermediates:
            if step_size is None:
                return torch.stack(res, dim=0)
            else:
                return torch.stack(res, dim=0)[order]
        else:
            return x_t
