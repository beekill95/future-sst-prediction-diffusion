from __future__ import annotations

import torch

from .beta_scheduler import BetaScheduler


class ForwardSampler:
    def __init__(self, beta_scheduler: BetaScheduler) -> None:
        self._beta_scheduler = beta_scheduler

    def __call__(self, sst: torch.Tensor, time_steps: torch.Tensor):
        return self.sample(sst, time_steps)

    @property
    def max_time_steps(self):
        return self._beta_scheduler.max_time_steps

    @torch.no_grad()
    def sample(self, sst: torch.Tensor, time_steps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform diffusion forward process based on the nice property:
        q(x_t | x_0) = Normal(x_t; \sqrt{\bar{\alpha_t}}x_0, (1 - \bar{\alpha_t})I

        Parameters
        ==========
        sst: torch.Tensor
            The original SST (x_0), can be unbatched input (C, H, W) or batched input (N, C, H, W).
        time_steps: torch.Tensor
            The number of time steps (t) to perform forward sampling,
            can be a tensor of 1 value for unbatched |sst| or a tensor (N,) for batched |sst|.
        """
        is_batched_input = len(sst.shape) == 4
        if is_batched_input:
            assert sst.shape[0] == time_steps.shape[0]
        else:
            sst = sst[None, ...]
            time_steps = time_steps[None, ...]

        # Generate random noise.
        noise = torch.randn_like(sst)

        # Generate noisy SSTs.
        sqrt_alpha_bar = self._beta_scheduler.sqrt_alpha_bar(time_steps)[..., None, None]
        sqrt_1_minus_alpha_bar = self._beta_scheduler.sqrt_1_minus_alpha_bar(time_steps)[..., None, None]
        noisy_sst = sqrt_alpha_bar * sst + noise * sqrt_1_minus_alpha_bar

        return (noisy_sst, noise) if is_batched_input else (noisy_sst[0], noise[0])
