from __future__ import annotations

import torch
import torch.nn as nn

from .beta_scheduler import BetaScheduler


class BackwardSampler:
    def __init__(self, model: nn.Module, beta_scheduler: BetaScheduler, device: str) -> None:
        self._model = model
        self._beta_scheduler = beta_scheduler
        self._device = device

    @torch.no_grad()
    def backward(self, noisy_sst: torch.Tensor, time_step: int) -> torch.Tensor:
        """
        Perform diffusion (unconditional) backward process (the sampling algorithm in the paper).

        Parameters
        ==========
        noisy_sst: torch.Tensor
            Noisy SST, a tensor of shape (N, C, H, W) for batched input or (C, H, W) for unbatched input.
        time_step: int
            Time step (diffusion time steps) of all SSTs.
        """
        self._model.eval()
        is_batched_input = len(noisy_sst.shape) == 4
        if is_batched_input:
            batch_size = noisy_sst.shape[0]
            time_steps = torch.tensor([time_step]*batch_size)
        else:
            noisy_sst = noisy_sst[None, ...]
            time_steps = torch.tensor([time_step])

        pred_noise = self._model(noisy_sst.to(self._device), time_steps.to(self._device)).cpu()
        means = self._means(noisy_sst, pred_noise, time_steps)

        if time_step == 0:
            return means if is_batched_input else means[0]
        else:
            posterior_variances = self._posterior_variances(time_steps)
            random_noise = torch.randn_like(noisy_sst)

            x = means + random_noise * torch.sqrt(posterior_variances)
            return x if is_batched_input else x[0]

    def _means(self, noisy_sst: torch.Tensor, pred_noise: torch.Tensor, time_steps: torch.Tensor):
        """
        Calculate mean.
        """
        # TODO: how to handle time_steps = 0?
        sqrt_alpha_recip = 1. / torch.sqrt(self._beta_scheduler.alpha(time_steps))
        sqrt_1_minus_alpha_bar = self._beta_scheduler.sqrt_1_minus_alpha_bar(time_steps)
        beta = self._beta_scheduler.beta(time_steps)

        return sqrt_alpha_recip * (noisy_sst - beta * pred_noise / sqrt_1_minus_alpha_bar)

    def _posterior_variances(self, time_steps: torch.Tensor) -> torch.Tensor:
        """
        Calculate posterior variances.

        Parameters
        ==========
        time_steps: torch.Tensor
            Time steps, can be a scalar tensor or a tensor of shape (N,).
        """
        # TODO: how to handle time_steps = 0?
        beta = self._beta_scheduler.beta(time_steps)
        one_minus_alpha_bar_t = self._beta_scheduler.one_minus_alpha_bar(time_steps)
        one_minus_alpha_bat_t_1 = self._beta_scheduler.one_minus_alpha_bar(time_steps - 1)
        return beta * one_minus_alpha_bat_t_1 / one_minus_alpha_bar_t
