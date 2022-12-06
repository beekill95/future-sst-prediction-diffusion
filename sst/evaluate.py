from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from .conditional_backward_sampler import ConditionalBackwardSampler
from .dataset.transforms import ScaleTemperature


@torch.no_grad()
def evaluate_mse(dataloader: DataLoader, backward_sampler: ConditionalBackwardSampler, max_time_steps: int, temperature_scaler: ScaleTemperature) -> float:
    total_mse = 0.
    nb_batches = len(dataloader)

    for X, y in tqdm(dataloader, total=nb_batches):
        # First generate noisy sst from normal distribution.
        noisy_sst = torch.randn_like(y)

        # Feed the noisy sst into backward sampler to get original sst.
        for t in reversed(range(max_time_steps)):
            noisy_sst = backward_sampler(noisy_sst, X, t)

        # Convert the predicted and y temperature into normal range.
        y = temperature_scaler.inverse(y)
        noisy_sst = torch.clip(noisy_sst, -1.5, 1.5)
        pred_sst = temperature_scaler.inverse(noisy_sst.cpu())

        # We then compare the predicted sst with the original sst.
        total_mse += float(F.mse_loss(pred_sst, y).item())

    # Return the average MSE.
    return total_mse / nb_batches
