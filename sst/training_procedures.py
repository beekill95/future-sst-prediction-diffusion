from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from .dataset import NOAA_OI_SST
from .forward_sampler import ForwardSampler
from .losses import mse_loss


class UnconditionalTrainingProcedure:
    def __init__(self, model: nn.Module, forward_sampler: ForwardSampler, device: str, lr=1e-3) -> None:
        self._model = model.to(device)
        self._forward_process = forward_sampler
        self._device = device

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        model.apply(_init_weights)

    def train(self, dataloader: DataLoader[NOAA_OI_SST], epoch: int) -> float:
        self._model.train()

        nb_batches = len(dataloader)
        total_loss = 0.
        for _, sst in tqdm(dataloader, total=nb_batches, desc=f'Training epoch {epoch}'):
            # Since this is an unconditional training procedure,
            # we just have to use the target y.

            loss = self._forward_step(sst)
            total_loss += float(loss.item())

            # Perform gradient descent.
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        return total_loss / nb_batches

    def _forward_step(self, sst: torch.Tensor):
        # Generate noisy SST.
        batch_size = sst.shape[0]
        random_time_steps = torch.randint(0, self._forward_process.max_time_steps, size=(batch_size,))
        noisy_sst, added_noise = self._forward_process(sst, random_time_steps)

        # Move the tensors to the desired device.
        noisy_sst, added_noise = noisy_sst.to(self._device), added_noise.to(self._device)
        random_time_steps = random_time_steps.to(self._device)

        # Feed the noisy SSTs to model to get back predicted noise.
        pred_noise = self._model(noisy_sst, random_time_steps)

        # Calculate the loss.
        loss = mse_loss(pred_noise, added_noise)
        return loss


class ConditionalOnPastSSTTrainingProcedure:
    def __init__(self, model: nn.Module, forward_sampler: ForwardSampler, device: str, lr=1e-3) -> None:
        self._model = model.to(device)
        self._forward_process = forward_sampler
        self._device = device

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        model.apply(_init_weights)

    def train(self, dataloader: DataLoader[NOAA_OI_SST], epoch: int) -> float:
        self._model.train()

        nb_batches = len(dataloader)
        total_loss = 0.
        for past_sst, sst in tqdm(dataloader, total=nb_batches, desc=f'Training epoch {epoch}'):
            loss = self._forward_step(sst, past_sst)
            total_loss += float(loss.item())

            # Perform gradient descent.
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        return total_loss / nb_batches

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader[NOAA_OI_SST], progress: bool = False) -> float:
        self._model.eval()

        nb_batches = len(dataloader)
        total_loss = 0.
        dataiter = (tqdm(dataloader, total=nb_batches, desc=f'Evaluation')
                    if progress
                    else dataloader)
        for past_sst, sst in dataiter:
            loss = self._forward_step(sst, past_sst)
            total_loss += float(loss.item())

        return total_loss / nb_batches

    def _forward_step(self, sst: torch.Tensor, past_sst: torch.Tensor):
        # Generate noisy SST.
        batch_size = sst.shape[0]
        random_time_steps = torch.randint(0, self._forward_process.max_time_steps, size=(batch_size,))
        noisy_sst, added_noise = self._forward_process(sst, random_time_steps)

        # Move the tensors to the desired device.
        noisy_sst, added_noise = noisy_sst.to(self._device), added_noise.to(self._device)
        random_time_steps = random_time_steps.to(self._device)
        past_sst = past_sst.to(self._device)

        # Feed the noisy SSTs to model to get back predicted noise.
        pred_noise = self._model(
            noisy_sst=noisy_sst,
            past_sst=past_sst,
            time_steps=random_time_steps)

        # Calculate the loss.
        loss = mse_loss(pred_noise, added_noise)
        return loss


def _init_weights(m: nn.Module):
    strategy_fn = nn.init.xavier_normal_

    match m:
        case nn.Linear:
            strategy_fn(m.weight)
        case _:
            print(f'Skipped init weight for module {m=}.')
