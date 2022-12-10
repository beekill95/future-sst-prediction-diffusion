import torch.nn.functional as F


def mse_loss(pred_noise, target_noise):
    return F.mse_loss(pred_noise, target_noise)


def smooth_l1_loss(pred_noise, target_noise):
    return F.smooth_l1_loss(pred_noise, target_noise)
